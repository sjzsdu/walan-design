"""
diffusers 本地生图服务（常驻 HTTP，模型只加载一次）

用 A1111 venv 的 Python 跑（有 torch 2.3.1 + MPS）：
  /Users/juzhongsun/.local/share/stable-diffusion-webui/venv/bin/python \
    scripts/diffusers_server.py

API: POST http://127.0.0.1:7861/generate
  Body (JSON): {prompt, negative_prompt, width, height, steps, cfg_scale, seed}
  Response (JSON): {image_path}  （PNG 保存到 /tmp/diffusers_out/）

健康检查: GET http://127.0.0.1:7861/health  → {status: ok}

核心：UNet 所有 Conv2d 的 padding_mode 改成 'circular'，
让卷积在画布边缘环绕读取对侧 → 生成即构造性四方连续。
"""

import json
import gc
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline

# 模型注册表：key → (路径, 加载方式)
# juggernaut = RunDiffusion Juggernaut XL v9（美学精调，2026-10-12 提质量换装）
# base       = SDXL base 1.0（原始底座，留作对比/回退）
MODELS = {
    "juggernaut": (
        os.path.expanduser("~/.local/share/walan-models/juggernaut-xl-v9"),
        "pretrained",
    ),
    "base": (
        "/Users/juzhongsun/.local/share/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors",
        "single_file",
    ),
}
DEFAULT_MODEL = os.environ.get("DIFFUSERS_DEFAULT_MODEL", "juggernaut")
OUT_DIR = "/tmp/diffusers_out"
PORT = 7861

# 单缓存：16GB 统一内存装不下两个 fp16 SDXL，切换时卸载旧的
_pipe = None
_pipe_key = None
_load_time = 0.0

# 生成锁：pipe() 不是线程安全的，确保同一时间只有一个推理请求
_gen_lock = threading.Lock()


def _apply_circular_padding(pipe) -> int:
    """UNet + VAE 都要打 circular padding：
    只改 UNet 时 latent 边缘连续但 VAE 解码 latent→像素时边缘仍断裂（实测比值 6.7）；
    UNet+VAE 都改后边缘连续性比值 0.52/0.66（完美四方连续）"""
    count = 0
    for component in (pipe.unet, pipe.vae):
        for module in component.modules():
            if isinstance(module, nn.Conv2d):
                if module.padding != (0, 0) and module.padding_mode != "circular":
                    module.padding_mode = "circular"
                    count += 1
    return count


def get_pipe(model_key: str = None):
    global _pipe, _pipe_key, _load_time
    key = model_key if model_key in MODELS else DEFAULT_MODEL
    if _pipe is not None and _pipe_key == key:
        return _pipe
    # 卸载旧模型释放 MPS 内存
    if _pipe is not None:
        print(f"[diffusers_server] 卸载 {_pipe_key}，切换到 {key}", flush=True)
        _pipe = None
        _pipe_key = None
        gc.collect()
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
    t0 = time.time()
    path, how = MODELS[key]
    print(f"[diffusers_server] 加载 {key}: {path}", flush=True)
    if how == "pretrained":
        pipe = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(
            path, torch_dtype=torch.float16, use_safetensors=True
        )
    pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    _pipe, _pipe_key = pipe, key
    _load_time = time.time() - t0
    count = _apply_circular_padding(_pipe)
    print(f"[diffusers_server] 加载完成 {_load_time:.1f}s, circular padding: {count} Conv2d", flush=True)
    return _pipe


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._json({
                "status": "ok",
                "model_loaded": _pipe is not None,
                "current_model": _pipe_key,
                "default_model": DEFAULT_MODEL,
                "available_models": list(MODELS.keys()),
                "load_time": _load_time,
            })
        else:
            self._json({"status": "running"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/generate":
            self._json({"error": "not found"}, 404)
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            req = json.loads(body)
            prompt = req.get("prompt", "")
            neg = req.get("negative_prompt", "")
            w = int(req.get("width", 1024))
            h = int(req.get("height", 1536))
            steps = int(req.get("steps", 30))
            cfg = float(req.get("cfg_scale", 7.5))
            seed = int(req.get("seed", -1))
            model_key = req.get("model") or None

            pipe = get_pipe(model_key)

            # 用生成锁保护 pipe() 调用 —— diffusers 推理不是线程安全的
            _gen_lock.acquire()
            try:
                gen = torch.Generator(device="mps")
                if seed >= 0:
                    gen.manual_seed(seed)

                t0 = time.time()
                print(f"[diffusers_server] 生成 {w}×{h} steps={steps} seed={seed}", flush=True)
                img = pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    width=w,
                    height=h,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=gen,
                ).images[0]
                elapsed = time.time() - t0

                os.makedirs(OUT_DIR, exist_ok=True)
                path = os.path.join(OUT_DIR, f"out_{int(time.time()*1000)}.png")
                img.save(path)
                print(f"[diffusers_server] 完成 {elapsed:.1f}s → {path}", flush=True)
                self._json({"image_path": path, "elapsed_s": round(elapsed, 1), "size": list(img.size), "model": _pipe_key})
            finally:
                _gen_lock.release()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json({"error": str(e)}, 500)

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[{self.client_address[0]}] {fmt % args}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[diffusers_server] 监听 http://127.0.0.1:{PORT}（ThreadingHTTPServer）", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.serve_forever()
