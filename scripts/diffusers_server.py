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
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline

MODEL_PATH = "/Users/juzhongsun/.local/share/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors"
OUT_DIR = "/tmp/diffusers_out"
PORT = 7861

# 模块级单例：模型只加载一次
_pipe = None
_load_time = 0.0


def get_pipe():
    global _pipe, _load_time
    if _pipe is not None:
        return _pipe
    t0 = time.time()
    print(f"[diffusers_server] 加载 SDXL: {MODEL_PATH}", flush=True)
    _pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    _pipe.to("mps")
    _pipe.set_progress_bar_config(disable=True)
    _load_time = time.time() - t0
    print(f"[diffusers_server] 加载完成 {_load_time:.1f}s, 打 circular padding", flush=True)
    # UNet + VAE 都要打 circular padding：
    # 只改 UNet 时 latent 边缘连续但 VAE 解码 latent→像素时边缘仍断裂（实测比值 6.7）；
    # UNet+VAE 都改后边缘连续性比值 0.52/0.66（完美四方连续）
    count = 0
    for component in (_pipe.unet, _pipe.vae):
        for module in component.modules():
            if isinstance(module, nn.Conv2d):
                if module.padding != (0, 0) and module.padding_mode != "circular":
                    module.padding_mode = "circular"
                    count += 1
    print(f"[diffusers_server] circular padding: {count} Conv2d (UNet+VAE)", flush=True)
    return _pipe


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._json({"status": "ok", "model_loaded": _pipe is not None, "load_time": _load_time})
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

            pipe = get_pipe()
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
            self._json({"image_path": path, "elapsed_s": round(elapsed, 1), "size": list(img.size)})
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
    print(f"[diffusers_server] 监听 http://127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
