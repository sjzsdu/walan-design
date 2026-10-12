"""
修复验证：UNet + VAE 都打 circular padding，快速生成验证边缘连续性
小分辨率 512×768 + 10 步，M2 MPS 上约 5 分钟
"""

import time

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np

MODEL = "/Users/juzhongsun/.local/share/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors"
OUT = "/tmp/diffusers_tiling_v2.png"

PROMPT = (
    "seamless floral pattern, watercolor wash, smooth color gradients, "
    "digital textile print, vibrant botanical flowers, "
    "balanced all-over repeat layout"
)
NEG = "flat vector, line hatching, silkscreen, human, face, frame, border, low quality"


def patch_circular(module, label):
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if m.padding != (0, 0) and m.padding_mode != "circular":
                m.padding_mode = "circular"
                count += 1
    print(f"  {label}: {count} Conv2d → circular", flush=True)
    return count


def edge_continuity(img):
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    # 水平
    h_edge = np.mean(np.abs(arr[:, 0, :] - arr[:, -1, :]))
    h_base = np.mean(np.abs(arr[:, :-1, :] - arr[:, 1:, :]))
    h_ratio = h_edge / (h_base + 1e-6)
    # 垂直
    v_edge = np.mean(np.abs(arr[0, :, :] - arr[-1, :, :]))
    v_base = np.mean(np.abs(arr[:-1, :, :] - arr[1:, :, :]))
    v_ratio = v_edge / (v_base + 1e-6)
    print(f"\n=== 边缘连续性 ===")
    print(f"水平: 边缘={h_edge:.1f} 基线={h_base:.1f} 比值={h_ratio:.3f} {'✓' if h_ratio < 1.5 else '✗'}")
    print(f"垂直: 边缘={v_edge:.1f} 基线={v_base:.1f} 比值={v_ratio:.3f} {'✓' if v_ratio < 1.5 else '✗'}")
    print(f"判定: {'✓ 四方连续' if h_ratio < 1.5 and v_ratio < 1.5 else '✗ 边缘断裂'}")
    # 2×2 平铺
    tiled = Image.new(img.mode, (w * 2, h * 2))
    for dy in range(2):
        for dx in range(2):
            tiled.paste(img, (dx * w, dy * h))
    tiled.save("/tmp/diffusers_tiling_v2_2x2.png")
    print(f"2×2 平铺: /tmp/diffusers_tiling_v2_2x2.png")


def main():
    print("[1/4] 加载 SDXL", flush=True)
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_single_file(MODEL, torch_dtype=torch.float16)
    pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    print(f"  完成 {time.time()-t0:.1f}s", flush=True)

    print("[2/4] 打 circular padding（UNet + VAE）", flush=True)
    patch_circular(pipe.unet, "unet")
    patch_circular(pipe.vae, "vae")

    print("[3/4] 生成 512×768 10步", flush=True)
    t0 = time.time()
    gen = torch.Generator(device="mps").manual_seed(42)
    img = pipe(
        prompt=PROMPT,
        negative_prompt=NEG,
        width=512,
        height=768,
        num_inference_steps=10,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    print(f"  完成 {time.time()-t0:.1f}s", flush=True)

    print(f"[4/4] 保存 + 验证", flush=True)
    img.save(OUT)
    print(f"  保存: {OUT} {img.size}", flush=True)
    edge_continuity(img)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
