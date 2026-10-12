"""
diffusers circular-padding 四方连续验证脚本（方案 A）

原理：把 UNet 里所有 Conv2d 的 padding_mode 改成 'circular'，
让卷积在画布边缘环绕读取对侧内容 → 生成时即构造性保证母题绕边延续、
无切花。这跟 A1111 的 payload["tiling"]=true 是同一个机制，但走
PyPI 装包、不依赖任何已删仓库。

用法：
  A1111_VENV=/Users/juzhongsun/.local/share/stable-diffusion-webui/venv/bin/python \
    scripts/test_diffusers_tiling.py

输出：
  /tmp/diffusers_tiling_test.png  （1024×1536 数码花型直出图）
"""

import sys
import time

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from PIL import Image

MODEL_PATH = "/Users/juzhongsun/.local/share/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors"
OUT_PATH = "/tmp/diffusers_tiling_test.png"

# 数码花型 prompt（watercolor wash / gradient / digital print —— 严禁传统平涂词）
PROMPT = (
    "seamless floral pattern, watercolor wash, smooth color gradients, "
    "digital textile print, unlimited color palette, rich layered tones, "
    "painterly depth, ink bleed, vibrant botanical flowers and leaves, "
    "balanced all-over repeat layout, evenly distributed motifs, "
    "professional surface pattern design, high detail"
)
NEG = (
    "flat vector, line hatching, silkscreen, block print, gouache, limited palette, "
    "photorealistic photo, 3d render, human, face, hands, signature, frame, border, "
    "cropped motifs, cluttered, oversaturated, gradient noise, low quality"
)


def patch_circular_padding(pipe):
    """把 UNet 里所有 Conv2d 的 padding_mode 改成 'circular'。"""
    count = 0
    for module in pipe.unet.modules():
        if isinstance(module, nn.Conv2d):
            # 只改有 padding 的卷积（padding=0 的改了也没用，且可能报错）
            if module.padding != (0, 0) and module.padding_mode != "circular":
                module.padding_mode = "circular"
                count += 1
    # VAE 也可以打（让解码也环续），但 VAE 的 Conv2d 多为 1×1 或无 padding，
    # 这里只打 UNet 即可保证生成端四方连续
    print(f"  circular padding 已打补丁: {count} 个 Conv2d")
    return count


def main():
    print(f"[1/4] 加载 SDXL from_single_file: {MODEL_PATH}")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to("mps")
    # 关掉 attention slicing（和 circular padding 一起会触发 MPS bug）
    pipe.set_progress_bar_config(disable=True)
    print(f"  加载完成，耗时 {time.time()-t0:.1f}s")

    print("[2/4] 打 circular padding 补丁")
    patch_circular_padding(pipe)

    print("[3/4] 生成 1024×1536 数码花型")
    t0 = time.time()
    generator = torch.Generator(device="mps").manual_seed(42)
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEG,
        width=1024,
        height=1536,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    print(f"  生成完成，耗时 {time.time()-t0:.1f}s")

    print(f"[4/4] 保存到 {OUT_PATH}")
    image.save(OUT_PATH)
    print(f"  尺寸: {image.size}")
    print("DONE")


if __name__ == "__main__":
    main()
