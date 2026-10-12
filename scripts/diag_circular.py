"""
诊断 circular padding：
1. MPS 上 nn.Conv2d(padding_mode='circular') 是否真的环绕
2. SDXL UNet + VAE 各有多少 padding>0 的 Conv2d
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline

MODEL = "/Users/juzhongsun/.local/share/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors"


def test_circular_on_mps():
    """最小验证：一个 3×3 Conv2d padding_mode=circular，输入左白右黑，看输出"""
    dev = "mps"
    conv = nn.Conv2d(1, 1, 3, padding=1, padding_mode="circular", bias=False).to(dev)
    # 权重全 1（做局部平均）
    with torch.no_grad():
        conv.weight.fill_(1.0)
    # 输入 1×1×4×4：左边两列白(1)，右边两列黑(0)
    x = torch.zeros(1, 1, 4, 4, device=dev)
    x[:, :, :, :2] = 1.0
    out = conv(x)  # (1,1,4,4)
    # circular padding 生效时：out[0,0,0,2]（右边黑区第一列）会读到 x[0,0,0,0]（左边白）
    # 因为 3×3 卷积在 col=2 时会读 col=1,2,3，其中 col=3 是黑——但如果 circular，col=3=0 会被 col=0=1 替换
    # 实际上更直接的测试：看输出边缘是否反映环绕
    # out 的第一行第一列 = mean of x[0][*][0..1 and circular wrap]
    print(f"输入 (4×4, 左白右黑):\n{x[0,0].cpu().numpy()}")
    print(f"输出 (3×3 conv, circular padding):\n{out[0,0].detach().cpu().numpy()}")
    # 如果 circular 生效：输出右边缘列(out[:,-1]) 会受左边缘列(x[:,0]=1) 影响
    # 如果 zeros padding：输出右边缘列只受 x[:,-2:] 影响（都是 0）
    right_edge = out[0, 0, :, -1].detach().mean().item()
    print(f"\n输出右边缘列均值: {right_edge:.3f}")
    if right_edge > 0.3:
        print("✓ circular padding 生效（右边缘读到了左边缘的白色）")
    else:
        print("✗ circular padding 未生效（右边缘仍是 0，说明用的是 zeros padding）")


def count_convs():
    """统计 UNet + VAE 的 Conv2d"""
    pipe = StableDiffusionXLPipeline.from_single_file(MODEL, torch_dtype=torch.float16)
    for name, module in [("unet", pipe.unet), ("vae", pipe.vae)]:
        total = 0
        with_pad = 0
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                total += 1
                if m.padding != (0, 0):
                    with_pad += 1
        print(f"{name}: {total} Conv2d, {with_pad} 个有 padding>0")


if __name__ == "__main__":
    print("=== 1. MPS circular padding 最小验证 ===")
    test_circular_on_mps()
    print("\n=== 2. 统计 Conv2d ===")
    count_convs()
