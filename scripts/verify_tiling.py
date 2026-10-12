"""验证 circular-padding 生图的四方连续性：周期残差比 + 2×2 平铺目检"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from walan_design.seam_quilt import find_periodic_tile

IMG = "/tmp/diffusers_tiling_test.png"
TILED_OUT = "/tmp/diffusers_tiling_2x2.png"


def make_2x2_tiled(img):
    """2×2 平铺，目检切花"""
    w, h = img.size
    tiled = Image.new(img.mode, (w * 2, h * 2))
    tiled.paste(img, (0, 0))
    tiled.paste(img, (w, 0))
    tiled.paste(img, (0, h))
    tiled.paste(img, (w, h))
    tiled.save(TILED_OUT)
    print(f"2×2 平铺保存: {TILED_OUT} ({w*2}×{h*2})")


def main():
    img = Image.open(IMG)
    print(f"原图: {img.size}")

    print("\n=== 周期残差比验证 ===")
    print("（ratio 越接近 1 = 越接近严格四方连续；云端 SDXL 无 tiling 图为 1.68-6.3）")
    tile, info = find_periodic_tile(img)
    print(f"  结果: {info}")
    if tile is not None:
        print(f"  ✓ 命中周期窗口: {tile.size}")
        tile.save("/tmp/diffusers_periodic_tile.png")
    else:
        print(f"  ✗ 未命中周期窗口（残差比 {info.get('ratio_x'):.3f}/{info.get('ratio_y'):.3f}，需 ≤{info.get('accept_ratio', 2.0)}）")

    print("\n=== 2×2 平铺目检 ===")
    make_2x2_tiled(img)
    print(f"\n请打开 {TILED_OUT} 目检：母题是否绕边延续、有无切花")


if __name__ == "__main__":
    main()
