"""
正确的 circular-padding 验证：边缘像素差异

circular padding 保证的是：UNet 卷积在画布边缘环绕读取对侧内容 →
生成的图像边缘像素"连续"：img 右边缘和左边缘能接上（平铺无突变）。

验证方法：
  - 水平边缘差异 = mean(|img[:, -1] - img[:, 0]|)
  - 垂直边缘差异 = mean(|img[-1, :] - img[0, :]|)
  - 基线（相邻像素差异）= mean(|img[:, :-1] - img[:, 1:]|)
  - 比值 = 边缘差异 / 基线
  - 比值≈1 = 边缘连续（circular padding 生效）
  - 比值>>1 = 边缘断裂（circular padding 没生效或图本身边缘不连续）
"""

import numpy as np
from PIL import Image


def edge_continuity(img_path):
    img = Image.open(img_path)
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    print(f"图片: {img.size} ({w}×{h})\n")

    # 水平方向（左右边缘）
    left = arr[:, 0, :]
    right = arr[:, -1, :]
    h_edge = np.mean(np.abs(left - right))
    h_baseline = np.mean(np.abs(arr[:, :-1, :] - arr[:, 1:, :]))
    h_ratio = h_edge / (h_baseline + 1e-6)

    # 垂直方向（上下边缘）
    top = arr[0, :, :]
    bottom = arr[-1, :, :]
    v_edge = np.mean(np.abs(top - bottom))
    v_baseline = np.mean(np.abs(arr[:-1, :, :] - arr[1:, :, :]))
    v_ratio = v_edge / (v_baseline + 1e-6)

    print(f"水平（左右边缘）:")
    print(f"  边缘差异 = {h_edge:.2f}")
    print(f"  相邻基线 = {h_baseline:.2f}")
    print(f"  比值 = {h_ratio:.3f}  {'✓ 连续' if h_ratio < 1.5 else '✗ 断裂'}")
    print()
    print(f"垂直（上下边缘）:")
    print(f"  边缘差异 = {v_edge:.2f}")
    print(f"  相邻基线 = {v_baseline:.2f}")
    print(f"  比值 = {v_ratio:.3f}  {'✓ 连续' if v_ratio < 1.5 else '✗ 断裂'}")
    print()
    print(f"判定: {'✓ circular padding 生效（四方连续）' if h_ratio < 1.5 and v_ratio < 1.5 else '✗ 边缘断裂（未达四方连续）'}")


if __name__ == "__main__":
    edge_continuity("/tmp/diffusers_tiling_test.png")
