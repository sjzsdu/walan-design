"""
超分辨率放大模块

AI 直出分辨率（~1-4 MP）远低于瓦栏要求（~15-33 MP），
需要用超分辨率算法放大到目标尺寸。

方案:
  1. 优先用 Real-ESRGAN（本地推理，质量最好）
  2. 回退用 PIL LANCZOS（简单但快，质量一般）
  3. 未来可接入 Stability AI 的 upscaling API
"""

import logging

import numpy as np
from PIL import Image

from .rules import cm_to_px

logger = logging.getLogger(__name__)


def get_target_resolution(dpi: int = 300) -> tuple[int, int]:
    """获取瓦栏目标分辨率（40cm × 60cm @ DPI）"""
    return (cm_to_px(40, dpi), cm_to_px(60, dpi))


def upscale_pil(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    PIL LANCZOS 放大（回退方案，不依赖额外库）。
    质量一般但速度快，作为 ESRGAN 不可用时的回退。
    """
    logger.info(f"  LANCZOS 放大: {img.width}×{img.height} → {target_w}×{target_h}")
    return img.resize((target_w, target_h), Image.LANCZOS)


def upscale_real_esrgan(img: Image.Image, target_w: int, target_h: int, scale: int = 4) -> Image.Image:
    """
    Real-ESRGAN 超分辨率放大。
    需要 pip install realesrgan 或 realesrgan-ncnn-vulkan 二进制。

    Real-ESRGAN 擅长放大 4× 并保持细节，适合花型。
    """
    try:
        # 尝试使用 realesrgan Python 包
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # 初始化模型（第一次加载较慢）
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path="realesrgan/RealESRGAN_x4plus.pth",
            model=model,
            tile=512,  # 分块处理，降低显存需求
            tile_pad=10,
            pre_pad=0,
            half=True,  # 半精度推理，Mac Metal 兼容
        )
        arr = np.array(img.convert("RGB"))
        output, _ = upsampler.enhance(arr)
        result = Image.fromarray(output)

        # 如果还不够大，再用 LANCZOS 补到目标尺寸
        if result.width < target_w or result.height < target_h:
            result = result.resize((target_w, target_h), Image.LANCZOS)

        logger.info(f"  Real-ESRGAN 放大: {img.width}×{img.height} → {result.width}×{result.height}")
        return result

    except ImportError:
        logger.warning("  Real-ESRGAN 未安装，回退到 LANCZOS")
        return upscale_pil(img, target_w, target_h)
    except Exception as e:
        logger.warning(f"  Real-ESRGAN 失败 ({e})，回退到 LANCZOS")
        return upscale_pil(img, target_w, target_h)


def upscale_image(
    img: Image.Image,
    target_dpi: int = 300,
    method: str = "auto",
) -> Image.Image:
    """
    超分辨率放大到瓦栏目标尺寸。

    method:
      - auto: 先试 ESRGAN，失败回退 LANCZOS
      - esrgan: 只用 Real-ESRGAN（需安装）
      - lanczos: 只用 PIL LANCZOS（快速但质量一般）
    """
    target_w, target_h = get_target_resolution(target_dpi)

    # 如果原图已经够大，不需要放大
    if img.width >= target_w and img.height >= target_h:
        logger.info(f"  原图已达标: {img.width}×{img.height} ≥ {target_w}×{target_h}")
        return img

    if method == "esrgan":
        return upscale_real_esrgan(img, target_w, target_h)
    elif method == "lanczos":
        return upscale_pil(img, target_w, target_h)
    else:  # auto
        return upscale_real_esrgan(img, target_w, target_h)
