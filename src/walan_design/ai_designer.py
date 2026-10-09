"""
AI 花型设计模块
多供应商架构：Stability API / DALL-E 3 / 本地 SD WebUI
生成花型图片 → 接回位 → 超分辨率放大 → 一花四色
"""

import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from walan_design.providers import GenerationParams, ImageProvider
from walan_design.upscaler import upscale_image

logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_placeholder(brief: dict, config: dict) -> list[Image.Image]:
    """SD 不可用时的占位图 — 生成有简单纹理的假花型"""
    common = config["design"].get("common", {})
    w, h = 1024, 1536  # 2:3 比例

    images = []
    batch = common.get("batch_size", 1)
    for _ in range(batch):
        arr = np.random.randint(180, 240, (h, w, 3), dtype=np.uint8)
        for _ in range(30):
            cx = np.random.randint(20, w - 20)
            cy = np.random.randint(20, h - 20)
            r = np.random.randint(5, 25)
            color = tuple(np.random.randint(40, 140, 3).tolist())
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
            arr[mask] = color
        images.append(Image.fromarray(arr))

    logger.info(f"生成 {len(images)} 张占位图: {brief.get('title', '')}")
    return images


def _shift_hsv(arr: np.ndarray, hue_shift: float = 0.0, sat_scale: float = 1.0, val_scale: float = 1.0) -> np.ndarray:
    """在 HSV 空间对图片做颜色变换"""
    img = Image.fromarray(arr.astype(np.uint8))
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    hsv[..., 0] = (hsv[..., 0] / 255.0 + hue_shift) % 1.0 * 255
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
    rgb = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
    return np.array(rgb)


def generate_color_variants(base_img: Image.Image, count: int = 4, dark_count: int = 2) -> list[Image.Image]:
    """
    一花四色（PDF 强制标准：2深底 + 2浅底，色相有明显区别）。
    用 HSV 变换在同一张图上生成不同配色，保持花型结构不变。
    """
    arr = np.array(base_img).astype(np.float32)
    variants = []

    dark_presets = [
        {"hue": 0.05, "sat": 0.7, "val": 0.55},
        {"hue": -0.08, "sat": 0.6, "val": 0.5},
    ]
    light_presets = [
        {"hue": -0.02, "sat": 0.8, "val": 1.25},
        {"hue": 0.08, "sat": 0.7, "val": 1.3},
    ]

    all_presets = dark_presets[:dark_count] + light_presets[: count - dark_count]

    for preset in all_presets[:count]:
        variant_arr = _shift_hsv(
            arr,
            hue_shift=preset["hue"],
            sat_scale=preset["sat"],
            val_scale=preset["val"],
        )
        variants.append(Image.fromarray(variant_arr.astype(np.uint8)))

    return variants


def make_seamless(img: Image.Image) -> Image.Image:
    """快速版四方连续：numpy 加速的边缘镜像融合"""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    blend = max(w // 16, 32)

    left = arr[:, :blend, :]
    right_flipped = arr[:, w - blend :, :][:, ::-1, :]
    for x in range(blend):
        # alpha 随 x 从 1 → 0：左边缘被镜像右边缘完全替换（保证 col0 == col(w-1)），
        # 越往内越保留原图；反方向会让接缝差异原样保留（见 quality check FAIL 复盘）
        alpha = 1 - x / blend
        arr[:, x, :] = left[:, x, :] * (1 - alpha) + right_flipped[:, x, :] * alpha

    top = arr[:blend, :, :]
    bottom_flipped = arr[h - blend :, :, :][::-1, :, :]
    for y in range(blend):
        alpha = 1 - y / blend
        arr[y, :, :] = top[y, :, :] * (1 - alpha) + bottom_flipped[y, :, :] * alpha

    return Image.fromarray(arr.astype(np.uint8))


def run(briefs: list, config: dict = None) -> list:
    if config is None:
        config = load_config()

    design_cfg = config["design"]
    engine = design_cfg["engine"]
    common = design_cfg.get("common", {})
    output_dir = Path(design_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化供应商（传入 design 子树：各供应商的配置块都在 design.* 命名空间下）
    provider = None
    try:
        provider = ImageProvider.create(engine, design_cfg)
        if provider.health_check():
            logger.info(f"✓ 供应商 {provider.name} 可用")
        else:
            logger.warning(f"供应商 {engine} 不可用，将使用占位图")
            provider = None
    except Exception as e:
        logger.warning(f"供应商初始化失败: {e}，将使用占位图")

    # 放大配置
    upscaler_cfg = config.get("upscaler", {})
    upscale_method = upscaler_cfg.get("method", "lanczos")
    target_dpi = upscaler_cfg.get("target_dpi", 300)

    results = []
    for i, brief in enumerate(briefs):
        title = brief.get("title", f"design_{i + 1}")
        logger.info(f"\n=== 设计 {i + 1}/{len(briefs)}: {title} ===")

        # 1. AI 生图
        images = []
        if provider is not None:
            sd_cfg = design_cfg.get("stable_diffusion", {})
            neg_prompt = common.get("negative_prompt", sd_cfg.get("negative_prompt", ""))
            params = GenerationParams(
                prompt=brief.get("sd_prompt", ""),
                negative_prompt=neg_prompt,
                width=design_cfg.get("stable_diffusion", {}).get("width", 1024),
                height=design_cfg.get("stable_diffusion", {}).get("height", 1536),
                batch_size=common.get("batch_size", 1),
                steps=design_cfg.get("stable_diffusion", {}).get("steps", 30),
                cfg_scale=design_cfg.get("stable_diffusion", {}).get("cfg_scale", 7.5),
                tiling=common.get("emphasize_tiling", True),
            )
            try:
                result = provider.generate(params)
                images = result.images
                logger.info(f"  {provider.name} 直出: {len(images)} 张, {images[0].size if images else 'N/A'}")
            except Exception as e:
                logger.error(f"  供应商生成失败: {e}")

        if not images:
            logger.warning("使用占位图")
            images = generate_placeholder(brief, config)

        # 2. 超分辨率放大
        logger.info(f"  超分辨率放大 (method={upscale_method}, target_dpi={target_dpi})...")
        upscaled = []
        for img in images:
            try:
                up = upscale_image(img, target_dpi=target_dpi, method=upscale_method)
                upscaled.append(up)
                logger.info(f"    {img.width}×{img.height} → {up.width}×{up.height}")
            except Exception as e:
                logger.warning(f"  放大失败 ({e})，使用原图")
                upscaled.append(img)

        # 3. 接回位（必须在放大之后：LANCZOS 插值会重新引入边缘差异，
        # 放大前融合的接缝到最终图上会退化，见 quality check seamless FAIL 复盘）
        if config["psd"].get("auto_seamless", True):
            logger.info("  处理四方连续接回位...")
            try:
                upscaled = [make_seamless(img) for img in upscaled]
            except Exception as e:
                logger.warning(f"  接回位处理失败: {e}")

        # 4. 一花四色
        color_count = design_cfg.get("color_variant_count", 4)
        dark_count = design_cfg.get("color_variant_mix", 2)
        base_image = upscaled[0] if upscaled else images[0]
        logger.info(f"  生成一花四色变体: {color_count} 个（{dark_count}深底 + {color_count - dark_count}浅底）")
        try:
            color_variants = generate_color_variants(base_image, count=color_count, dark_count=dark_count)
            all_images = upscaled + color_variants
        except Exception as e:
            logger.warning(f"  配色变体生成失败: {e}")
            all_images = upscaled

        # 5. 保存
        image_paths = []
        dpi = config["psd"]["dpi"]
        for j, img in enumerate(all_images):
            fp = output_dir / f"{title}_{j + 1}.png"
            img.save(str(fp), "PNG", dpi=(dpi, dpi))
            image_paths.append(str(fp))
            logger.info(f"  保存: {fp} ({img.width}×{img.height})")

        results.append(
            {
                "brief": brief,
                "image_paths": image_paths,
                "base_image_count": len(upscaled),
                "color_variant_count": len(all_images) - len(upscaled),
                "provider": provider.name if provider else "placeholder",
            }
        )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    config = load_config()
    from walan_design.trend_collector import _local_fallback_briefs

    briefs = _local_fallback_briefs(config["trend"]["brief_count"], config["design"]["style_preferences"])
    results = run(briefs, config)
    print(f"\n生成完成，共 {len(results)} 个设计")
