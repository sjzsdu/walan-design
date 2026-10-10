"""
AI 花型设计模块
多供应商架构：Stability API / DALL-E 3 / 本地 SD WebUI
生成花型图片 → AI 视觉自评（低分重生成）→ 超分辨率放大 → 接回位 → 一花四色（专业调色板）
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


def _generate_with_review(
    provider,
    brief: dict,
    config: dict,
    sd_cfg: dict,
    common: dict,
) -> tuple[list, list]:
    """
    生图 + 视觉自评闭环。
    返回 (images, critiques)：critiques 记录每轮评分过程。
    """
    from walan_design.prompt_engine import build_negative_prompt, build_sd_prompt
    from walan_design.visual_critic import critique_image

    vcfg = config.get("design", {}).get("visual_review", {})
    enabled = vcfg.get("enabled", False)
    max_retries = int(vcfg.get("max_retries", 2)) if enabled else 0

    base_prompt = build_sd_prompt(brief, config)
    neg_prompt = build_negative_prompt(config)

    images, critiques = [], []
    prompt = base_prompt
    for attempt in range(max_retries + 1):
        params = GenerationParams(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=sd_cfg.get("width", 1024),
            height=sd_cfg.get("height", 1536),
            batch_size=common.get("batch_size", 1),
            steps=sd_cfg.get("steps", 30),
            cfg_scale=sd_cfg.get("cfg_scale", 7.5),
            tiling=common.get("emphasize_tiling", True),
        )
        result = provider.generate(params)
        images = result.images
        logger.info(f"  {provider.name} 直出: {len(images)} 张, {images[0].size if images else 'N/A'}")

        if not enabled or not images:
            break

        critique = critique_image(images[0], brief, config)
        critique["attempt"] = attempt + 1
        critique["prompt"] = prompt
        critiques.append(critique)

        if critique["passed"]:
            break
        if attempt < max_retries and critique.get("suggestions"):
            prompt = f"{base_prompt}, {critique['suggestions']}"
            logger.info(f"  低于阈值，带改进建议重生成 (第 {attempt + 2} 次): {prompt[:100]}...")

    return images, critiques


def run(briefs: list, config: dict = None) -> list:
    if config is None:
        config = load_config()

    design_cfg = config["design"]
    engine = design_cfg["engine"]
    common = design_cfg.get("common", {})
    output_dir = Path(design_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sd_cfg = design_cfg.get("stable_diffusion", {})

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

        # 1. AI 生图 + 视觉自评闭环
        images, critiques = [], []
        if provider is not None:
            try:
                images, critiques = _generate_with_review(provider, brief, config, sd_cfg, common)
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

        # 4. 一花四色（专业调色板重着色）
        from walan_design.recolor import generate_color_variants

        color_count = design_cfg.get("color_variant_count", 4)
        dark_count = design_cfg.get("color_variant_mix", 2)
        palette_cfg = design_cfg.get("recolor", {}).get("palettes")
        base_image = upscaled[0] if upscaled else images[0]
        logger.info(f"  生成一花四色变体: {color_count} 个（{dark_count}深底 + {color_count - dark_count}浅底）")
        try:
            color_variants = generate_color_variants(
                base_image,
                count=color_count,
                dark_count=dark_count,
                palette_cfg=palette_cfg,
                cluster_count=int(design_cfg.get("recolor", {}).get("cluster_count", 8)),
            )
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
                "visual_review": critiques,  # 自评记录（含最终得分与建议）
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
