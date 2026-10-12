"""
AI 花型设计模块
多供应商架构：Stability API / DALL-E 3 / 本地 SD WebUI
生成花型图片 → AI 视觉自评（低分重生成）→ 超分辨率放大 → 接回位 → 一花四色（专业调色板）
"""

import json
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
    """四方连续接回位（方法由 config psd.seamless_method 注入）

    quilt（默认，2026-10-11 起）：相位错位 + 静区寻径拼接（seam_quilt，确定性算法）。
      两片相位差 Δ px 的拼接对差异≈纹理基线，DP 寻径绕开细节走静区，
      宽羽化兜底——零 AI、零 API 成本、确定可复现。替代"offset 硬切 + AI inpaint 修缝"
      （inpaint 是彩票：2026-10-11 实测 3/3 失败——六角形被涂成纯棕十字、小碎花发糊）。

    offset（旧）：四象限重排 + 硬切窄羽化，中央缝交给 seam_healer（AI inpaint）修复。
      接缝处内容级断裂仍可能出现，已不推荐。

    mirror：边缘镜像融合（数学上无缝但有镜像对称痕迹）。
    """
    method = _SEAMLESS_METHOD[0] if _SEAMLESS_METHOD[0] else "quilt"
    if method == "quilt":
        from walan_design.seam_quilt import quilt_offset

        return quilt_offset(img, skip_px=_SEAMLESS_SKIP[0])
    if method == "mirror":
        return _make_seamless_mirror(img)

    # ---- offset 四象限偏移 ----
    # 注意：w/h 可能是奇数（4724×7087 高为奇数），四个区域必须完整覆盖整图，
    # 否则最后一行/列不会被任何象限写入（表现为底边黑线、上下边缘差异巨大）
    w, h = img.size
    hw, hh = w // 2, h // 2
    result = Image.new(img.mode, (w, h))
    result.paste(img.crop((0, 0, hw, hh)), (w - hw, h - hh))   # 左上 → 右下
    result.paste(img.crop((hw, 0, w, hh)), (0, h - hh))        # 右上 → 左下
    result.paste(img.crop((0, hh, hw, h)), (w - hw, 0))        # 左下 → 右上
    result.paste(img.crop((hw, hh, w, h)), (0, 0))             # 右下 → 左上
    # 拼合缝位置：竖缝 x=w-hw，横缝 y=h-hh（奇数尺寸时偏离正中）

    # ---- 硬切 + 8px 窄羽化（无淡化带，无双影）----
    arr = np.array(result).astype(np.float32)
    sx, sy = w - hw, h - hh
    feather = max(1, min(8, sx - 1, sy - 1, w - sx - 1, h - sy - 1))
    for x in range(feather):
        a = 0.5 * (1 - x / feather)
        L, R = arr[:, sx - 1 - x, :], arr[:, sx + x, :]
        arr[:, sx - 1 - x, :] = L * (1 - a) + R * a
        arr[:, sx + x, :] = R * (1 - a) + L * a
    for y in range(feather):
        a = 0.5 * (1 - y / feather)
        T, B = arr[sy - 1 - y, :, :], arr[sy + y, :, :]
        arr[sy - 1 - y, :, :] = T * (1 - a) + B * a
        arr[sy + y, :, :] = B * (1 - a) + T * a

    return Image.fromarray(arr.astype(np.uint8))


# seamless_method 注入点（pipeline 里 set_seamless_method(config) 调用）
_SEAMLESS_METHOD = [None]
_SEAMLESS_SKIP = [6]


def set_seamless_method(config: dict):
    method = config.get("psd", {}).get("seamless_method", "quilt")
    _SEAMLESS_METHOD[0] = method if method in ("quilt", "offset", "mirror") else "quilt"
    try:
        _SEAMLESS_SKIP[0] = max(2, int(config.get("psd", {}).get("seam_skip_px", 6)))
    except (TypeError, ValueError):
        _SEAMLESS_SKIP[0] = 6


def _make_seamless_mirror(img: Image.Image) -> Image.Image:
    """旧版：边缘镜像融合（数学上边缘相等，但有镜像对称痕迹）"""
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

    # 放大配置
    upscaler_cfg = config.get("upscaler", {})
    upscale_method = upscaler_cfg.get("method", "lanczos")
    target_dpi = upscaler_cfg.get("target_dpi", 300)

    from walan_design.diagnostics import is_hard_failure, provider_blocked, record_attention

    # 断点续跑（设计级）：先扫一遍哪些设计已有完整产物（design_result.json 标记）。
    # 全部已有产物时不需要生图供应商 —— 余额耗尽也能把后续 PSD/质检/上传跑完。
    runs_dir = Path(config.get("pipeline", {}).get("runs_dir", "output/runs"))

    def _marker_result(brief: dict):
        title = brief.get("title", "")
        marker = runs_dir / title / "design_result.json"
        if not marker.exists():
            return None
        try:
            prev = json.loads(marker.read_text(encoding="utf-8"))
            if prev.get("image_paths") and all(Path(p).exists() for p in prev["image_paths"]):
                return prev
        except (json.JSONDecodeError, OSError):
            pass
        return None

    cached = {i: _marker_result(b) for i, b in enumerate(briefs)}
    need_generation = any(v is None for v in cached.values())

    # 初始化供应商（传入 design 子树：各供应商的配置块都在 design.* 命名空间下）
    provider = None
    allow_placeholder = design_cfg.get("allow_placeholder", False)
    init_err = ""
    if need_generation:
        try:
            provider = ImageProvider.create(engine, design_cfg)
            if provider.health_check():
                logger.info(f"✓ 供应商 {provider.name} 可用")
            else:
                init_err = getattr(provider, "health_reason", "") or f"供应商 {engine} 健康检查未通过"
                logger.error(f"✗ {init_err}")
                provider = None
        except Exception as e:
            init_err = f"{type(e).__name__}: {e}"
            logger.warning(f"供应商初始化失败: {e}")

        if provider is None and not allow_placeholder:
            # 硬故障直接中止：余额耗尽/key 失效时用假图跑完全批只会产出废品
            # （2026-10-11 事故：余额 $0.001 静默跑完 25 个废品 PSD）
            raise provider_blocked(engine, init_err)
    else:
        logger.info("全部设计已有完整产物（断点续跑），跳过生图供应商初始化")

    results = []
    failures = []  # 软故障记录 [{title, reason}]：单设计失败被跳过
    for i, brief in enumerate(briefs):
        title = brief.get("title", f"design_{i + 1}")
        logger.info(f"\n=== 设计 {i + 1}/{len(briefs)}: {title} ===")

        # 步骤级 checkpoint 只保证"design 步骤重跑"，这里保证步骤内每个设计不重做
        if cached.get(i) is not None:
            logger.info(f"  ↳ 断点续跑：已有完整产物（{len(cached[i]['image_paths'])} 张图），跳过生成")
            results.append(cached[i])
            continue

        # 1. AI 生图 + 视觉自评闭环
        images, critiques = [], []
        gen_err = ""
        if provider is not None:
            try:
                images, critiques = _generate_with_review(provider, brief, config, sd_cfg, common)
            except Exception as e:
                gen_err = f"{type(e).__name__}: {e}"
                if is_hard_failure(str(e)):
                    # 余额耗尽/key 失效/网络断：后续每个设计都会同样失败，立即中止
                    raise provider_blocked(engine, str(e)) from e
                logger.error(f"  供应商生成失败: {e}")

        if not images:
            if allow_placeholder:
                logger.warning("使用占位图（design.allow_placeholder=true，仅调试用）")
                images = generate_placeholder(brief, config)
            else:
                reason = f"AI 生图失败: {gen_err}" if gen_err else "AI 生图失败（未返回图片）"
                logger.error(f"  {reason}，跳过该设计")
                record_attention(config, f"生图:{title}", reason)
                failures.append({"title": title, "reason": reason})
                continue

        # 1.5 周期窗口裁剪（机会性治本）：SDXL 偶尔画出准周期纹样，以周期 p 裁窗
        #    可让 tile 天然严格平铺且母题完整；测不到周期就按整图走后续 quilt（fail-open）。
        #    2026-10-11 实测：SDXL 输出多数无平移周期性（4 图校准 0 命中），此步是免费彩票。
        if config["psd"].get("periodic_window", True):
            try:
                from walan_design.seam_quilt import find_periodic_tile

                kept = []
                for img in images:
                    tile, _info = find_periodic_tile(img)
                    kept.append(tile if tile is not None else img)
                images = kept
            except Exception as e:
                logger.warning(f"  周期窗口搜索失败({e})，按整图继续")

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
            set_seamless_method(config)
            logger.info(f"  处理四方连续接回位 (method={_SEAMLESS_METHOD[0]})...")
            try:
                upscaled = [make_seamless(img) for img in upscaled]
            except Exception as e:
                logger.warning(f"  接回位处理失败: {e}")

            # 3.5 SD inpaint 重绘接缝十字带（offset 硬切的内容级修复）。
            # 只处理 upscaled[0]（基准图）——四色变体由基准图重着色派生，自动继承修复。
            if (
                _SEAMLESS_METHOD[0] == "offset"
                and provider is not None
                and config["psd"].get("seam_inpaint", True)
                and upscaled
            ):
                from walan_design.seam_healer import heal_seams

                band = int(config["psd"].get("seam_inpaint_band", 140))
                # 主体描述（英文）供重绘 prompt 参考：帮助模型画出同主题的母题
                subj = (brief.get("sd_prompt") or "").strip().rstrip(",")[:80]
                logger.info(f"  SD inpaint 重绘接缝带 (band={band}px)...")
                try:
                    upscaled[0] = heal_seams(upscaled[0], provider, band_px=band, subject=subj)
                except Exception as e:
                    if is_hard_failure(str(e)):
                        raise provider_blocked(engine, str(e)) from e
                    logger.warning(f"  接缝重绘失败，保留硬切+羽化兜底: {e}")
                    record_attention(
                        config,
                        f"接缝重绘:{title}",
                        f"SD inpaint 接缝重绘失败（回退窄羽化，接缝处可能有细线）: {e}",
                    )

        # 4. 一花四色（按花型类型分路：传统=调色板坡道，数码=色相偏移保渐变）
        from walan_design.recolor import generate_color_variants

        color_count = design_cfg.get("color_variant_count", 4)
        dark_count = design_cfg.get("color_variant_mix", 2)
        recolor_cfg = design_cfg.get("recolor", {})
        palette_cfg = recolor_cfg.get("palettes")
        base_image = upscaled[0] if upscaled else images[0]
        # 模式判定：config 显式指定优先，否则按 brief/全局类型的数码标记走 hue_shift
        recolor_mode = recolor_cfg.get("mode", "auto")
        if recolor_mode not in ("palette", "hue_shift"):
            is_digital = (
                (brief.get("floral_type") or "").lower() == "digital"
                or design_cfg.get("brief_type", "mixed") == "digital"
            )
            recolor_mode = "hue_shift" if is_digital else "palette"
        logger.info(f"  生成一花四色变体: {color_count} 个（模式={recolor_mode}）")
        try:
            color_variants = generate_color_variants(
                base_image,
                count=color_count,
                dark_count=dark_count,
                palette_cfg=palette_cfg,
                cluster_count=int(recolor_cfg.get("cluster_count", 8)),
                mode=recolor_mode,
            )
            all_images = upscaled + color_variants
        except Exception as e:
            logger.warning(f"  配色变体生成失败: {e}")
            all_images = upscaled

        # 5. 保存：集中到每个花型一个目录 output/runs/{title}/
        run_dir = runs_dir / title
        run_dir.mkdir(parents=True, exist_ok=True)
        # 花型目录里也保存一份 brief.json 和 visual_review.json
        (run_dir / "brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "visual_review.json").write_text(json.dumps(critiques, ensure_ascii=False, indent=2), encoding="utf-8")

        image_paths = []
        dpi = config["psd"]["dpi"]
        for j, img in enumerate(all_images):
            fp = run_dir / f"design_{j + 1}.png"
            img.save(str(fp), "PNG", dpi=(dpi, dpi))
            image_paths.append(str(fp))
            logger.info(f"  保存: {fp} ({img.width}×{img.height})")

        result = {
            "brief": brief,
            "run_dir": str(run_dir),
            "image_paths": image_paths,
            "base_image_count": len(upscaled),
            "color_variant_count": len(all_images) - len(upscaled),
            "provider": provider.name if provider else "placeholder",
            "visual_review": critiques,  # 自评记录（含最终得分与建议）
        }
        # 完成标记：全部 PNG 落盘后才写，供断点续跑时跳过该设计
        (run_dir / "design_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)

    # 软故障汇总：显著提示 + CLI 结束时渲染黄色面板（attention.json）
    if failures:
        logger.warning(
            f"\n⚠️ {len(failures)}/{len(briefs)} 个设计生成失败被跳过，本批产出不完整:\n"
            + "\n".join(f"  - {f['title']}: {f['reason'][:120]}" for f in failures)
        )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    config = load_config()
    from walan_design.trend_collector import _local_fallback_briefs

    briefs = _local_fallback_briefs(config["trend"]["brief_count"], config["design"]["style_preferences"])
    results = run(briefs, config)
    print(f"\n生成完成，共 {len(results)} 个设计")
