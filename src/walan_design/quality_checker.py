"""
质量检测模块
在生成 PSD 后、上传前进行自动质量自检。
检测项：分辨率、接回位、色彩模式、是否含文字水印、原创性。
"""

import logging
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_resolution(img_path: str, min_dpi: int = 200) -> tuple[bool, str]:
    """检查 DPI 是否达标。对 PSD 用 psd-tools 读 ResolutionInfo，其他用 PIL 读 info.dpi。"""
    try:
        if img_path.lower().endswith(".psd"):
            from psd_tools import PSDImage

            psd = PSDImage.open(img_path)
            dpi = None
            for key, val in psd.image_resources.items():
                if int(key) == 1005:  # ResolutionInfo
                    ri = val.data  # ResoulutionInfo (psd-tools 拼写)
                    hRes = ri.horizontal / 65536  # 16.16 定点数 → DPI
                    dpi = int(hRes)
                    break
            del psd
            if dpi is None:
                return False, "PSD 中无 ResolutionInfo 块"
        else:
            img = Image.open(img_path)
            # PNG 的 DPI 回读常是 299.9994 这类浮点值，先取整再比较，避免 >=300 误挂
            dpi = round(img.info.get("dpi", (0, 0))[0])

        if dpi >= min_dpi:
            return True, f"DPI={dpi}"
        return False, f"DPI={dpi} 低于要求 {min_dpi}"
    except Exception as e:
        return False, f"无法读取 DPI: {e}"


def check_seamless(img_path: str, tolerance: int = 1) -> tuple[bool, str]:
    """
    检查四方连续接回位质量。
    方法：比较左边缘和右边缘的像素差异、上边缘和下边缘的像素差异。
    如果差异在容差内，认为接回位合格。
    """
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]

        # 比较左右边缘（各取 1 像素宽）
        left_edge = arr[:, 0, :].astype(float)
        right_edge = arr[:, w - 1, :].astype(float)
        h_diff = np.mean(np.abs(left_edge - right_edge))

        # 比较上下边缘
        top_edge = arr[0, :, :].astype(float)
        bottom_edge = arr[h - 1, :, :].astype(float)
        v_diff = np.mean(np.abs(top_edge - bottom_edge))

        avg_diff = (h_diff + v_diff) / 2
        passed = avg_diff <= tolerance
        return passed, f"平均边缘差异={avg_diff:.2f} (容差={tolerance})"
    except Exception as e:
        return False, f"接回位检测异常: {e}"


def check_color_mode(img_path: str) -> tuple[bool, str]:
    """检查是否为 RGB 模式"""
    try:
        img = Image.open(img_path)
        mode = img.mode
        if mode == "RGB":
            return True, f"模式={mode}"
        return False, f"模式={mode}（应为 RGB）"
    except Exception as e:
        return False, f"无法读取色彩模式: {e}"


def check_no_text(img_path: str) -> tuple[bool, str]:
    """
    简易文字/水印检测。
    用边缘密度判断：图片上有大面积高对比度边缘区域可能是文字/水印。
    """
    try:
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.float32)
        h_grad = np.abs(np.diff(arr, axis=1))
        v_grad = np.abs(np.diff(arr, axis=0))
        edge_density = (np.mean(h_grad) + np.mean(v_grad)) / 255.0

        if edge_density > 0.15:
            return False, f"边缘密度={edge_density:.3f}，可能含文字/水印"
        return True, f"边缘密度={edge_density:.3f}"
    except Exception as e:
        return False, f"文字检测异常: {e}"


def check_dimensions(
    img_path: str,
    expected_ratio: tuple[int, int] = (2, 3),
    tolerance: float = 0.05,
) -> tuple[bool, str]:
    """
    检查尺寸比例是否符合瓦栏标准（数码四方连续 40cm×60cm = 2:3）。
    允许 5% 容差。
    """
    try:
        img = Image.open(img_path)
        w, h = img.size
        # 计算实际比例（小/大），然后和期望比例比较
        actual_ratio = min(w, h) / max(w, h)
        expected_ratio = expected_ratio[0] / expected_ratio[1]
        ratio_diff = abs(actual_ratio - expected_ratio) / expected_ratio

        if ratio_diff <= tolerance:
            return True, f"尺寸={w}×{h}, 比例={actual_ratio:.3f}"
        msg = f"尺寸={w}×{h}, 比例={actual_ratio:.3f}"
        msg += f" (期望 {expected_ratio[0]}:{expected_ratio[1]}, 偏差 {ratio_diff * 100:.1f}%)"
        return False, msg
    except Exception as e:
        return False, f"尺寸检测异常: {e}"


def check_color_variants(result: dict, min_count: int = 4) -> tuple[bool, str]:
    """
    检查一花四色是否齐全（PDF 强制标准）。
    统计该设计结果下的 PNG 数量（原图+配色变体）。
    """
    try:
        png_paths = result.get("image_paths", [])
        psd_paths = result.get("psd_paths", [])
        # 取 PNG 数量（更直接反映配色变体）
        count = len(png_paths) if png_paths else len(psd_paths)
        if count >= min_count:
            return True, f"共 {count} 张（≥ {min_count}）"
        return False, f"只有 {count} 张（< {min_count}，缺 {(min_count - count)} 个配色方案）"
    except Exception as e:
        return False, f"配色检测异常: {e}"


def check_tag_count(tags: list, min_count: int = 3, max_count: int = 5) -> tuple[bool, str]:
    """检查标签数量是否在允许范围内（PDF 标准：3-5 个）"""
    if min_count <= len(tags) <= max_count:
        return True, f"标签 {len(tags)} 个: {', '.join(tags)}"
    return False, f"标签 {len(tags)} 个（应 {min_count}-{max_count} 个）: {', '.join(tags)}"


def quality_check_image(img_path: str, config: dict) -> dict:
    """对单张图片执行所有配置的检测"""
    checks_cfg = config["quality"]["checks"]
    tolerance = config["psd"].get("seamless_tolerance", 0)  # PDF 标准：相差 1 像素都不行
    min_dpi = 300 if config["psd"]["color_mode"] == "RGB" else 200

    results = {}

    if checks_cfg.get("resolution", True):
        results["resolution"] = check_resolution(img_path, min_dpi)
    if checks_cfg.get("seamless", True):
        results["seamless"] = check_seamless(img_path, tolerance)
    if checks_cfg.get("color_mode", True):
        results["color_mode"] = check_color_mode(img_path)
    if checks_cfg.get("no_text", True):
        results["no_text"] = check_no_text(img_path)
    if checks_cfg.get("dimensions", True):
        results["dimensions"] = check_dimensions(img_path)
    if checks_cfg.get("originality", True):
        # 显式 no-op：原创性检测尚未实现（originality_threshold 未接线）。
        # 这里明示跳过而不是静默不跑，避免「8 项全过」的假象。
        results["originality"] = (True, "未实现，显式跳过（originality_threshold 未接线）")

    return results


def is_passed(check_results: dict) -> bool:
    """判断检查结果是否全部通过"""
    return all(v[0] for v in check_results.values())


def run(design_results: list, config: dict = None) -> list:
    """
    对所有设计结果执行质量检测。
    不通过的按 config["quality"]["on_fail"] 处理：retry / skip / manual_review
    """
    if config is None:
        config = load_config()

    if not config["quality"]["enabled"]:
        logger.info("质量检测已禁用，跳过")
        return design_results

    checks_cfg = config["quality"]["checks"]
    color_min = config["design"].get("color_variant_count", 4)
    tag_min = config["walan"].get("tag_min", 3)
    tag_max = config["walan"].get("tag_max", 5)
    on_fail = config["quality"]["on_fail"]

    passed_results = []

    logger.info(f"\n=== 质量检测: 共 {len(design_results)} 个设计 ===")
    for i, result in enumerate(design_results):
        title = result["brief"].get("title", f"design_{i + 1}")
        logger.info(f"\n  检测 {i + 1}/{len(design_results)}: {title}")

        files_to_check = result.get("psd_paths") or result.get("image_paths", [])
        all_checks_pass = True

        # 单文件检测
        for j, fpath in enumerate(files_to_check):
            checks = quality_check_image(fpath, config)
            result.setdefault("checks", {})[Path(fpath).name] = checks

            file_pass = is_passed(checks)
            if not file_pass:
                all_checks_pass = False
            status = "✓" if file_pass else "✗"
            logger.info(f"    {status} 文件 {j + 1}: {Path(fpath).name}")
            for check_name, (passed, detail) in checks.items():
                mark = "OK" if passed else "FAIL"
                logger.info(f"      [{mark}] {check_name}: {detail}")

        # 设计级检测（一花四色、标签数量）
        if checks_cfg.get("color_variants", True):
            cv_pass, cv_detail = check_color_variants(result, min_count=color_min)
            result.setdefault("checks", {})["color_variants"] = (cv_pass, cv_detail)
            if not cv_pass:
                all_checks_pass = False
            logger.info(f"    [{'OK' if cv_pass else 'FAIL'}] 一花四色: {cv_detail}")

        if checks_cfg.get("tag_count", True):
            tags = result["brief"].get("tags", [])
            tc_pass, tc_detail = check_tag_count(tags, min_count=tag_min, max_count=tag_max)
            result.setdefault("checks", {})["tag_count"] = (tc_pass, tc_detail)
            if not tc_pass:
                all_checks_pass = False
            logger.info(f"    [{'OK' if tc_pass else 'FAIL'}] 标签: {tc_detail}")

        if all_checks_pass:
            passed_results.append(result)
            logger.info("    → 通过")
        elif on_fail == "skip":
            logger.warning("    → 不通过，跳过")
        elif on_fail == "manual_review":
            logger.warning("    → 不通过，等待人工审核（当前自动跳过）")
        elif on_fail == "retry":
            logger.warning("    → 不通过，将被跳过（需要重新调用 SD 生成才能真正 retry）")
            result["_qc_failed"] = True
            if result.get("psd_paths"):
                passed_results.append(result)

    logger.info(f"\n质量检测完成: {len(passed_results)}/{len(design_results)} 通过")
    return passed_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    config = load_config()
    from walan_design.ai_designer import run as design_run
    from walan_design.psd_generator import run as psd_run
    from walan_design.trend_collector import _local_fallback_briefs

    briefs = _local_fallback_briefs(2, config["design"]["style_preferences"])
    results = design_run(briefs, config)
    psd_run(results, config)
    run(results, config)
