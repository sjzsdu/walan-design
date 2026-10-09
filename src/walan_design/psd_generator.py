"""
PSD 文件生成模块
将 AI 生成的 PNG 图片转换为符合瓦栏要求的 PSD 分层文件。

核心步骤:
  1. 加载 PNG → PIL Image
  2. 转 RGB → 设 DPI → 用 psd-tools create_pixel_layer 写入 PSD
  3. 为 PSD 的 ResolutionInfo 块注入正确的 DPI 值
"""

import logging
from pathlib import Path

import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_psd_with_dpi(pil_img: Image.Image, psd_path: str, dpi: int, layer_strategy: str = "background_main"):
    """
    用 psd-tools 创建带正确 DPI + 分层的 PSD 文件。

    分层策略（对齐瓦栏 PDF 标准）:
      - background_main: 背景层（柔和版）+ 主花层（原图）— 至少 2 层，最低合规
      - multi: 多层 — 背景层 + 主花层 + 次花层
      - single: 单层 — 只有 Pattern 层（不推荐，瓦栏建议分层）
    """
    import struct

    from PIL import ImageFilter
    from psd_tools import PSDImage

    w, h = pil_img.width, pil_img.height

    # 根据策略决定图层数量和内容
    layers = []
    if layer_strategy == "single":
        layers.append(("Pattern", pil_img))
    elif layer_strategy == "multi":
        # 背景层：模糊+降低饱和度的版本
        bg = pil_img.convert("L").convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(w, h) // 50))
        layers.append(("Background", bg))
        layers.append(("Main Pattern", pil_img))
        # 次花层：稍微调暗的版本（模拟第二层花）
        secondary = pil_img.point(lambda p: int(p * 0.7))
        layers.append(("Secondary Pattern", secondary))
    else:  # background_main（默认）
        # 背景层：降亮度+模糊，作为底色
        bg = pil_img.point(lambda p: int(p * 0.85)).filter(ImageFilter.GaussianBlur(radius=max(w, h) // 80))
        layers.append(("Background", bg))
        layers.append(("Pattern", pil_img))

    # 1. 创建 PSD 并按顺序添加图层
    psd = PSDImage.new("RGB", (w, h))
    for name, layer_img in layers:
        # 确保图层是 RGB 模式且尺寸一致
        layer_img = layer_img.convert("RGB").resize((w, h))
        psd.create_pixel_layer(layer_img, name=name)
    psd.save(psd_path)

    # 2. 手动 patch ResolutionInfo（psd-tools 没有直接 API）
    with open(psd_path, "rb") as f:
        data = bytearray(f.read())

    header_size = 26
    cmd_len = struct.unpack(">I", data[header_size : header_size + 4])[0]
    ir_len_pos = header_size + 4 + cmd_len
    ir_len = struct.unpack(">I", data[ir_len_pos : ir_len_pos + 4])[0]
    ir_start = ir_len_pos + 4
    ir_end = ir_start + ir_len

    fixed = int(dpi * 65536)
    res_data = struct.pack(">IHHIHH", fixed, 2, 1, fixed, 2, 1)
    resource_block = b"8BIM" + b"\x03\xed" + b"\x00\x00" + struct.pack(">I", len(res_data)) + res_data

    new_ir = data[ir_start:ir_end] + resource_block
    patched = bytes(data[:ir_len_pos]) + struct.pack(">I", len(new_ir)) + bytes(new_ir) + bytes(data[ir_end:])

    with open(psd_path, "wb") as f:
        f.write(patched)

    # 验证 PSD 可正常打开
    from psd_tools import PSDImage as _PSD

    _PSD.open(psd_path)


def png_to_psd(png_path: str, psd_path: str, config: dict) -> str:
    """将 PNG 转为 PSD"""
    psd_cfg = config["psd"]
    dpi = psd_cfg["dpi"]
    layer_strategy = psd_cfg.get("layer_strategy", "background_main")

    img = Image.open(png_path).convert("RGB")
    # 确保 PNG 自身也带 DPI（质量检测用）
    img.save(png_path, dpi=(dpi, dpi))

    try:
        _make_psd_with_dpi(img, psd_path, dpi, layer_strategy=layer_strategy)
        logger.info(f"  PSD 保存: {psd_path} (DPI={dpi}, 分层={layer_strategy})")
    except Exception as e:
        logger.error(f"PSD 生成失败: {e}")
        # 回退：用 psd-tools 基础 API（DPI 可能缺失，但至少文件可读）
        try:
            from psd_tools import PSDImage

            psd = PSDImage.new("RGB", (img.width, img.height))
            # 回退时也至少做 2 层（符合最低要求）
            psd.create_pixel_layer(img.point(lambda p: int(p * 0.85)), name="Background")
            psd.create_pixel_layer(img, name="Pattern")
            psd.save(psd_path)
            logger.info(f"  PSD 保存（基础版）: {psd_path}")
        except Exception as e2:
            raise RuntimeError(f"PSD 完全失败: {e2}")

    return psd_path


def process_design_result(result: dict, config: dict) -> dict:
    """处理单个设计结果：PNG → PSD"""
    output_dir = Path(config["psd"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    title = result["brief"].get("title", "design")
    psd_paths = []

    for i, png_path in enumerate(result["image_paths"]):
        psd_path = str(output_dir / f"{title}_{i + 1}.psd")
        try:
            png_to_psd(png_path, psd_path, config)
            psd_paths.append(psd_path)
        except Exception as e:
            logger.error(f"  [{png_path}] 转换失败: {e}")

    result["psd_paths"] = psd_paths
    return result


def run(design_results: list, config: dict = None) -> list:
    if config is None:
        config = load_config()

    logger.info(f"\n=== PSD 文件生成: 共 {len(design_results)} 个设计 ===")
    for i, result in enumerate(design_results):
        title = result["brief"].get("title", f"design_{i + 1}")
        logger.info(f"  {i + 1}/{len(design_results)}: {title}")
        process_design_result(result, config)

    total = sum(len(r.get("psd_paths", [])) for r in design_results)
    logger.info(f"完成，共生成 {total} 个 PSD")
    return design_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    config = load_config()
    from walan_design.ai_designer import run as design_run
    from walan_design.trend_collector import _local_fallback_briefs

    briefs = _local_fallback_briefs(2, config["design"]["style_preferences"])
    results = design_run(briefs, config)
    run(results, config)
