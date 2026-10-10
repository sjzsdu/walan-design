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


def _split_by_colors(
    img: Image.Image,
    n_colors: int = 8,
    max_colors: int = 24,
    color_tolerance: float = 8.0,
    pure_color: bool = True,
) -> list[tuple]:
    """
    传统花型按色分色：量化提取主色 → 每个主色生成一个 RGBA 透明图层。

    严格制版合规：传统印花每层一色对应一个网版，层内必须是单一颜色。
      - pure_color=True（默认）：每层填充主色纯色，严格单色可制版
      - pure_color=False：层内保留原像素（保留纹理，但层内可能多色）

    自适应加色：若某层内平均色差 > color_tolerance（颜色簇太粗），
    增加主色数重试（最多 max_colors），降低纯色化的视觉损失。

    返回 [(主色(R,G,B), RGBA图层, 覆盖率), ...] 按覆盖率降序（大面积在下层）。
    覆盖率 <0.05% 的杂色被并入忽略（抗锯齿噪点）。
    """
    import numpy as np

    rgb_img = img.convert("RGB")
    idx_arr = arr = palette_colors = None

    # 自适应：从 n_colors 开始，层内平均色差超标就加色重试
    for n in range(n_colors, max_colors + 1, 2):
        quant = rgb_img.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
        pal = quant.getpalette() or []
        n_actual = min(n, len(pal) // 3)
        palette_colors = [tuple(pal[i * 3 : i * 3 + 3]) for i in range(n_actual)]
        idx_arr = np.array(quant)
        arr = np.array(rgb_img)

        # 计算每个簇的平均色差
        max_cluster_err = 0.0
        for ci, rgb in enumerate(palette_colors):
            mask = idx_arr == ci
            if not mask.any():
                continue
            err = np.abs(arr[mask].astype(np.float32) - np.array(rgb, dtype=np.float32)).mean()
            max_cluster_err = max(max_cluster_err, float(err))
        if max_cluster_err <= color_tolerance:
            break
        logger.info(f"  分色: {n} 色时层内最大色差 {max_cluster_err:.1f} > 容差 {color_tolerance}，增加色数")

    h, w = idx_arr.shape

    # 合并近重复主色：自适应加色后可能出现 4 个几乎相同的深绿层（色差 <容差），
    # 制版时一色一网版，近重复色是浪费。把彼此距离 < color_tolerance 的主色
    # 并入最先出现（覆盖率更大）的主色，重新分配像素索引
    accepted = []  # [(主色RGB, 原始索引)]
    remap = {}
    for ci, rgb in enumerate(palette_colors):
        target = None
        for aci, (argb, _src_ci) in enumerate(accepted):
            d = float(np.abs(np.array(rgb, np.float32) - np.array(argb, np.float32)).mean())
            if d < color_tolerance:
                target = aci
                break
        if target is None:
            accepted.append((rgb, ci))
            remap[ci] = ci
        else:
            remap[ci] = accepted[target][1]
    if len(accepted) < len(palette_colors):
        for src_ci, dst_ci in remap.items():
            if src_ci != dst_ci:
                idx_arr[idx_arr == src_ci] = dst_ci
        # 按合并后索引重排调色板（保持 idx 索引与列表位置一致）
        kept = sorted({ci for _, ci in accepted})
        # 关键：把 idx_arr 里的原始索引重新映射成 0..N-1 连续索引，
        # 否则 palette_colors 列表长度变短后，用 enumerate 的 ci 去 mask 永远匹配不到旧高索引值
        old_to_new = {old: new for new, old in enumerate(kept)}
        new_idx = np.zeros_like(idx_arr)
        for old, new in old_to_new.items():
            new_idx[idx_arr == old] = new
        idx_arr = new_idx
        palette_colors = [palette_colors[ci] for ci in kept]
        logger.info(f"  合并近重复色: {len(remap)} → {len(kept)} 色")

    total = h * w

    result = []
    for ci, rgb in enumerate(palette_colors):
        mask = idx_arr == ci
        coverage = float(mask.sum()) / total
        if coverage < 0.0005:  # 忽略 <0.05% 杂色
            continue
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if pure_color:
            # 严格单色：整层用主色纯色填充（制版合规）
            rgba[mask, :3] = rgb
        else:
            # 保留该色内的原像素纹理（层内可能存在多色）
            rgba[mask, :3] = arr[mask]
        rgba[mask, 3] = 255
        result.append((rgb, Image.fromarray(rgba, "RGBA"), coverage))

    result.sort(key=lambda x: -x[2])
    logger.info(f"  分色完成: 最终 {len(palette_colors)} 主色, {len(result)} 个有效层")
    return result


def _detect_floral_type(img: Image.Image, fidelity_threshold: float = 0.95) -> str:
    """
    自动判断传统花型 / 数码花型。

    判据：传统花型颜色离散（整个花型几个固定色），median-cut 量化到 8 色
    后几乎无损还原；数码花型有渐变过渡，量化损失大。
    用量化还原度（1 - 平均像素误差/255）区分：
      - 还原度 >= threshold → "traditional"（颜色集中，可按色制版）
      - 否则 → "digital"（渐变丰富，不适合按色分层）

    在 512px 缩略图上计算，成本低。
    """
    import numpy as np

    small = img.convert("RGB")
    small.thumbnail((512, 512), Image.LANCZOS)
    quant = small.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    restored = quant.convert("RGB")

    arr = np.array(small, dtype=np.float32)
    res = np.array(restored, dtype=np.float32)
    mean_err = float(np.abs(arr - res).mean())
    fidelity = 1.0 - mean_err / 255.0
    floral_type = "traditional" if fidelity >= fidelity_threshold else "digital"
    logger.info(f"  花型类型检测: 还原度={fidelity:.3f} → {floral_type}")
    return floral_type


def _make_psd_with_dpi(
    pil_img: Image.Image,
    psd_path: str,
    dpi: int,
    layer_strategy: str = "auto",
    compression: str = "zip_prediction",
    color_layer_count: int = None,
    max_colors: int = None,
    color_tolerance: float = None,
    pure_color: bool = None,
):
    """
    用 psd-tools 创建带正确 DPI + 分层 + 压缩的 PSD 文件。

    分层策略（对齐瓦栏 PDF 标准）:
      - background_main: 背景层（柔和版）+ 主花层（原图）— 数码花型最低合规
      - color_layers: 按颜色分色分层（传统花型制版标准）— 每个主色一层 RGBA 透明图层，
        大面积色在下、细碎色在上，买家（印花厂）可直接按色制版
      - multi: 多层 — 背景层 + 主花层 + 次花层
      - single: 单层 — 只有 Pattern 层（不推荐，瓦栏建议分层）

    压缩策略（实测 4724×7087×2 层 PSD）:
      - raw:   351 MB (无压缩，现用路径)
      - rle:   113 MB (默认，好一些)
      - zip:    54 MB (不错)
      - zip_prediction: 42 MB (最佳，Photoshop CS5+ 支持)
    """
    import struct

    from PIL import ImageFilter
    from psd_tools import PSDImage
    from psd_tools.constants import Compression

    # 压缩策略映射（config 字符串 → Compression 枚举）
    compr_map = {
        "raw": Compression.RAW,
        "rle": Compression.RLE,
        "zip": Compression.ZIP,
        "zip_prediction": Compression.ZIP_WITH_PREDICTION,
    }
    compr = compr_map.get(compression, Compression.ZIP_WITH_PREDICTION)

    w, h = pil_img.width, pil_img.height

    # 根据策略决定图层数量和内容
    layers = []  # (name, image, is_rgba)
    if layer_strategy == "color_layers":
        # 传统花型制版标准：按颜色分色，每色一层透明图层
        n_colors = color_layer_count or 8
        color_layers = _split_by_colors(
            pil_img,
            n_colors=n_colors,
            max_colors=max_colors or 24,
            color_tolerance=color_tolerance or 8.0,
            pure_color=True if pure_color is None else pure_color,
        )
        logger.info(f"  按色分色: {len(color_layers)} 个主色层")
        # 覆盖率大的先添加（= 底层），细碎颜色在上层
        for i, (rgb, rgba_img, _coverage) in enumerate(color_layers):
            hex_color = "#{:02X}{:02X}{:02X}".format(*rgb)
            layers.append((f"C{i+1}_{hex_color}", rgba_img, True))
        # 保护：若分色后不足 2 层（如近单色图），补一层完整合成
        if len(layers) < 2:
            layers.append(("Pattern", pil_img, False))
    elif layer_strategy == "single":
        layers.append(("Pattern", pil_img, False))
    elif layer_strategy == "multi":
        # 背景层：模糊+降低饱和度的版本
        bg = pil_img.convert("L").convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(w, h) // 50))
        layers.append(("Background", bg, False))
        layers.append(("Main Pattern", pil_img, False))
        # 次花层：稍微调暗的版本（模拟第二层花）
        secondary = pil_img.point(lambda p: int(p * 0.7))
        layers.append(("Secondary Pattern", secondary, False))
    else:  # background_main（默认，数码花型）
        # 背景层：降亮度+模糊，作为底色
        bg = pil_img.point(lambda p: int(p * 0.85)).filter(ImageFilter.GaussianBlur(radius=max(w, h) // 80))
        layers.append(("Background", bg, False))
        layers.append(("Pattern", pil_img, False))

    # 1. 创建 PSD 并按顺序添加图层（每个图层用指定压缩）
    # 关键：按色透明图层必须用 RGBA 文档模式——RGB 文档下 psd-tools 会把
    # 图层 alpha 转成 USER_LAYER_MASK 而非透明通道，导致透明度丢失
    doc_mode = "RGBA" if layer_strategy == "color_layers" else "RGB"
    psd = PSDImage.new(doc_mode, (w, h))
    for name, layer_img, is_rgba in layers:
        if is_rgba:
            # 按色透明图层：保持 RGBA，只校正尺寸
            if layer_img.size != (w, h):
                layer_img = layer_img.resize((w, h))
        else:
            layer_img = layer_img.convert("RGB").resize((w, h))
        psd.create_pixel_layer(layer_img, name=name, compression=compr)

    # 关键：强制 merged ImageData 也用压缩（PSDImage.new 默认是 RAW）
    # 注意：psd-tools 读取 RGBA 文档的 ZIP 系 image_data 时有读端误报
    # （"decode failed, replaced with black"），但写入的数据是标准 zlib 流，
    # Photoshop 按规范可正常解压。图层与 preview 统一 zip_prediction 保证体积。
    psd._record.image_data.compression = compr

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

    # XMP 元数据（0x0424）— Photoshop 标准做法，PIL/exiftool 等解析器读 DPI 的另一条路径
    xmp = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:tiff="http://ns.adobe.com/tiff/1.0/">\n'
        f'   <tiff:XResolution>{dpi}/1</tiff:XResolution>\n'
        f'   <tiff:YResolution>{dpi}/1</tiff:YResolution>\n'
        '   <tiff:ResolutionUnit>2</tiff:ResolutionUnit>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    ).encode("utf-8")
    # PSD 规范: resource data 按 2 字节对齐，奇数长度补 1 字节 pad（pad 不计入长度字段）
    resource_block += (
        b"8BIM" + b"\x04\x24" + b"\x00\x00" + struct.pack(">I", len(xmp)) + xmp + (b"\x00" if len(xmp) % 2 else b"")
    )

    new_ir = data[ir_start:ir_end] + resource_block
    patched = bytes(data[:ir_len_pos]) + struct.pack(">I", len(new_ir)) + bytes(new_ir) + bytes(data[ir_end:])

    with open(psd_path, "wb") as f:
        f.write(patched)

    # 验证 PSD 可正常打开
    from psd_tools import PSDImage as _PSD

    _PSD.open(psd_path)


def png_to_psd(png_path: str, psd_path: str, config: dict, floral_type: str = None) -> str:
    """将 PNG 转为 PSD。floral_type 可显式传入（auto 策略时避免重复检测）"""
    psd_cfg = config["psd"]
    dpi = psd_cfg["dpi"]
    layer_strategy = psd_cfg.get("layer_strategy", "auto")
    compression = psd_cfg.get("compression", "zip_prediction")
    color_layer_count = psd_cfg.get("color_layer_count")

    img = Image.open(png_path).convert("RGB")
    # 确保 PNG 自身也带 DPI（质量检测用）
    img.save(png_path, dpi=(dpi, dpi))

    # auto 策略：按花型类型自动选分层方式
    if layer_strategy == "auto":
        if floral_type is None:
            floral_type = _detect_floral_type(img, psd_cfg.get("fidelity_threshold", 0.95))
        layer_strategy = "color_layers" if floral_type == "traditional" else "background_main"
        logger.info(f"  auto 分层: {floral_type} → {layer_strategy}")

    try:
        _make_psd_with_dpi(
            img, psd_path, dpi,
            layer_strategy=layer_strategy,
            compression=compression,
            color_layer_count=color_layer_count,
            max_colors=psd_cfg.get("max_color_count"),
            color_tolerance=psd_cfg.get("color_tolerance"),
            pure_color=psd_cfg.get("pure_color_layers"),
        )
        logger.info(f"  PSD 保存: {psd_path} (DPI={dpi}, 分层={layer_strategy}, 压缩={compression})")
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
    """处理单个设计结果：PNG → PSD，并检测花型类型（供上传分类联动）"""
    # PSD 文件跟 run_dir 走（花型目录集中存放），若没有 run_dir 则用旧 config 兜底
    if result.get("run_dir"):
        output_dir = Path(result["run_dir"])
    else:
        output_dir = Path(config["psd"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    title = result["brief"].get("title", "design")
    psd_paths = []

    # auto 策略时检测一次花型类型，整个设计的所有变体共用
    # （同一设计的 4 个配色变体风格一致，类型相同）
    floral_type = None
    if config["psd"].get("layer_strategy", "auto") == "auto" and result.get("image_paths"):
        floral_type = _detect_floral_type(
            Image.open(result["image_paths"][0]),
            config["psd"].get("fidelity_threshold", 0.95),
        )
        result["floral_type"] = floral_type

    for i, png_path in enumerate(result["image_paths"]):
        psd_path = str(output_dir / f"{title}_{i + 1}.psd")
        try:
            png_to_psd(png_path, psd_path, config, floral_type=floral_type)
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
