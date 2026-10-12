"""
一花四色专业化 — 按花型类型分路的重着色。

传统花型（palette 模式）：
1. 图片量化为 N 个主色簇（median-cut）
2. 目标调色板按明度排序，作为"明度→颜色"的渐变坡道
3. 每个主色簇按其明度位置映射到坡道上的目标色
4. 像素级重映射时保留原始明暗纹理（局部明度/簇明度 缩放），细节不糊
效果：变体配色直接来自专业配色方案（瓦栏流行色系），符合按色制版。

数码花型（hue_shift 模式，2026-10-11 新增）：
调色板模式会把渐变压成色块（用户实测：数码花型看着像传统花型）。
改为 HSV 色相旋转 + 明度调整 + 白场混合——完整保留渐变过渡与丰富色域，
只改变整体色调，符合"数码印花不受套色限制"的本质。
"""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# 内置调色板（config 缺省时使用）。深浅分组，每组内色彩关系为完整方案。
BUILTIN_PALETTES = {
    "dark": [
        # 墨蓝金 —— 高级感靛蓝系
        {"name": "墨蓝金", "colors": ["#101828", "#1D2939", "#475467", "#D0A85C", "#EFEFEA"]},
        # 酒红夜曲 —— 勃艮第酒红系
        {"name": "酒红夜曲", "colors": ["#2B0F14", "#5C1A22", "#8C3041", "#C98A6B", "#E8D5C4"]},
        # 松林暮色 —— 森林绿系
        {"name": "松林暮色", "colors": ["#12211A", "#1F3A2C", "#3E5F4A", "#B0895E", "#E3DAC9"]},
    ],
    "light": [
        # 奶油乡村 —— 奶油底大地色
        {"name": "奶油乡村", "colors": ["#FDF6EC", "#F0E3CE", "#D9B99B", "#A67B5B", "#6B5B4C"]},
        # 雾霾粉彩 —— 莫兰迪粉
        {"name": "雾霾粉彩", "colors": ["#F9F1F0", "#EFD9D1", "#D8B4A6", "#A2938B", "#5C5450"]},
        # 薄荷清晨 —— 清新浅绿系
        {"name": "薄荷清晨", "colors": ["#F4F9F4", "#DCE9DD", "#B5CDB1", "#7A9E86", "#44614F"]},
    ],
}


def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _luminance(rgb: tuple) -> float:
    """相对明度 0-1（Rec.709）"""
    r, g, b = [c / 255.0 for c in rgb[:3]]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _select_palettes(palette_cfg: dict, count: int, dark_count: int) -> list:
    """从配置中选出 count 个调色板（dark_count 深底 + 其余浅底），循环取用"""
    dark_pool = palette_cfg.get("dark") or BUILTIN_PALETTES["dark"]
    light_pool = palette_cfg.get("light") or BUILTIN_PALETTES["light"]
    selected = [dark_pool[i % len(dark_pool)] for i in range(dark_count)]
    light_needed = count - dark_count
    selected += [light_pool[i % len(light_pool)] for i in range(light_needed)]
    return selected[:count]


def _palette_ramp(palette_colors: list[np.ndarray]) -> np.ndarray:
    """调色板颜色按明度排序 → 返回排序后的颜色数组（作为明度坡道控制点）"""
    arr = np.array(palette_colors, dtype=np.float32)
    lums = np.array([_luminance(c) for c in palette_colors])
    order = np.argsort(lums)
    return arr[order]


def _ramp_lookup(ramp: np.ndarray, t: np.ndarray) -> np.ndarray:
    """t∈[0,1] 数组 → 坡道插值颜色 (n,3)"""
    n = len(ramp)
    if n == 1:
        return np.tile(ramp[0], (len(t), 1))
    pos = t * (n - 1)
    idx = np.clip(np.floor(pos).astype(int), 0, n - 2)
    frac = (pos - idx).reshape(-1, 1)
    return ramp[idx] * (1 - frac) + ramp[idx + 1] * frac


def _recolor_with_palette(img: Image.Image, palette_hex: list, cluster_count: int = 8) -> Image.Image:
    """单张图 → 单个调色板重着色"""
    rgb = img.convert("RGB")

    # 1. 量化取主色簇（dithering off，保结构）
    small = rgb  # quantize 在全图上做，PIL 内部足够快
    quant = small.quantize(colors=cluster_count, method=Image.MEDIANCUT, dither=Image.NONE)
    pal = quant.getpalette()[: cluster_count * 3]
    centers = np.array(pal, dtype=np.float32).reshape(-1, 3)
    indices = np.array(quant, dtype=np.int32)  # (H, W)

    # 2. 簇明度 → 归一化位置 → 坡道目标色
    cluster_lums = np.array([_luminance(c) for c in centers])
    order = np.argsort(cluster_lums)
    ramp = _palette_ramp([np.array(_hex_to_rgb(h), dtype=np.float32) for h in palette_hex])
    t = np.linspace(0.02, 0.98, len(order))  # 避免纯黑白溢出
    target_by_rank = _ramp_lookup(ramp, t)

    target_centers = np.empty_like(centers)
    for rank, ci in enumerate(order):
        target_centers[ci] = target_by_rank[rank]

    # 3. 像素重映射：目标色 ×（像素局部明度 / 簇明度），保留纹理细节
    pixel_arr = np.array(rgb, dtype=np.float32)
    pixel_lum = pixel_arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32) / 255.0
    cluster_lum_safe = np.clip(cluster_lums[indices], 0.05, 1.0)
    ratio = np.clip(pixel_lum / cluster_lum_safe, 0.35, 1.8)[..., None]

    out = np.clip(target_centers[indices] * ratio, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


# 数码花型色相偏移预设：2 深 + 2 浅，hue=色相旋转角度，value=明度系数，
# sat=饱和度系数，whiten=向白混合比例（0-1，浅底用，保留渐变的"粉化"）
HUE_SHIFT_PRESETS = [
    {"name": "深调·青蓝转调", "hue": 150, "value": 0.68, "sat": 1.08, "whiten": 0.0},
    {"name": "深调·暖棕转调", "hue": 30, "value": 0.80, "sat": 1.00, "whiten": 0.0},
    {"name": "浅调·雾蓝粉化", "hue": 220, "value": 1.12, "sat": 0.92, "whiten": 0.16},
    {"name": "浅调·晨光粉化", "hue": 300, "value": 1.22, "sat": 0.85, "whiten": 0.28},
]


def _hue_shift_variant(img: Image.Image, preset: dict) -> Image.Image:
    """数码花型变体：HSV 色相旋转 + 明度/饱和度调整 + 可选白场混合。

    与 palette 模式的本质区别：不做任何量化/映射，渐变像素关系原样保留，
    只是整个色彩空间被旋转/调深浅 —— 输出仍是"无套色限制"的数码图。
    """
    hsv = np.array(img.convert("HSV"), dtype=np.float32)
    h = (hsv[..., 0] + preset["hue"] / 360.0 * 255.0) % 256.0
    s = np.clip(hsv[..., 1] * preset["sat"], 0, 255)
    v = np.clip(hsv[..., 2] * preset["value"], 0, 255)
    out = Image.fromarray(np.dstack([h, s, v]).astype(np.uint8), "HSV").convert("RGB")

    whiten = preset.get("whiten", 0.0)
    if whiten > 0:
        arr = np.array(out, dtype=np.float32)
        arr = arr * (1 - whiten) + 255.0 * whiten
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return out


def generate_color_variants(
    base_img: Image.Image,
    count: int = 4,
    dark_count: int = 2,
    palette_cfg: dict = None,
    cluster_count: int = 8,
    mode: str = "palette",
) -> list:
    """
    一花四色。
    mode="palette"   传统花型：调色板坡道重着色（按色制版观感）
    mode="hue_shift" 数码花型：色相偏移重着色（保留渐变与丰富色域）
    count>4 时预设循环取用。
    """
    if count <= 0:
        return []

    variants = []
    if mode == "hue_shift":
        for i in range(count):
            p = HUE_SHIFT_PRESETS[i % len(HUE_SHIFT_PRESETS)]
            logger.info(f"    数码转调[{p['name']}]: hue+{p['hue']}° v={p['value']} whiten={p['whiten']}")
            variants.append(_hue_shift_variant(base_img, p))
        return variants

    palettes = _select_palettes(palette_cfg or {}, count, dark_count)
    for p in palettes:
        logger.info(f"    配色方案[{p.get('name', '?')}]: {p['colors'][:3]}...")
        variants.append(_recolor_with_palette(base_img, p["colors"], cluster_count))
    return variants


def describe_variant_hue_spread(variants: list) -> float:
    """调试辅助：变体间平均色相差（0-1），验证色相差异明显"""
    hues = []
    for v in variants:
        small = v.convert("HSV").resize((32, 32))
        hsv = np.array(small, dtype=np.float32)
        hues.append(hsv[..., 0].mean() / 255.0)
    if len(hues) < 2:
        return 0.0
    diffs = [abs(hues[i] - hues[j]) for i in range(len(hues)) for j in range(i + 1, len(hues))]
    return float(np.mean(diffs))
