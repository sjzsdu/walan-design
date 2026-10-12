"""
SD inpaint 接缝重绘 — offset 硬切缝的内容级修复

offset 四象限偏移后，接缝（竖缝 x=sx、横缝 y=sy）两侧是原图不相关的边缘内容，
硬切会有内容断裂线。旧方案宽幅交叉淡化把缝两侧等距列（恰为原图镜像位置）
混在一起 → 完美对称的双影蝴蝶（2026-10-11 用户实测否决）。

本模块把缝十字带交给 Stability edit/inpaint 依据周围纹样重绘：
  1. 全图缩到 API 尺寸（1024×1536）
  2. mask：缝十字带涂白（重绘），其余涂黑（保留）
  3. inpaint 生成 → 与缩图按 mask 合成（带外强制原像素，防 API 全局漂移）
  4. 放大回原尺寸，只把带内像素贴回（带边缘 8px 羽化）
  5. 图案其余部分保持全分辨率原像素，成本 = 1 次 inpaint（4 credits）/设计

颜色锁定（2026-10-11 实测关键）：inpaint 模型会发明异色（蓝纹样里画出
绿叶/紫花），必须从图中提取主色生成颜色锁定 prompt + 排除未出现色系的
负向词。纯"continue the pattern"式弱 prompt 实测必然漂移。
"""

import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

API_W, API_H = 1024, 1536  # inpaint 处理尺寸（2:3 标准比例）

# 颜色 sanity check：重绘带均值色 vs 周围纹样均值色的最大容许距离（0-255 RGB 欧氏）
_COLOR_DRIFT_LIMIT = 60.0

# 色名锚点：(名称, 色系, 参考RGB)。取图中主色的最近锚点生成颜色锁定 prompt。
_ANCHORS = [
    ("white", "white", (250, 250, 248)),
    ("cream", "white", (244, 234, 214)),
    ("beige", "white", (226, 208, 180)),
    ("light gray", "gray", (205, 205, 205)),
    ("gray", "gray", (135, 135, 135)),
    ("charcoal", "gray", (65, 65, 65)),
    ("black", "black", (18, 18, 18)),
    ("red", "red", (200, 40, 40)),
    ("crimson", "red", (155, 25, 60)),
    ("maroon", "red", (110, 35, 45)),
    ("pink", "pink", (240, 165, 180)),
    ("blush pink", "pink", (244, 205, 205)),
    ("orange", "orange", (235, 125, 35)),
    ("terracotta", "orange", (195, 105, 65)),
    ("gold", "yellow", (215, 175, 65)),
    ("yellow", "yellow", (242, 212, 70)),
    ("olive green", "green", (130, 135, 55)),
    ("green", "green", (65, 145, 70)),
    ("sage green", "green", (155, 175, 135)),
    ("teal", "teal", (45, 135, 135)),
    ("cyan", "teal", (85, 190, 200)),
    ("blue", "blue", (55, 95, 200)),
    ("sky blue", "blue", (135, 185, 232)),
    ("navy blue", "blue", (30, 45, 105)),
    ("purple", "purple", (125, 65, 175)),
    ("lavender", "purple", (185, 168, 222)),
    ("magenta", "purple", (195, 55, 155)),
    ("brown", "brown", (120, 82, 52)),
    ("tan", "brown", (190, 158, 112)),
    ("coffee brown", "brown", (80, 55, 38)),
]

# 负向词候选色系：图里没出现的色系全部排除（防 inpaint 发明异色）
_NEG_FAMILIES = ["red", "pink", "orange", "yellow", "green", "teal", "blue", "purple", "brown"]
_NEG_FIXED = "multicolor, clashing colors, border, frame, vignette"


def seam_positions(w: int, h: int) -> tuple[int, int]:
    """offset 缝位置（与 ai_designer.make_seamless 的 (w-hw, h-hh) 保持一致）"""
    hw, hh = w // 2, h // 2
    return w - hw, h - hh


def build_seam_mask(w: int, h: int, band_px: int, size: tuple[int, int] = (API_W, API_H)) -> Image.Image:
    """缝十字带 mask（白=重绘）。band_px 为全分辨率下的带半宽。"""
    sw, sh = size
    sx, sy = seam_positions(w, h)
    cx, cy = round(sx * sw / w), round(sy * sh / h)
    kx = max(2, round(band_px * sw / w))
    ky = max(2, round(band_px * sh / h))
    m = np.zeros((sh, sw), dtype=np.uint8)
    m[:, max(0, cx - kx) : min(sw, cx + kx)] = 255
    m[max(0, cy - ky) : min(sh, cy + ky), :] = 255
    return Image.fromarray(m, "L")


def _edge_ramp(n: int, lo: int, hi: int, feather: int) -> np.ndarray:
    """1D 权重：[lo,hi) 内=1，带外=0，两端各 feather px 线性过渡"""
    w = np.zeros(n, dtype=np.float32)
    w[lo:hi] = 1.0
    for i in range(feather):
        a = 1.0 - (i + 1) / (feather + 1)
        if lo - 1 - i >= 0:
            w[lo - 1 - i] = max(w[lo - 1 - i], a)
        if hi + i < n:
            w[hi + i] = max(w[hi + i], a)
    return w


def _dominant_colors(img: Image.Image, n: int = 5) -> list:
    """median-cut 量化取主色 → [(rgb, count)] 按 count 降序"""
    q = img.convert("RGB").quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    counts = sorted(q.getcolors() or [], reverse=True)
    return [(tuple(pal[idx * 3 : idx * 3 + 3]), cnt) for cnt, idx in counts]


def _nearest_anchor(rgb: tuple) -> tuple:
    """最近锚点 → (名称, 色系)"""
    arr = np.array(rgb, dtype=np.float32)
    best, best_d = _ANCHORS[0], float("inf")
    for name, family, ref in _ANCHORS:
        d = float(np.linalg.norm(arr - np.array(ref, dtype=np.float32)))
        if d < best_d:
            best, best_d = (name, family), d
    return best


def _saturation(rgb: tuple) -> float:
    mx, mn = max(rgb), min(rgb)
    return (mx - mn) / mx if mx else 0.0


def build_color_locked_prompts(img: Image.Image, subject: str = "") -> tuple[str, str]:
    """
    从图中提取主色 → 颜色锁定 (prompt, negative_prompt)。

    实测（青花飞鸟枝桠）：'blue and white ..., dominant navy blue motifs on
    white background, ...' + 负向词排除绿/紫/棕 → inpaint 重绘色完美协调；
    无颜色锁定时模型发明绿叶紫花。
    """
    dominant = _dominant_colors(img, 5)
    if not dominant:
        return ("seamlessly continue the textile pattern through the gap", "")

    named = [(rgb, cnt, _nearest_anchor(rgb)) for rgb, cnt in dominant]

    # 母题色：主色里饱和度×权重 最高者（背景通常低饱和大权重）
    motif = max(named, key=lambda x: _saturation(x[0]) * (x[1] ** 0.5))[2][0]
    # 背景色：与母题不同名的最高权重色（优先低饱和浅色）
    bg_cands = [x for x in named if x[2][0] != motif]
    bg_cands.sort(key=lambda x: (_saturation(x[0]) < 0.3, x[1]), reverse=True)
    bg = bg_cands[0][2][0] if bg_cands else motif

    subj = f" {subject.strip()}" if subject.strip() else " textile pattern"
    prompt = (
        f"{motif} and {bg}{subj}, dominant {motif} motifs on {bg} background, "
        "seamlessly continue the pattern through the gap, "
        "matching the surrounding motifs, scale and painting style"
    )

    present = {x[2][1] for x in named}
    neg_colors = [f for f in _NEG_FAMILIES if f not in present]
    negative = ", ".join(neg_colors + [_NEG_FIXED])

    logger.info(f"  接缝重绘颜色锁定: motif={motif}, bg={bg}, 排除色系={neg_colors}")
    return prompt, negative


def heal_seams(img: Image.Image, provider, band_px: int = 140, subject: str = "") -> Image.Image:
    """
    重绘 offset 接缝十字带。provider 需实现 inpaint(image, mask, prompt, negative_prompt) 接口。
    band_px：全分辨率带半宽（默认 140px，约占 4724 宽的 3%）。
    subject：花型主体描述（英文，来自 brief.sd_prompt），供重绘 prompt 参考。

    失败（含重绘带颜色漂移 sanity check 不过）抛 RuntimeError，
    调用方（ai_designer）按硬/软故障分流处理。
    """
    w, h = img.size
    sx, sy = seam_positions(w, h)
    small = img.convert("RGB").resize((API_W, API_H), Image.LANCZOS)
    mask = build_seam_mask(w, h, band_px)

    prompt, negative = build_color_locked_prompts(small, subject=subject)
    healed = provider.inpaint(small, mask, prompt, negative_prompt=negative)

    # 按 mask 与缩图合成：带外强制原像素（防 API 全局漂移），mask 边缘 2px 软过渡
    hs = np.array(healed).astype(np.float32)
    ss = np.array(small).astype(np.float32)
    m = np.array(mask.filter(ImageFilter.GaussianBlur(2)))
    m = (m.astype(np.float32) / 255.0)[..., None]
    comp = hs * m + ss * (1 - m)
    comp_img = Image.fromarray(comp.astype(np.uint8)).resize((w, h), Image.LANCZOS)

    # 只贴带内像素回全分辨率原图（带边缘 8px 羽化）
    rep = np.array(comp_img).astype(np.float32)
    arr = np.array(img.convert("RGB")).astype(np.float32)
    feather = 8
    wx = _edge_ramp(w, max(0, sx - band_px), min(w, sx + band_px), feather)
    wy = _edge_ramp(h, max(0, sy - band_px), min(h, sy + band_px), feather)
    weight = np.maximum(wx[None, :], wy[:, None])[:, :, None]  # 十字并集
    out = arr * (1 - weight) + rep * weight

    _check_color_drift(out, w, h, sx, sy, band_px)

    logger.info(f"  接缝重绘完成（band={band_px}px, API {API_W}×{API_H}, 缝位 {sx},{sy}）")
    return Image.fromarray(out.astype(np.uint8))


def _check_color_drift(out: np.ndarray, w: int, h: int, sx: int, sy: int, band_px: int,
                       limit: float = _COLOR_DRIFT_LIMIT):
    """重绘带均值色 vs 周围环带均值色：色系跑偏（模型发明异色）视为失败。"""
    def cross(band: int) -> np.ndarray:
        cx = _edge_ramp(w, max(0, sx - band), min(w, sx + band), 4)
        cy = _edge_ramp(h, max(0, sy - band), min(h, sy + band), 4)
        return np.maximum(cx[None, :], cy[:, None])

    band_m = cross(band_px) > 0.5
    ring_m = (cross(int(band_px * 2.5)) > 0.5) & ~band_m
    if band_m.sum() < 100 or ring_m.sum() < 100:
        return
    d = float(np.linalg.norm(out[band_m].mean(axis=0) - out[ring_m].mean(axis=0)))
    if d > limit:
        raise RuntimeError(
            f"接缝重绘颜色漂移：重绘带均值色偏离周围纹样 {d:.0f}/255（阈值 {limit:.0f}），已放弃本次重绘"
        )
    logger.debug(f"  重绘带颜色检查通过（偏移 {d:.1f}）")
