"""
确定性四方连续接缝：相位错位 + 静区寻径（texture quilting）

为什么不用 AI inpaint 修缝（2026-10-11 实测 3/3 失败的教训）：
  offset 硬切后把缝带交给 SD inpaint 重绘 —— 模型可能画出好内容，
  也可能输出"平均色糊"（六角形纹样被涂成纯棕色十字）或错误结构，
  且无 seed 不可复现。自动售卖流水线不能赌运气。

确定性方案（本模块）：
  1. 输出由两片组成：左片=原图原相（c_L=0），右片=原图相位前移 Δ px（c_R=Δ）
  2. 接缝处的拼接对从"左右边缘强拼"（A[w-1]|A[0]，差异巨大）变成
     "相隔 Δ+1 px 的内容错位"（差异≈纹理基线，特征尺寸远大于 Δ 时近乎不可见）
  3. 接缝位置逐行 DP 寻径（经典 min-cut quilting）：路径可左右蜿蜒 ±K px，
     每行选 |A[y,i]-A[y,i+1+Δ]| 最小处穿过 —— 自动绕开细节、走背景静区
  4. 平铺环绕点同理： wrap 对 = 相隔 Δ-1 px，同为近基线
  5. 代价：纹样相位在缝处整体前移 Δ px（一个细条被跳过）—— 对特征尺寸
     50-500px 的花型不可见；换取：零 AI、零 API 成本、结果确定可复现

竖向、横向各做一次（先竖后横），输出仍是严格四方连续。
"""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _min_path(cost: np.ndarray) -> np.ndarray:
    """DP 最小路径：cost (H, W)，路径每步可走 i-1/i/i+1，返回每行选的状态 idx (H,)"""
    h, w = cost.shape
    dp = cost[0].copy()
    back = np.zeros((h, w), dtype=np.int32)
    for y in range(1, h):
        # 三邻域最小值
        prev = dp
        shifted_l = np.empty(w)
        shifted_r = np.empty(w)
        shifted_l[1:] = prev[:-1]
        shifted_r[:-1] = prev[1:]
        shifted_l[0] = prev[0]
        shifted_r[-1] = prev[-1]
        stacked = np.vstack([shifted_l, prev, shifted_r])  # (3, w)
        choice = np.argmin(stacked, axis=0)  # 0=左移 1=直行 2=右移
        dp = stacked[choice, np.arange(w)] + cost[y]
        back[y] = choice
    # 回溯（正向在边界处 clamp 过：choice 0/2 在 x=0/w-1 时等价直行，回溯对应 clamp）
    path = np.zeros(h, dtype=np.int64)
    path[-1] = int(np.argmin(dp))
    for y in range(h - 1, 0, -1):
        c = back[y, path[y]]
        path[y - 1] = np.clip(path[y] + (c - 1), 0, w - 1)  # choice 0→-1, 1→0, 2→+1
    return path


def _plan_axis(a: np.ndarray, skip_px: int, window: int) -> tuple:
    """
    为一条轴做寻径规划：试多个相位候选 Δ，选拼接差异最小的。
    返回 (delta, 每行/列边界 t, join_diff)。
    """
    h, w, _ = a.shape
    K = max(24, min(window, w // 4))
    t0 = w // 2
    rows = np.arange(h)
    d_max = max(2, w // 64)
    deltas = sorted({d for d in (skip_px - 4, skip_px - 2, skip_px, skip_px + 2, skip_px + 4) if 2 <= d <= d_max})
    if not deltas:
        deltas = [d_max]

    i_lo = np.arange(t0 - 1 - K, t0 - 1 + K + 1) % w
    best = None
    for d in deltas:
        i_hi = (i_lo + 1 + d) % w
        D = np.abs(a[rows[:, None], i_lo[None, :]].astype(np.int16)
                   - a[rows[:, None], i_hi[None, :]].astype(np.int16)).mean(axis=2)  # (h, 2K+1)
        path = _min_path(D)
        m = float(D[rows, path].mean())
        if best is None or m < best[0]:
            best = (m, d, path)
    join_diff, delta, path = best
    t = np.clip((t0 - 1 - K + path + 1).astype(np.int64), 2, w - 2)
    return delta, t, join_diff


def _quilt_axis(arr: np.ndarray, skip_px: int, window: int, axis: int) -> tuple:
    """
    沿 axis 做两片相位错位拼接 + 逐行 min-cut 寻径 + 宽羽化。
    axis=1: 竖缝（左右两片，逐行选边界）；axis=0: 横缝（上下两片，逐列选边界）。
    返回 (拼接结果, 实际走径的拼接对平均差异)。
    """
    if axis == 0:
        arr = arr.transpose(1, 0, 2)  # 转成统一按"竖缝"处理，最后转回
    h, w, _ = arr.shape
    delta, t, join_diff = _plan_axis(arr, skip_px, window)
    t = t.astype(np.int32)

    # 组装：x < t(y) 取左片 A[y, x]；x >= t(y) 取右片 A[y, (x+delta)%w]
    # int32 控制内存（4724×7087 全尺寸下 int64 会到 268MB）
    rows = np.arange(h)
    cols = np.arange(w, dtype=np.int32)
    src = np.where(cols[None, :] < t[:, None], cols[None, :], (cols[None, :] + delta) % w)
    out = arr[rows[:, None], src]  # uint8

    # 宽羽化：缝两侧 ±f/2 内，左片（相位 x）与右片（相位 x+delta）线性过渡。
    # 两侧内容仅差 delta px 相位（相邻内容，非镜像关系）——混合只是局部轻微柔化，
    # 不会重现旧版 offset 宽幅淡化"双影蝴蝶"（那是全图远处镜像列的混合）。
    # 切到花朵时表现为边缘淡出而非拦腰截断。
    f = max(12, 3 * delta)
    half = f // 2
    for k in range(-half, half + 1):
        alpha = (k + half) / f  # 0=纯左片 … 1=纯右片
        x = np.clip(t + k, 0, w - 1)
        left_val = arr[rows, x].astype(np.float32)
        right_val = arr[rows, (x + delta) % w].astype(np.float32)
        out[rows, x] = np.clip(left_val * (1.0 - alpha) + right_val * alpha + 0.5, 0, 255).astype(np.uint8)

    if axis == 0:
        out = out.transpose(1, 0, 2)
    return np.ascontiguousarray(out), join_diff


def quilt_offset(img: Image.Image, skip_px: int = 6, window: int = 200) -> Image.Image:
    """四方连续：竖向 + 横向两轮 相位错位+静区寻径 拼接（确定性，替代 AI inpaint）"""
    arr = np.asarray(img.convert("RGB"))
    arr, join_v = _quilt_axis(arr, skip_px, window, axis=1)
    logger.info(f"  竖缝寻径拼接差异 {join_v:.1f}（相位错位 {skip_px}px，确定性算法，无 AI）")
    arr, join_h = _quilt_axis(arr, skip_px, window, axis=0)
    logger.info(f"  横缝寻径拼接差异 {join_h:.1f}")
    return Image.fromarray(arr)


def seam_join_quality(img: Image.Image, skip_px: int = 6, window: int = 200) -> tuple:
    """
    质检用：当前图若按本算法在窗口内寻径，能达到的最优拼接差异（竖/横）。
    返回 (竖缝最优均值差, 横缝最优均值差, 图内基线)。拼接对差异 ≈ 基线 → 接好了。
    """
    arr = np.asarray(img.convert("RGB"))
    results = []
    base = None
    for axis in (1, 0):
        a = arr if axis == 1 else arr.transpose(1, 0, 2)
        _, _, best = _plan_axis(a, skip_px, window)
        results.append(best)
        # 基线：内部随机相邻列差异（避开寻径窗口）
        w = a.shape[1]
        inner_x = np.clip(np.linspace(w // 8, w - w // 8, 64).astype(int), 1, w - 2)
        base = float(np.abs(a[:, inner_x].astype(np.int16) - a[:, inner_x - 1].astype(np.int16)).mean())
    return results[0], results[1], base


# ---------------------------------------------------------------------------
# 周期窗口搜索（四方连续治本，2026-10-11）
#
# quilt 只能让"像素衔接"，补不回 SDXL 画布裁掉的半朵花——大母题在四边被切时，
# 平铺出"切腰的花"，人眼判定没接好（复古波浪植物纹实测）。
#
# 出路：SDXL 对 all-over 纹样常画出**准周期**内容——存在 p 使 A[x+p]≈A[x]。
# 以 p 为窗口裁出的图天然严格周期：平铺无缝，且边缘被切的花的延续内容
# 已因周期性复制进窗口内 → **每朵花完整**。这是纺织设计师"找回位"的手法。
# 找不到可接受周期（大 focal 母题、非重复构图）时返回 None，回退整图 quilt。
# ---------------------------------------------------------------------------


def _box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """numpy 均值盒滤波（借助 cumsum，O(n)），k 为奇数"""
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ap = np.pad(a, ((pad + 1, pad), (pad + 1, pad)), mode="edge")
    c = np.cumsum(np.cumsum(ap, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = a.shape
    return (c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]) / (k * k)


def _axis_period(
    gray: np.ndarray,
    axis: int,
    min_frac: float,
    max_frac: float,
    mask_lo: int = None,
    mask_hi: int = None,
    candidates=None,
) -> tuple:
    """
    单轴周期搜索：返回 (best_p, best_residual, baseline)。
    residual(p) = mean|A[x+p]-A[x]|（在高通图上算，排除慢漂移/明暗渐变干扰）。
    mask_lo/hi: 屏蔽落在该索引区间内的比较对（旧产物中央缝带污染时用）。
    candidates: 显式给定候选周期列表（纵轴按纵横比过滤后传入）。
    """
    n = gray.shape[1] if axis == 1 else gray.shape[0]
    if candidates is not None:
        cand = [p for p in candidates if 8 <= p < n]
        if not cand:
            return None, np.inf, None
    else:
        lo, hi = max(8, int(n * min_frac)), max(9, int(n * max_frac))
        cand = range(lo, hi + 1)

    # 基线：相邻差（同样按掩码剔除）
    if axis == 1:
        dbase = np.abs(gray[:, 1:] - gray[:, :-1])
    else:
        dbase = np.abs(gray[1:, :] - gray[:-1, :])
    if mask_lo is not None:
        idx = np.arange(n - 1)
        m = ~((idx >= mask_lo) & (idx <= mask_hi))
        if axis == 1:
            baseline = float(dbase[:, m].mean()) + 1e-6
        else:
            baseline = float(dbase[m, :].mean()) + 1e-6
    else:
        baseline = float(dbase.mean()) + 1e-6

    best_p, best_r = None, np.inf
    for p in cand:
        if axis == 1:
            d = np.abs(gray[:, p:] - gray[:, :-p])  # (h, n-p), d[:,x]=|A[x+p]-A[x]|
        else:
            d = np.abs(gray[p:, :] - gray[:-p, :])  # (n-p, w)
        if mask_lo is not None:
            x = np.arange(n - p)
            bad = ((x >= mask_lo) & (x <= mask_hi)) | (
                (x + p >= mask_lo) & (x + p <= mask_hi)
            )
            keep = ~bad
            if axis == 1:
                r = float(d[:, keep].mean())
            else:
                r = float(d[keep, :].mean())
        else:
            r = float(d.mean())
        if r < best_r:
            best_p, best_r = p, r
    return best_p, best_r, baseline


def find_periodic_tile(
    img: Image.Image,
    min_frac: float = 0.55,
    max_frac: float = 0.97,
    aspect: float = 1.5,
    aspect_tol: float = 0.06,
    accept_ratio: float = 2.0,
    max_dim: int = 512,
    mask_center: bool = False,
) -> tuple:
    """
    在生成图上搜索周期窗口。命中 → 返回 (窗口图, info)；不命中 → (None, info)。
    info: {px, py, ratio_x, ratio_y}（低分辨率坐标下的周期值与残差比）。

    aspect: 期望纵横比 h/w（=1.5 对应瓦栏 2:3），纵轴候选限制在 aspect±tol 内，
    避免 upscale 拉伸超过 6%。
    mask_center: 屏蔽各轴中央 ±1/4 区间的比较对——用于已经过中央缝合的成品图
    （校准/质检场景）；生产流程在缝合前调用原始生成图，无需屏蔽。
    """
    arr = np.asarray(img.convert("RGB"))
    H, W = arr.shape[:2]
    scale = min(1.0, max_dim / max(H, W))
    if scale < 1.0:
        # BOX=面积平均，缩小不产生摩尔纹；BILINEAR 隔点采样会搅掉周期性（实测教训）
        small = np.asarray(
            img.convert("RGB").resize((int(W * scale), int(H * scale)), Image.BOX)
        )
    else:
        small = arr
    gray0 = small.mean(axis=2).astype(np.float32)
    sh, sw = gray0.shape
    # 高通：去慢漂移/明暗渐变（SDXL 图常有整体渐变，会抬高周期残差淹没真周期）
    gray = gray0 - _box_blur(gray0, max(9, max(sh, sw) // 14))

    def _band(n):
        return (n // 4, n - n // 4) if mask_center else (None, None)

    mx_lo, mx_hi = _band(sw)
    my_lo, my_hi = _band(sh)

    # 横轴：自由搜最优周期
    px, res_x, base_x = _axis_period(gray, axis=1, min_frac=min_frac, max_frac=max_frac,
                                     mask_lo=mx_lo, mask_hi=mx_hi)
    if px is None:
        return None, {"ratio_x": 99.0, "ratio_y": 99.0}
    # 纵轴：候选限制纵横比（py/px ≈ aspect），选残差最小者
    lo, hi = max(8, int(sh * min_frac)), max(9, int(sh * max_frac))
    cand = [
        p for p in range(lo, hi + 1)
        if aspect * (1 - aspect_tol) <= p / px <= aspect * (1 + aspect_tol)
    ]
    py, res_y, base_y = _axis_period(gray, axis=0, min_frac=min_frac, max_frac=max_frac,
                                     mask_lo=my_lo, mask_hi=my_hi, candidates=cand)
    if py is None:
        return None, {"ratio_x": res_x / base_x, "ratio_y": 99.0}

    ratio_x, ratio_y = res_x / base_x, res_y / base_y
    info = {
        "px": px, "py": py,
        "ratio_x": ratio_x, "ratio_y": ratio_y,
    }
    if ratio_x > accept_ratio or ratio_y > accept_ratio:
        return None, info

    # 窗口起点：找 wrap 衔接最自然处（右边缘列 ≈ 左边缘列的前一列，同 quilt 拼接对哲学）
    def _best_offset(period: int, n: int, axis: int) -> int:
        if n - period < 3:
            return 0
        step = max(1, (n - period) // 128)
        costs = []
        for x0 in range(1, n - period, step):
            if axis == 1:
                c = float(np.abs(gray[:, x0 + period - 1] - gray[:, x0 - 1]).mean())
            else:
                c = float(np.abs(gray[x0 + period - 1, :] - gray[x0 - 1, :]).mean())
            costs.append((c, x0))
        return min(costs)[1]

    x0 = _best_offset(px, sw, axis=1)
    y0 = _best_offset(py, sh, axis=0)

    # 全分辨率裁剪
    fx0, fy0 = int(round(x0 / scale)), int(round(y0 / scale))
    fpx = int(round(px / scale))
    fpy = int(round(py / scale))
    fx0 = min(fx0, W - fpx)
    fy0 = min(fy0, H - fpy)
    tile = img.convert("RGB").crop((fx0, fy0, fx0 + fpx, fy0 + fpy))
    logger.info(
        f"  周期窗口: {fpx}×{fpy} @({fx0},{fy0}) "
        f"残差比 竖{ratio_x:.2f}/横{ratio_y:.2f}（< {accept_ratio} 判为准周期，母题完整）"
    )
    return tile, info
