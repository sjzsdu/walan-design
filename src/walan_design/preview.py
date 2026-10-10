"""四方连续平铺预览 — 直接弹窗查看，不在项目目录生成文件

把花型按真实物理尺寸（瓦栏标准 40×60cm/格）平铺到指定物理大小画布
（默认 1.5m 宽 × 1m 高，模拟真实面料门幅），用 macOS 预览打开供人工检查
接回位质量：接缝是否可见、镜像对称痕迹、融合带是否突兀。

图片仅写入系统临时目录（/tmp/walan_preview/），项目目录零文件；
支持三种输入：PNG 路径 / 花型目录 / 瓦栏设计号。
"""

import io
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("walan_design.preview")

TILE_W_CM = 40  # 瓦栏标准花型尺寸
TILE_H_CM = 60
PREVIEW_ROOT = Path(tempfile.gettempdir()) / "walan_preview"


def _needs_seamless(img: Image.Image) -> tuple:
    """判断是否需要接回位：边缘差异 vs 图内相邻列基线（对 JPEG 噪声/分辨率鲁棒）

    已接回位的图，边缘列就像普通内部相邻列（差异≈基线）；
    未接回位的图，边缘差异远超基线。返回 (需要处理, 左右差异, 上下差异)。
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    left = np.asarray(rgb.crop((0, 0, 1, h)), dtype=np.int16)
    right = np.asarray(rgb.crop((w - 1, 0, w, h)), dtype=np.int16)
    top = np.asarray(rgb.crop((0, 0, w, 1)), dtype=np.int16)
    bottom = np.asarray(rgb.crop((0, h - 1, w, h)), dtype=np.int16)
    lr = abs(left - right).mean()
    tb = abs(top - bottom).mean()
    # 基线：缩到 256 后内部相邻列的平均差异（含压缩噪声）
    small = np.asarray(rgb.resize((256, 256)), dtype=np.int16)
    base = abs(small[:, 1:, :] - small[:, :-1, :]).mean()
    tol = 2.0 * base + 5.0
    return (lr > tol or tb > tol), lr, tb


def tile_image(img: Image.Image, width_m: float = 1.5, height_m: float = 1.0, px_per_cm: int = 20) -> Image.Image:
    """未接回位的图先做四方连续处理，然后按 40×60cm/格 平铺到物理尺寸画布"""
    need, lr, tb = _needs_seamless(img)
    if need:
        from walan_design.ai_designer import make_seamless

        logger.info(f"  未接回位（L/R={lr:.1f} T/B={tb:.1f}），先做四方连续处理")
        img = make_seamless(img)
    else:
        logger.info(f"  已接回位（L/R={lr:.1f} T/B={tb:.1f}），直接平铺")

    cw, ch = int(width_m * 100 * px_per_cm), int(height_m * 100 * px_per_cm)
    tw, th = TILE_W_CM * px_per_cm, TILE_H_CM * px_per_cm
    tile = img.convert("RGB").resize((tw, th), Image.LANCZOS)

    canvas = Image.new("RGB", (cw, ch))
    cols, rows = cw // tw + 1, ch // th + 1
    for r in range(rows):
        for c in range(cols):
            canvas.paste(tile, (c * tw, r * th))
    return canvas.crop((0, 0, cw, ch))


def open_previews(named_images: list):
    """逐张平铺后用系统看图程序打开（仅写系统临时目录，项目目录零文件）"""
    PREVIEW_ROOT.mkdir(exist_ok=True)
    # 清理超过 24 小时的旧临时目录
    now = time.time()
    for d in PREVIEW_ROOT.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > 86400:
            shutil.rmtree(d, ignore_errors=True)

    out_dir = Path(tempfile.mkdtemp(prefix=f"{int(now)}_", dir=PREVIEW_ROOT))
    paths = []
    for name, img in named_images:
        p = out_dir / f"{name}.jpg"
        img.save(p, quality=88)
        paths.append(p)

    if paths:
        subprocess.run(["open", *[str(x) for x in paths]], check=False)
    return paths


def resolve_local_design(design_id: str, runs_dir: str) -> dict:
    """按瓦栏设计号找本地花型目录（upload_result.json 里记录了设计号）

    返回 {dir, pngs, variant}，找不到返回 None。
    设计号对应唯一上传的变体（如 赤陶热带叶_1 → design_1.png）。
    """
    runs = Path(runs_dir)
    if not runs.exists():
        return None
    # 不能用 \b 边界：中文"进"也算 word 字符，"1680457进入..."匹配不上；
    # 用数字前后断言：前面和后面都不是数字即可
    pat = re.compile(rf"(?<!\d){re.escape(design_id)}(?!\d)")
    for d in sorted(runs.iterdir()):
        ur = d / "upload_result.json"
        if not (d.is_dir() and ur.exists()):
            continue
        try:
            if not pat.search(ur.read_text(encoding="utf-8")):
                continue
        except Exception:
            continue

        # 从 upload.json 的 psd_path 解析变体号（{标题}_N.psd → design_N.png）
        variant = None
        task_file = d / "upload.json"
        if task_file.exists():
            try:
                m = re.search(r"_([0-9]+)\.psd", task_file.read_text(encoding="utf-8"))
                if m:
                    variant = m.group(1)
            except Exception:
                pass
        if variant:
            pngs = [d / f"design_{variant}.png"]
            pngs = [p for p in pngs if p.exists()] or sorted(d.glob("design_*.png"))
        else:
            pngs = sorted(d.glob("design_*.png"))
        if not pngs:
            pngs = sorted(d.glob("*.png"))
        if pngs:
            return {"dir": d, "pngs": pngs, "variant": variant}
    return None


def fetch_walan_image(design_id: str, state_path: str) -> Image.Image:
    """按设计号从瓦栏控制台抓缩略图（300×450，低分辨率，仅够粗看重复结构）"""
    from playwright.sync_api import sync_playwright

    from walan_design.upload_executor import _goto_designs_list

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(storage_state=state_path)
        page = ctx.new_page()
        try:
            _goto_designs_list(page)
            if "login" in page.url.lower():
                raise RuntimeError("登录已失效，请先运行 walan-design login")
            img = page.query_selector(f'img[src*="{design_id}"]')
            if not img:  # 已下架的作品在 off 视图
                _goto_designs_list(page, view="off")
                img = page.query_selector(f'img[src*="{design_id}"]')
            if not img:
                raise RuntimeError(f"瓦栏第 1 页（全部/已下架视图）均未找到设计 {design_id}")
            src = img.get_attribute("src")
            r = ctx.request.get(src)
            if r.status != 200:
                raise RuntimeError(f"缩略图下载失败 HTTP {r.status}")
            return Image.open(io.BytesIO(r.body())).convert("RGB")
        finally:
            browser.close()
