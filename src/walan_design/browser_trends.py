"""
真实趋势采集 — 通过 ego-browser 打开真实浏览器抓取流行花型数据。

站点实测结论（2026-10）：
- Pinterest：✅ 代理可用，返回图片+描述，国际趋势最丰富的信号
- Behance：需代理
- Patternbank：偶尔可达，不稳定
- 瓦栏公开区（walanwalan.com/designs）：✅ 可访问 — 平台真实在售花型+价格
- POP服装趋势网（pop-fashion.com）：✅ 可访问 — 中文趋势资讯标题

流程：
1. Python 构造 JS 脚本 → subprocess 调 `ego-browser nodejs -e <js>`
2. JS 逐站点打开、滚动、收集（站点专属选择器）
3. stdout/stderr 中的 TREND_JSON: 行传回 Python
4. Python 下载图片 + 采集去重（pHash 比已存图）→ 供视觉 LLM 分析
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# 站点专属收集逻辑（浏览器端 evaluate 执行）
SITE_COLLECTORS = {
    # 瓦栏公开区：设计卡片 = 图 + 价格区间（在售花型，最强信号）
    "walan": r"""
      (maxPerSite) => {
        const cards = Array.from(document.querySelectorAll(".design_grid"));
        return cards.slice(0, maxPerSite).map((card) => {
          const img = card.querySelector("img.img-responsive");
          const link = card.querySelector("a.design_grid_link");
          const price = (card.querySelector(".priceinfo .price") || {}).textContent || "";
          return {
            src: img ? (img.currentSrc || img.src || "") : "",
            alt: (link || {}).name || "design",
            photoid: (link || {}).dataset ? link.dataset.photoid : "",
            price: price.replace(/\s+/g, " ").trim(),
          };
        }).filter((it) => it.src);
      }
    """,
    # POP服装趋势网：趋势文章标题
    "popfashion": r"""
      (maxPerSite) => {
        const items = Array.from(document.querySelectorAll("a"));
        const seen = new Set();
        const out = [];
        for (const a of items) {
          const t = (a.title || a.textContent || "").trim();
          if (t.length >= 6 && t.length <= 60 && !seen.has(t)) {
            seen.add(t);
            out.push({ title: t });
          }
          if (out.length >= maxPerSite) break;
        }
        return out;
      }
    """,
    # Pinterest：搜索页抓取 pin 图 + 描述 + 跳转链接（代理可用）
    "pinterest": r"""
      (maxPerSite) => {
        // Pinterest 新版 DOM：grid 内有大量 img
        const imgs = Array.from(document.querySelectorAll("img"));
        const seenSrc = new Set();
        const out = [];
        for (const img of imgs) {
          const src = img.currentSrc || img.src || "";
          if (!src || !src.includes("pinimg")) continue;  // 只要 pinimg CDN 的图
          if (seenSrc.has(src)) continue;
          seenSrc.add(src);
          // 往上找最近的 a 拿 pin 链接和标题
          const parentA = img.closest("a");
          const alt = (img.alt || "").slice(0, 120);
          const href = parentA ? parentA.href : "";
          out.push({ src, alt, href });
          if (out.length >= maxPerSite) break;
        }
        return out;
      }
    """,
}

SITE_URLS = {
    "walan": ["https://www.walanwalan.com/designs/all/page{page}/"],
    "popfashion": ["https://www.pop-fashion.com/"],
    "pinterest": [
        "https://www.pinterest.com/search/pins/?q={keyword}"
    ],
    "behance": ["https://www.behance.net/search?search={keyword}"],
    "patternbank": ["https://patternbank.com/trend-reports/"],
}

JS_TEMPLATE = r"""
const task = await taskSpace("walan-trend-collect");
const page = task.page("p1");
const SITE = __SITE__;
const URLS = __URLS__;
const SCROLLS = __SCROLLS__;
const WAIT_MS = __WAIT_MS__;
const MAX_PER_SITE = __MAX_PER_SITE__;
const COLLECTOR = __COLLECTOR__;
const out = [];
for (const url of URLS) {
  try {
    await page.goto(url);
    await page.waitForLoadState("networkidle").catch(() => {});
    await page.waitForTimeout(WAIT_MS);
    for (let i = 0; i < SCROLLS; i++) {
      await page.mouse.wheel(0, 2400).catch(() => {});
      await page.waitForTimeout(WAIT_MS);
    }
    const items = await page.evaluate(COLLECTOR, MAX_PER_SITE);
    out.push({ site: SITE, url, ok: true, items: items });
    console.error("  [browser] " + url + ": " + items.length + " items");
  } catch (e) {
    out.push({ site: SITE, url, ok: false, error: String(e).slice(0, 200), items: [] });
  }
}
console.log("TREND_JSON:" + JSON.stringify(out));
"""


def _build_js(site: str, browser_cfg: dict, keywords: str = "") -> str:
    js = JS_TEMPLATE
    # Pinterest / Behance 用多关键词，每个关键词一条 URL
    if site == "pinterest" and keywords:
        kw_list = [k.strip() for k in keywords.split(",")[:4]]  # 最多 4 个关键词
        urls = [SITE_URLS[site][0].replace("{keyword}", requests.utils.quote(kw)) for kw in kw_list]
    elif site in SITE_URLS:
        urls = SITE_URLS[site]
    else:
        urls = []
    js = js.replace("__SITE__", json.dumps(site))
    js = js.replace("__URLS__", json.dumps(urls))
    js = js.replace("__SCROLLS__", str(browser_cfg.get("scrolls", 2)))
    js = js.replace("__WAIT_MS__", str(browser_cfg.get("scroll_wait_ms", 1500)))
    js = js.replace("__MAX_PER_SITE__", str(browser_cfg.get("max_images_per_keyword", 60)))
    js = js.replace("__COLLECTOR__", SITE_COLLECTORS.get(site, "() => []"))
    return js


def _parse_trend_json(stdout: str, stderr: str) -> list:
    """TREND_JSON 行可能在 stdout 或 stderr（ego-browser 输出路由不保证）"""
    for stream in (stdout, stderr):
        for line in (stream or "").splitlines():
            if line.startswith("TREND_JSON:"):
                return json.loads(line[len("TREND_JSON:") :])
    return []


def _run_ego_browser(js: str, timeout_s: int) -> list:
    result = subprocess.run(
        ["ego-browser", "nodejs", "-e", js],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stdout, stderr = result.stdout or "", result.stderr or ""
    if result.returncode != 0:
        logger.warning(f"ego-browser 退出码 {result.returncode}: {stderr[:200]}")
    blocks = _parse_trend_json(stdout, stderr)
    if not blocks:
        logger.warning(f"未找到 TREND_JSON 输出。stdout 前200字符: {stdout[:200]!r} stderr 前200字符: {stderr[:200]!r}")
    return blocks


def _walan_page_offset() -> int:
    """瓦栏翻页：用一年中的第几天 mod 400（瓦栏总共约 438 页）"""
    today = datetime.now()
    day_of_year = int(today.strftime("%j"))
    return day_of_year % 400


def _collect_walan(config: dict) -> list:
    """瓦栏公开区 → items（含设计号/价格/图片），下载图片供视觉分析"""
    browser_cfg = (config.get("trend", {}) or {}).get("browser", {})
    # 翻页轮换
    page_offset = _walan_page_offset() if browser_cfg.get("walan_page_rotation", True) else 0
    urls = [f"https://www.walanwalan.com/designs/all/page{1 + page_offset}/"]
    js = JS_TEMPLATE
    js = js.replace("__SITE__", json.dumps("walan"))
    js = js.replace("__URLS__", json.dumps(urls))
    js = js.replace("__SCROLLS__", str(browser_cfg.get("scrolls", 2)))
    js = js.replace("__WAIT_MS__", str(browser_cfg.get("scroll_wait_ms", 1500)))
    js = js.replace("__MAX_PER_SITE__", str(browser_cfg.get("max_images_per_keyword", 60)))
    js = js.replace("__COLLECTOR__", SITE_COLLECTORS["walan"])
    blocks = _run_ego_browser(js, int(browser_cfg.get("timeout_s", 180)))

    items = []
    for block in blocks:
        if not block.get("ok"):
            logger.warning(f"  瓦栏采集失败: {block.get('error', '?')}")
            continue
        for it in block.get("items", []):
            price = it.get("price", "")
            desc = f"瓦栏在售花型 #{it.get('photoid', '')} 价格{price or '未知'}"
            items.append(
                {
                    "source": "walan_live",
                    "keyword": "walan_public",
                    "description": desc,
                    "photoid": it.get("photoid", ""),
                    "price": price,
                    "src": it.get("src", ""),
                }
            )
    logger.info(f"  瓦栏公开区 (page{1+page_offset}): {len(items)} 个在售花型")

    # 按配置下载数量均匀取样下载
    download_dir = Path(browser_cfg.get("download_dir", "output/trends/images"))
    max_download = int(browser_cfg.get("max_download", 12))
    threshold = int(browser_cfg.get("dedup_phash_threshold", 6))
    step = max(1, len(items) // max(max_download, 1))
    sampled = items[::step][: max_download * 2]  # 多备一些，下载失败+去重有余量
    downloaded = _download_with_dedup(sampled, download_dir, max_download, prefix="walan", phash_threshold=threshold)
    logger.info(f"  下载 {len(downloaded)} 张在售花型图（已去重）→ {download_dir}")
    return downloaded


def _collect_pinterest(config: dict, keywords: str) -> list:
    """Pinterest 搜索页 → items（含 pinimg CDN 图 + alt 描述）"""
    browser_cfg = (config.get("trend", {}) or {}).get("browser", {})
    js = _build_js("pinterest", browser_cfg, keywords)
    blocks = _run_ego_browser(js, int(browser_cfg.get("timeout_s", 180)))

    items = []
    for block in blocks:
        if not block.get("ok"):
            logger.warning(f"  Pinterest 采集失败: {block.get('error', '?')}")
            continue
        for it in block.get("items", []):
            src = it.get("src", "")
            if not src:
                continue
            # 跳过小图/非 pinimg CDN
            if "pinimg" not in src:
                continue
            alt = (it.get("alt") or "")[:100]
            href = it.get("href", "")
            items.append(
                {
                    "source": "pinterest",
                    "keyword": "pinterest_search",
                    "description": alt or "Pinterest pin",
                    "src": src,
                    "href": href,
                }
            )
    logger.info(f"  Pinterest: {len(items)} 张 pinimg CDN 图片")

    if not items:
        return items

    download_dir = Path(browser_cfg.get("download_dir", "output/trends/images"))
    max_download = int(browser_cfg.get("max_download", 12))
    threshold = int(browser_cfg.get("dedup_phash_threshold", 6))
    # 均匀取样
    step = max(1, len(items) // max(max_download, 1))
    sampled = items[::step][: max_download * 2]
    downloaded = _download_with_dedup(sampled, download_dir, max_download, prefix="pin", phash_threshold=threshold)
    logger.info(f"  下载 {len(downloaded)} 张 Pinterest 图（已去重）→ {download_dir}")
    return downloaded


def _collect_text_site(site: str, config: dict) -> list:
    """文字类趋势站（POP服装趋势网等）→ 标题 items"""
    browser_cfg = (config.get("trend", {}) or {}).get("browser", {})
    js = _build_js(site, browser_cfg)
    blocks = _run_ego_browser(js, int(browser_cfg.get("timeout_s", 120)))

    items = []
    for block in blocks:
        if not block.get("ok"):
            logger.warning(f"  {site} 采集失败: {block.get('error', '?')}")
            continue
        for it in block.get("items", []):
            if it.get("title"):
                items.append({"source": site, "keyword": site, "description": it["title"]})
    logger.info(f"  {site}: {len(items)} 条趋势标题")
    return items


def collect_via_browser(sites_str: str, config: dict) -> list:
    """
    主入口：sites_str 逗号分隔（复用 trend.sources[*].keywords 字段填站点名）。
    支持: walan / popfashion / pinterest / behance / patternbank
    返回 trend items（图片类带本地 image_path）
    """
    sites = [s.strip() for s in sites_str.split(",") if s.strip()]
    all_items = []
    for site in sites:
        try:
            if site == "walan":
                all_items.extend(_collect_walan(config))
            elif site == "pinterest":
                # Pinterest 需要多关键词，从 trend.sources 里找
                kw = "pattern design, fabric pattern, floral pattern, geometric pattern"
                for src in config.get("trend", {}).get("sources", []):
                    if src.get("name") == "pinterest":
                        kw = src.get("keywords", kw)
                        break
                all_items.extend(_collect_pinterest(config, kw))
            elif site in SITE_COLLECTORS:
                all_items.extend(_collect_text_site(site, config))
            else:
                logger.warning(f"未知浏览器趋势源: {site}，可用: {', '.join(SITE_URLS)}")
        except Exception as e:
            logger.warning(f"站点 {site} 采集异常: {e}")
    return all_items


# ============================================================
# 采集去重（和已存图片 pHash 比对）
# ============================================================

def _compute_phash_from_path(img_path: str) -> str:
    """从文件计算 pHash（懒加载 dedup_checker）"""
    try:
        from PIL import Image
        from walan_design.dedup_checker import compute_phash

        img = Image.open(img_path).convert("RGB")
        return compute_phash(img)
    except Exception:
        return ""


def _load_existing_phashes(download_dir: Path, prefixes: list[str]) -> dict[str, str]:
    """加载已有图片的 {filename: phash} 映射（缓存到 .phash_cache.json）"""
    cache_path = download_dir / ".phash_cache.json"
    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            cache = {}

    # 扫目录里的图片
    existing = sorted(download_dir.glob("*.jpg"))
    for fp in existing:
        if fp.name not in cache:
            cache[fp.name] = _compute_phash_from_path(str(fp))

    cache_path.write_text(json.dumps(cache))
    return cache


def _download_with_dedup(
    items: list,
    download_dir: Path,
    max_download: int,
    prefix: str = "trend",
    phash_threshold: int = 6,
) -> list:
    """下载图片 + pHash 去重（和已存图比对，汉明距离 ≤ threshold 的跳过）"""
    from walan_design.dedup_checker import hamming_distance

    download_dir.mkdir(parents=True, exist_ok=True)
    existing_cache = _load_existing_phashes(download_dir, [prefix])
    existing_hashes = [h for h in existing_cache.values() if h]

    downloaded = []
    deduped_skipped = 0
    for item in items:
        if len(downloaded) >= max_download:
            break
        url = item.get("src", "")
        if not url:
            continue
        # 先用 URL 指纹检查本地是否已经下载过
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        local_name = f"{prefix}_{url_hash}.jpg"
        local_path = download_dir / local_name
        if local_path.exists():
            # 已在本地（可能上次下过），直接用
            item["image_path"] = str(local_path)
            downloaded.append(item)
            continue

        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200 or len(resp.content) < 5000:
                continue
            fp = download_dir / local_name
            fp.write_bytes(resp.content)

            # 采集去重：和已存图比 pHash
            import numpy as np
            from PIL import Image
            from walan_design.dedup_checker import compute_phash

            try:
                img = Image.open(fp).convert("RGB")
                new_hash = compute_phash(img)
                # 和已有所有图比
                is_dup = any(hamming_distance(new_hash, eh) <= phash_threshold for eh in existing_hashes)
                if is_dup:
                    fp.unlink()  # 重复，删掉
                    deduped_skipped += 1
                    continue
                # 通过去重 → 加入缓存
                existing_hashes.append(new_hash)
                existing_cache[local_name] = new_hash
            except Exception as e:
                logger.debug(f"  pHash 去重跳过（文件异常）: {e}")

            item["image_path"] = str(fp)
            downloaded.append(item)
        except Exception as e:
            logger.debug(f"下载失败 {url}: {e}")

    if deduped_skipped:
        logger.info(f"  采集去重: 跳过 {deduped_skipped} 张重复图")
    # 更新缓存
    cache_path = download_dir / ".phash_cache.json"
    cache_path.write_text(json.dumps(existing_cache))
    return downloaded

