"""
真实趋势采集 — 通过 ego-browser 打开真实浏览器抓取流行花型数据。

站点实测结论（2026-10，国内网络）：
- Pinterest / Behance：直连与浏览器均被墙（ERR_CONNECTION_CLOSED），弃用
- 瓦栏公开区（walanwalan.com/designs）：✅ 可访问 — 平台真实在售花型+价格，最强趋势信号
- POP服装趋势网（pop-fashion.com）：✅ 可访问 — 中文趋势资讯标题

流程：
1. Python 构造 JS 脚本 → subprocess 调 `ego-browser nodejs -e <js>`
2. JS 逐站点打开、滚动、收集（站点专属选择器）
3. stdout/stderr 中的 TREND_JSON: 行传回 Python
4. Python 下载图片供视觉 LLM 分析（瓦栏 CDN 可直连）
"""

import json
import logging
import subprocess
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
}

SITE_URLS = {
    "walan": ["https://www.walanwalan.com/designs/all/page{page}/"],
    "popfashion": ["https://www.pop-fashion.com/"],
    # 保留 pinterest 定义：被墙时优雅失败
    "pinterest": [],
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
for (const urlTpl of URLS) {
  const url = urlTpl.replace("{page}", "1");
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


def _build_js(site: str, browser_cfg: dict) -> str:
    js = JS_TEMPLATE
    js = js.replace("__SITE__", json.dumps(site))
    js = js.replace("__URLS__", json.dumps(SITE_URLS.get(site, [])))
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


def _collect_walan(config: dict) -> list:
    """瓦栏公开区 → items（含设计号/价格/图片），下载图片供视觉分析"""
    browser_cfg = (config.get("trend", {}) or {}).get("browser", {})
    js = _build_js("walan", browser_cfg)
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
    logger.info(f"  瓦栏公开区: {len(items)} 个在售花型")

    # 按配置下载数量均匀取样下载
    download_dir = Path(browser_cfg.get("download_dir", "output/trends/images"))
    max_download = int(browser_cfg.get("max_download", 12))
    step = max(1, len(items) // max(max_download, 1))
    sampled = items[::step][: max_download * 2]  # 多备一些，下载失败有余量
    downloaded = _download_images(sampled, download_dir, max_download, prefix="walan")
    logger.info(f"  下载 {len(downloaded)} 张在售花型图供视觉分析 → {download_dir}")
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
    主入口：sites_str 逗号分隔（同 trend.sources[*].keywords 字段复用，填站点名）。
    支持: walan / popfashion / pinterest(会被墙，保留兼容)
    返回 trend items（图片类带本地 image_path）
    """
    sites = [s.strip() for s in sites_str.split(",") if s.strip()]
    all_items = []
    for site in sites:
        try:
            if site == "walan":
                all_items.extend(_collect_walan(config))
            elif site == "pinterest":
                all_items.extend(_collect_pinterest_compat(config))
            elif site in SITE_COLLECTORS:
                all_items.extend(_collect_text_site(site, config))
            else:
                logger.warning(f"未知浏览器趋势源: {site}，可用: {', '.join(SITE_URLS)}")
        except Exception as e:
            logger.warning(f"站点 {site} 采集异常: {e}")
    return all_items


def _collect_pinterest_compat(config: dict) -> list:
    """Pinterest 兼容路径（国内被墙，仅在网络可用时有意义）"""
    browser_cfg = (config.get("trend", {}) or {}).get("browser", {})
    js = _build_js("pinterest", browser_cfg).replace("__URLS__", '["https://www.pinterest.com/search/pins/?q=fabric%20pattern"]')
    js = js.replace("__SITE__", '"pinterest"').replace("__COLLECTOR__", "() => []")
    blocks = _run_ego_browser(js, int(browser_cfg.get("timeout_s", 120)))
    n_ok = sum(1 for b in blocks if b.get("ok"))
    if not n_ok:
        logger.warning("  Pinterest 不可达（国内网络限制），建议改用 walan/popfashion 源")
    return []


def _download_images(items: list, download_dir: Path, max_download: int, prefix: str = "trend") -> list:
    """下载图片到本地，返回成功下载的 item 列表（含 image_path）"""
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for item in items:
        if len(downloaded) >= max_download:
            break
        url = item.get("src", "")
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200 or len(resp.content) < 5000:
                continue
            fp = download_dir / f"{prefix}_{len(downloaded) + 1:02d}.jpg"
            fp.write_bytes(resp.content)
            item["image_path"] = str(fp)
            downloaded.append(item)
        except Exception as e:
            logger.debug(f"下载失败 {url}: {e}")
    return downloaded
