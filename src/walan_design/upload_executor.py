"""瓦栏 Playwright 上传执行器

登录态方案（参考 Sau CLI）：
  1. 首次运行 `walan-design login`：headed 模式开浏览器 → 用户手动登录瓦栏
     → 等 URL 跳转到 console 即成功 → context.storage_state() 保存到
     output/state/walan_session.json
  2. 后续 upload 自动加载 storage_state，headless 直接进入已登录页面，
     无需重复登录（cookie 过期会被检测到，自动提示重新 login）
  3. 也支持直接复用本机 Chrome 用户数据目录（更快，但 Chrome 必须已关闭）
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger("walan_design.upload")

STATE_FILE = "output/state/walan_session.json"
CHROME_PROFILE_DIR = "~/Library/Application Support/Google/Chrome/Default"


def _get_state_path() -> Path:
    return Path(STATE_FILE)


# ---------- 登录态管理 ----------

def save_login_state(state_path: str = STATE_FILE, headless: bool = False) -> Path:
    """打开 headed 浏览器让用户手动登录，保存 storage_state"""
    from playwright.sync_api import sync_playwright

    sp = Path(state_path)
    sp.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--lang=zh-CN"],
        )
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.walanwalan.com/console/designs/all/page1/")
        print("\n⏳ 请在浏览器中完成瓦栏登录（微信扫码/手机号登录均可）")
        print("登录成功进入控制台后，回到终端按 Enter 继续...")
        input()
        context.storage_state(path=str(sp))
        browser.close()

    print(f"✅ 登录态已保存到 {sp}")
    return sp


def check_login_status(state_path: str = STATE_FILE) -> dict:
    """检查登录态有效性，返回 {valid, reason, pending_count, works}

    works: 控制台能看到的最新作品列表（设计号+标题，最多 10 条）
    pending_count: "公开区(待审核)"状态的作品数
    """
    from playwright.sync_api import sync_playwright

    sp = Path(state_path)
    if not sp.exists():
        return {"valid": False, "reason": f"登录态文件不存在: {sp}（请先运行 walan-design login）"}

    age_min = (time.time() - sp.stat().st_mtime) / 60
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=str(sp))
        page = context.new_page()
        try:
            page.goto("https://www.walanwalan.com/console/designs/all/page1/", wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(2500)
            if "login" in page.url.lower():
                browser.close()
                return {
                    "valid": False,
                    "reason": f"登录已失效（storage_state 保存于 {age_min:.0f} 分钟前）。请重新运行 walan-design login",
                }

            # 已登录：抓作品状态统计
            # 卡片标题在卡片文本首行（"1680457-赤陶热带叶_1"），img.alt 为空不可用；
            # 侧栏有权威状态计数（全部 440 / 审核中 5 / VIP区 81 / 公开区 354 / 已下架 17）
            info = page.evaluate(
                """() => {
                    const cards = [...document.querySelectorAll('[class*="imggrid_holder_"]')].slice(0, 20);
                    const works = cards.map(c => {
                        const m = c.className.match(/imggrid_holder_(\\d+)/);
                        const lines = (c.innerText || '').split('\\n').map(s => s.trim()).filter(Boolean);
                        let title = '';
                        if (m) {
                            const hit = lines.find(l => l.startsWith(m[1] + '-'));
                            if (hit) title = hit.slice(m[1].length + 1, m[1].length + 31);
                            else if (lines.length) title = lines[0].slice(0, 30);
                        }
                        const status = lines.find(l => /待审核|已下架|VIP区|公开区|新上传|密码区/.test(l)) || '';
                        return {id: m ? m[1] : '', title, status};
                    }).filter(w => w.id);
                    const body = document.body.innerText;
                    const cnt = (re) => { const m = body.match(re); return m ? parseInt(m[1]) : null; };
                    const stats = {
                        all: cnt(/全部\\s*(\\d+)/),
                        reviewing: cnt(/审核中\\s*(\\d+)/),
                        vip: cnt(/VIP区\\s*(\\d+)/),
                        public: cnt(/公开区\\s*(\\d+)/),
                        off: cnt(/已下架\\s*(\\d+)/),
                    };
                    return {works, stats};
                }"""
            )
            browser.close()
            stats = info.get("stats") or {}
            pending = stats.get("reviewing")
            if pending is None:  # 侧栏解析失败时按可见卡片兜底
                pending = sum(1 for w in info.get("works", []) if "待审核" in w.get("status", ""))
            return {
                "valid": True,
                "reason": f"登录有效（storage_state 已保存 {age_min:.0f} 分钟）",
                "pending_count": pending or 0,
                "stats": stats,
                "works": info.get("works", []),
            }
        except Exception as e:
            browser.close()
            return {"valid": False, "reason": f"检查失败: {e}"}


def _goto_designs_list(page, view: str = "all"):
    """打开控制台列表页（瓦栏服务器偶尔响应慢，超时重试一次）

    view: all=全部（默认） / off=已下架（下架后的作品会移出"全部"默认视图）
    """
    url = f"https://www.walanwalan.com/console/designs/{view}/page1/"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
    except Exception:
        page.wait_for_timeout(2000)
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
    # 卡片由 AJAX 异步渲染，必须等卡片出现而不是固定 sleep
    try:
        page.wait_for_selector('[class*="imggrid_holder_"]', timeout=20000)
    except Exception:
        pass  # 该视图无作品是正常的（如已下架视图为空）
    page.wait_for_timeout(800)


def takedown_designs(design_ids: list, state_path: str = STATE_FILE) -> list:
    """下架指定设计号的作品

    瓦栏后台没有删除功能（全页面无删除入口），状态变更只有：
    公开区(2) / 密码区(4) / 已下架(9)。下架 = 从公开区撤回，
    不再对外展示/售卖，待审核的中止审核；可随时重新上架。

    返回 [{id, ok, detail}]
    """
    from playwright.sync_api import sync_playwright

    ids = [str(i).strip() for i in design_ids if str(i).strip()]
    if not ids:
        return []

    sp = Path(state_path)
    if not sp.exists():
        return [{"id": i, "ok": False, "detail": f"登录态文件不存在: {sp}（请先 walan-design login）"} for i in ids]

    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=str(sp))
        page = context.new_page()
        page.on("dialog", lambda d: d.accept())  # 自动接受 confirm/alert 弹窗

        try:
            _goto_designs_list(page)
            if "login" in page.url.lower():
                return [{"id": i, "ok": False, "detail": "登录已失效，请重新运行 walan-design login"} for i in ids]

            for did in ids:
                r = _takedown_one(page, did)
                if not r["ok"] and "未找到" in r["detail"]:
                    # "全部"视图找不到 → 查已下架视图（下架后的作品会移出"全部"默认视图）
                    _goto_designs_list(page, view="off")
                    if page.query_selector(f'[class*="imggrid_holder_{did}"]'):
                        r = {"id": did, "ok": True, "detail": "该作品已处于下架状态，无需重复操作"}
                    else:
                        r = {"id": did, "ok": False, "detail": "全部/已下架视图第 1 页均未找到，请核对设计号（老作品可能翻页）"}
                results.append(r)
                # 每个作品下架后整页刷新过，重新加载列表页保证下次选卡干净
                _goto_designs_list(page)
        except Exception as e:
            for i in ids[len(results):]:
                results.append({"id": i, "ok": False, "detail": f"异常: {e}"})
        finally:
            browser.close()

    return results


def _takedown_one(page, did: str) -> dict:
    """在已加载的列表页下架单个设计"""
    card = page.query_selector(f'[class*="imggrid_holder_{did}"]')
    if not card:
        return {"id": did, "ok": False, "detail": "本页（第 1 页）未找到该设计号，老作品请翻页或核对设计号"}

    card.click()
    page.wait_for_timeout(600)
    sel = page.evaluate("""() => document.querySelector(".txtcardids")?.value || "" """)
    if did not in sel:
        return {"id": did, "ok": False, "detail": f"选卡失败（txtcardids={sel!r}）"}

    clicked = page.evaluate(
        """() => {
            const btns = [...document.querySelectorAll("button, a, input[type=button]")];
            const btn = btns.find(b => b.textContent.trim() === "已下架");
            if (!btn) return false;
            btn.click();
            return true;
        }"""
    )
    if not clicked:
        return {"id": did, "ok": False, "detail": "未找到下架按钮"}

    page.wait_for_timeout(4500)
    # 双重验证：确认文本 + 卡片状态文本
    msg = page.evaluate(
        """(did) => {
            const m = document.body.innerText.match(new RegExp(did + "\\\\s*进入已下架"));
            return m ? m[0] : null;
        }""",
        did,
    )
    st = page.evaluate(
        """(did) => {
            const c = document.querySelector('[class*="imggrid_holder_' + did + '"]');
            return c ? c.innerText : "";
        }""",
        did,
    )
    if msg or "已下架" in st:
        return {"id": did, "ok": True, "detail": msg or "卡片状态已变为已下架"}
    return {"id": did, "ok": False, "detail": "已点击下架按钮但未确认到状态变化，请后台核对"}


def _check_login(page) -> bool:
    """检查当前页面是否已登录（控制台 / 上传页能访问 = 已登录）"""
    try:
        page.goto("https://www.walanwalan.com/console/designs/all/page1/", wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(1500)
        # 已登录能看到 console 相关元素；未登录会跳转到首页或出现登录表单
        title = page.title()
        url = page.url
        if "login" in url.lower() or "登录" in title:
            return False
        # 检查是否有 .imggrid（控制台设计卡片）或 select[name=content]（上传管理表单）
        has_card = page.query_selector(".imggrid, .select[name=content]") is not None
        return has_card or "console" in url
    except Exception:
        return False


# ---------- 上传动作 ----------

def _build_browser(config: dict):
    """启动带登录态的 Playwright 浏览器（同步 API）"""
    from playwright.sync_api import sync_playwright

    state = _get_state_path()
    if not state.exists():
        raise RuntimeError(
            f"未找到登录态文件 {state}\n"
            f"请先运行: walan-design login"
        )

    p = sync_playwright().start()
    browser = p.chromium.launch(
        headless=True,
        args=["--disable-blink-features=AutomationControlled", "--lang=zh-CN"],
    )
    context = browser.new_context(storage_state=str(state))
    return p, browser, context


# content（内容分类）关键词映射：从 tags/title 推断
CONTENT_MAP = [
    (("豹纹", "动物纹", "斑马", "动物"), "动物纹"),
    (("格子",), "格子"),
    (("条纹",), "条纹"),
    (("几何",), "几何"),
    (("卡通",), "卡通"),
    (("民族",), "民族风"),
    (("佩斯利", "腰果"), "佩斯利"),
    (("风景", "山水"), "风景"),
    (("热带", "棕榈"), "热带"),
    (("迷彩",), "迷彩"),
    (("中国风", "青花"), "中国风"),
    (("黑白",), "黑白"),
    (("抽象",), "抽象"),
    (("花", "植物", "叶", "田园", "玫瑰", "百合"), "花卉"),  # 放最后兜底
]


def _pick_content(task: dict) -> str:
    """从 tags/title 推断瓦栏 content 内容分类"""
    text = " ".join(task.get("tags", [])) + " " + task.get("title", "")
    for keywords, content in CONTENT_MAP:
        if any(k in text for k in keywords):
            return content
    return "花卉"


def _set_select_by_text(page, name_attr: str, text: str, required: bool = True):
    """按 option 文本匹配选值"""
    result = page.evaluate(
        """([sel, text]) => {
            const el = document.querySelector(`select[name="${sel}"]`);
            if (!el) return {ok:false, msg:`缺少下拉 ${sel}`};
            const opt = [...el.options].find(o => o.text.trim().includes(text));
            if (!opt) return {ok:false, msg:`下拉 ${sel} 无匹配项: ${text}`};
            el.value = opt.value;
            el.dispatchEvent(new Event("change", {bubbles: true}));
            return {ok:true};
        }""",
        [name_attr, text],
    )
    if result.get("ok"):
        return None
    msg = result.get("msg", "未知错误")
    if required:
        raise RuntimeError(msg)
    logger.warning(f"  {msg}")
    return msg


def _set_tags(page, tags: list):
    """标签多选（select[name=keyword]，value=标签文本）"""
    result = page.evaluate(
        """(tags) => {
            const $kw = document.querySelector('select[name="keyword"]');
            if (!$kw) return {ok:false, msg:"缺少标签下拉"};
            const valid = [];
            [...$kw.options].forEach(o => {
                const v = o.value.trim(), t = o.text.trim();
                if (tags.includes(v) || tags.includes(t)) valid.push(v || t);
            });
            if (!valid.length) return {ok:false, msg:"无有效标签: " + tags.join("/")};
            // playwright 的 select_option 支持数组
            return {ok:true, valid};
        }""",
        tags,
    )
    if not result.get("ok"):
        raise RuntimeError(result.get("msg", "标签设置失败"))
    valid = result["valid"]
    page.select_option('select[name="keyword"]', valid)
    return valid


def execute_upload(task: dict, config: dict, timeout: int = 600) -> dict:
    """用 Playwright 执行单个瓦栏上传任务，返回 {ok, step, detail}

    task 支持 skip_upload=True：文件已上传到服务器（如上次中断）时，
    直接从新上传列表页选卡片继续管理+定价+发布，避免重复上传。
    """
    try:
        p, browser, context = _build_browser(config)
    except RuntimeError as e:
        return {"ok": False, "step": "登录态", "detail": str(e)}

    page = context.new_page()
    title = task["title"]
    psd_path = task["psd_path"]
    tags = task.get("tags", [])
    category_name = task.get("category_name", "数码花型")
    buyout = str((task.get("price") or {}).get("buyout", 299))
    upload_url = task["upload_url"]
    skip_upload = bool(task.get("skip_upload"))

    try:
        # ---------- 登录态校验 ----------
        if not _check_login(page):
            return {
                "ok": False,
                "step": "登录态过期",
                "detail": "storage_state 已失效，请重新运行 walan-design login",
            }

        # ---------- 第 1 步：打开上传页并上传文件（skip_upload 时跳过） ----------
        if skip_upload:
            # 文件已在服务器：直接开新上传列表页选卡片
            page.goto(
                "https://www.walanwalan.com/console/designs/new/page1/",
                wait_until="domcontentloaded",
            )
            page.wait_for_timeout(2500)
            logger.info(f"  {title}: 跳过上传，从新上传列表继续")
        else:
            page.goto(upload_url, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)

            # file input 可能隐藏在按钮后，先尝试直接 set_input_files
            try:
                page.locator('input[name="file"]').first.set_input_files(psd_path)
            except Exception:
                # 隐藏 input → force 点击按钮触发
                page.click("text=上传设计", timeout=10000)
                page.wait_for_timeout(500)
                page.locator('input[name="file"]').first.set_input_files(psd_path)

            logger.info(f"  {title}: 文件已提交，等待服务器处理...")
            # 上传完成后跳转到新上传列表页（designs/new/...?msg=...已上传）
            page.wait_for_url("**/designs/new/**", timeout=360000)
            page.wait_for_timeout(2500)
            logger.info(f"  {title}: 服务器校验通过，已进入新上传列表")
        # 点第一张卡片（刚上传的设计），管理表单同页展开
        page.wait_for_selector(".imggrid", timeout=30000)
        page.click(".imggrid")
        page.wait_for_selector('select[name="content"]', timeout=20000, state="visible")
        page.wait_for_timeout(1500)

        # ---------- 第 2 步：管理表单 ----------
        # content=内容分类（从 tags 推断）→ 选完级联出现 apply（应用）下拉 → 选女装
        _set_select_by_text(page, "content", _pick_content(task))
        page.wait_for_selector('select[name="apply"]', timeout=10000, state="visible")
        _set_select_by_text(page, "apply", task.get("apply_name", "女装"))
        _set_select_by_text(page, "category", category_name)
        _set_select_by_text(page, "repeat", "四方连续")
        _set_select_by_text(page, "hassource", "我有PSD", required=False)
        _set_tags(page, tags)
        page.click("button.savebutton")
        page.wait_for_timeout(2500)
        # 宽限重试
        if "已保存" not in page.content():
            page.click("button.savebutton")
            page.wait_for_timeout(3000)
        logger.info(f"  {title}: 管理表单已保存")

        # ---------- 第 3 步：销售定价 ----------
        # tab 切换靠页面 JS 函数 switchview('cards')，点 <li> 无效必须调函数
        nav_ok = page.evaluate("""() => {
            if (typeof switchview === 'function') { switchview('cards'); return true; }
            return false;
        }""")
        if not nav_ok:
            return {"ok": False, "step": "销售-导航", "detail": "未找到销售 tab"}
        page.wait_for_timeout(2000)

        # 选中第一张卡片（销售页卡片 class 是 imggrid_<id>，不是纯 imggrid）
        # 注意：卡片点击是 toggle——已选中时再点会取消选中！先查 txtcardids
        SEL_CARD_JS = """() => {
            const v = document.querySelector('.txtcardids')?.value || '';
            if (v) return 'already';
            const card = [...document.querySelectorAll('div')].find(e =>
                /(^|\\s)imggrid_\\d+/.test(e.className) && !e.className.includes('imggrid_holder'));
            if (card) { card.click(); return 'clicked'; }
            return 'none';
        }"""
        picked = page.evaluate(SEL_CARD_JS)
        if picked == "none":
            return {"ok": False, "step": "销售-选卡", "detail": "未找到销售页卡片 (imggrid_<id>)"}
        page.wait_for_timeout(1000)
        card_id = page.evaluate(
            """() => document.querySelector(".txtcardids")?.value || "" """
        )
        if not card_id:
            return {"ok": False, "step": "销售-选卡", "detail": "txtcardids 为空，卡片未选中"}

        # 销售类型 = 买断+下载
        page.check('input[name="pricetype"][value="minedownload"]')
        page.wait_for_timeout(500)

        # 价格档位（选 text 含买断价的 option；找不到就用表单默认）
        page.evaluate(
            """(buyout) => {
                const sel = document.querySelector(".pricevalue");
                if (!sel) return;
                const opt = [...sel.options].find(o => o.value.includes(buyout) || o.text.includes(buyout));
                if (opt) { sel.value = opt.value; sel.dispatchEvent(new Event("change", {bubbles: true})); }
            }""",
            buyout,
        )
        page.wait_for_timeout(500)

        # 设价提交
        page.click(".btnsetprice")
        logger.info(f"  {title}: 定价已提交，等待刷新...")
        page.wait_for_timeout(6000)

        # 刷新后：重切销售 tab（switchview）并确认选中（toggle 语义：有值不点）
        page.wait_for_timeout(3000)
        page.evaluate("""() => {
            if (typeof switchview === 'function') switchview('cards');
        }""")
        page.wait_for_timeout(2000)
        page.evaluate(
            """() => {
                const v = document.querySelector(".txtcardids")?.value || "";
                if (v) return;
                const card = [...document.querySelectorAll('div')].find(e =>
                    /(^|\\s)imggrid_\\d+/.test(e.className) && !e.className.includes('imggrid_holder'));
                if (card) card.click();
            }"""
        )
        page.wait_for_timeout(1000)
        card_id2 = page.evaluate(
            """() => document.querySelector(".txtcardids")?.value || "" """
        )
        if not card_id2:
            return {"ok": False, "step": "发布-选卡", "detail": "刷新后 txtcardids 为空"}

        # 点公开区按钮
        pub_clicked = page.evaluate(
            """() => {
                const btns = [...document.querySelectorAll("button, a, input[type=button]")];
                const pub = btns.find(b => b.textContent.trim() === "公开区");
                if (!pub) return false;
                pub.click();
                return true;
            }"""
        )
        if not pub_clicked:
            return {"ok": False, "step": "发布-按钮", "detail": "未找到公开区按钮"}
        page.wait_for_timeout(4000)

        # 捕获确认文本
        msg = page.evaluate(
            """() => {
                const m = document.body.innerText.match(/(\\d+)\\s*进入公开区\\(待审核\\)/);
                return m ? m[0] : null;
            }"""
        )
        if msg:
            return {"ok": True, "step": "完成", "detail": msg}
        return {"ok": True, "step": "完成", "detail": "已提交公开区（未捕获确认文本，请后台核对）"}

    except Exception as e:
        return {"ok": False, "step": "异常", "detail": str(e)}
    finally:
        try:
            browser.close()
            p.stop()
        except Exception:
            pass
