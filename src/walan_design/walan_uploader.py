"""
瓦栏自动上传模块
使用 ego-lite（Trae 内置 browser_use agent）完成浏览器操作，免去登录。

工作方式：
  1. Python 侧准备好上传数据（PSD 路径、价格、标签等），生成 upload_task.json
  2. 调用 Trae 的 browser_use agent 执行浏览器上传操作
  3. agent 使用用户已登录的浏览器 profile，无需重新登录

优势：
  - 不需要 Playwright / Selenium，不装额外浏览器
  - 直接复用用户已有的登录态
  - 上传逻辑由 AI agent 自适应完成，不依赖页面元素选择器
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_price_config(config: dict) -> dict:
    """根据 price_tier 构建价格参数"""
    walan_cfg = config["walan"]
    tier = walan_cfg["price_tier"]

    if tier == "tier_299":
        return {"buyout": 299, "download": 99, "psd": 50, "tier_label": "tier_299"}
    elif tier == "tier_399":
        return {"buyout": 399, "download": 199, "psd": 50, "tier_label": "tier_399"}
    elif tier == "custom":
        cp = walan_cfg["custom_price"]
        return {"buyout": cp["buyout"], "download": cp["download"], "psd": cp["psd"], "tier_label": "custom"}
    else:
        return {"buyout": 299, "download": 99, "psd": 50, "tier_label": "tier_299"}


def prepare_upload_tasks(design_results: list, config: dict) -> list:
    """
    将设计结果转换为上传任务列表。
    每个任务包含：PSD 文件路径、标题、价格、标签、发布区域。
    """
    walan_cfg = config["walan"]
    price_cfg = _build_price_config(config)
    zone = walan_cfg["auto_publish_zone"]

    tasks = []
    for i, result in enumerate(design_results):
        brief = result.get("brief", {})
        title = brief.get("title", f"design_{i + 1}")
        psd_files = result.get("psd_paths", [])

        if not psd_files:
            logger.warning(f"跳过 {title}：无 PSD 文件")
            continue

        # 每个设计选第一个 PSD 上传
        psd_path = psd_files[0]
        if not Path(psd_path).exists():
            logger.warning(f"跳过 {title}：PSD 文件不存在 {psd_path}")
            continue

        task = {
            "title": title,
            "psd_path": str(Path(psd_path).resolve()),
            "theme": brief.get("theme", ""),
            "tags": brief.get("tags", []),
            "sell_type": walan_cfg["sell_type"],
            "price": price_cfg,
            "zone": zone,
            "upload_url": walan_cfg["upload_url"],
        }
        tasks.append(task)
        logger.info(f"  准备上传任务: {title} → {psd_path}")

    return tasks


def build_browser_prompt(task: dict) -> str:
    """
    为 browser_use agent 构建上传指令。
    agent 会打开浏览器、导航到上传页、执行上传操作。
    因为使用用户已登录的浏览器，所以不需要登录步骤。
    """
    psd_path = task["psd_path"]
    title = task["title"]
    tags = "、".join(task.get("tags", []))
    sell_type_text = "买断+下载" if task["sell_type"] == "buyout_download" else "仅买断"
    price = task["price"]
    zone_text = "公开区" if task["zone"] == "public" else "VIP区"
    upload_url = task["upload_url"]

    prompt = f"""请帮我完成瓦栏花型上传操作：

1. 打开页面：{upload_url}
2. 点击"上传设计"按钮
3. 在文件选择对话框中选择文件：{psd_path}
4. 等待文件上传完成
5. 填写花型名称：{title}
6. 选择售卖类型：{sell_type_text}
7. 选择价格档位：买断{price["buyout"]}元，下载{price["download"]}元，PSD{price["psd"]}元
8. 添加标签：{tags}
9. 点击"上传"或"提交"按钮
10. 上传成功后，将花型发布到{zone_text}

请逐步完成每一步操作，每步等待页面加载完成后再继续下一步。
"""
    return prompt


def save_upload_tasks(tasks: list, config: dict) -> str:
    """保存上传任务到 JSON 文件，供 browser_use agent 读取"""
    output_dir = Path(config["psd"]["output_dir"]).parent / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_file = output_dir / f"upload_tasks_{timestamp}.json"

    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "task_count": len(tasks),
                "tasks": tasks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"上传任务已保存: {task_file}")
    return str(task_file)


def run(design_results: list, config: dict = None) -> dict:
    """
    准备上传任务并调用 browser_use agent 执行上传。

    返回:
      {
        "task_file": str,        # 上传任务文件路径
        "task_count": int,       # 任务数量
        "tasks": list,           # 任务列表
        "prompts": list,         # 每个任务对应的 browser prompt
        "status": "ready" | "no_tasks",
      }
    """
    if config is None:
        config = load_config()

    logger.info(f"\n=== 瓦栏上传准备: 共 {len(design_results)} 个设计 ===")

    # 1. 准备上传任务
    tasks = prepare_upload_tasks(design_results, config)

    if not tasks:
        logger.warning("没有可上传的任务")
        return {"status": "no_tasks", "task_count": 0}

    # 2. 为每个任务生成 browser prompt
    prompts = []
    for task in tasks:
        prompt = build_browser_prompt(task)
        prompts.append(prompt)
        logger.info(f"\n--- 上传任务: {task['title']} ---")
        logger.info(f"  文件: {task['psd_path']}")
        logger.info(f"  价格: 买断{task['price']['buyout']} 下载{task['price']['download']}")
        logger.info(f"  标签: {', '.join(task.get('tags', []))}")
        logger.info(f"  发布区: {task['zone']}")

    # 3. 保存任务文件
    task_file = save_upload_tasks(tasks, config)

    # 4. 保存 prompts 供 pipeline 调用
    output_dir = Path(config["psd"]["output_dir"]).parent / "temp"
    prompt_file = output_dir / "upload_prompts.json"
    with open(prompt_file, "w", encoding="utf-8") as f:
        json.dump([{"title": t["title"], "prompt": p} for t, p in zip(tasks, prompts)], f, ensure_ascii=False, indent=2)

    logger.info("\n上传准备完成:")
    logger.info(f"  任务数: {len(tasks)}")
    logger.info(f"  任务文件: {task_file}")
    logger.info(f"  Prompt 文件: {prompt_file}")
    logger.info("  状态: ready（等待 browser_use agent 执行）")

    return {
        "status": "ready",
        "task_file": task_file,
        "prompt_file": str(prompt_file),
        "task_count": len(tasks),
        "tasks": tasks,
        "prompts": prompts,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    config = load_config()

    # 测试：生成占位设计结果并准备上传
    from walan_design.ai_designer import run as design_run
    from walan_design.psd_generator import run as psd_run
    from walan_design.trend_collector import _local_fallback_briefs

    briefs = _local_fallback_briefs(1, config["design"]["style_preferences"])
    results = design_run(briefs, config)
    psd_run(results, config)

    upload_info = run(results, config)
    if upload_info["status"] == "ready":
        print(f"\n准备好 {upload_info['task_count']} 个上传任务")
        print(f"任务文件: {upload_info['task_file']}")
        print("\n第一个任务的 browser prompt:")
        print(upload_info["prompts"][0])
