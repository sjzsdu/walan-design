"""
主流程编排
串联：趋势采集 → AI 生图 → 接回位 → PSD 生成 → 质量检测 → 瓦栏上传

运行方式:
  python -m src.pipeline                    # 完整流水线
  python -m src.pipeline --mode collect_only
  python -m src.pipeline --mode design_only
  python -m src.pipeline --mode upload_only
  python -m src.pipeline --count 3          # 只生成3个
"""

import argparse
import logging
import time
from pathlib import Path

import yaml

logger = logging.getLogger("pipeline")


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def step_header(title: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def run_collect(config: dict) -> list:
    step_header("① 趋势采集 & 设计 Brief 生成")
    from walan_design.trend_collector import run as collect_run

    briefs = collect_run(config)
    logger.info(f"完成，共生成 {len(briefs)} 个 Brief")
    return briefs


def run_design(briefs: list, config: dict) -> list:
    step_header("② AI 花型设计 & 四方连续处理")
    from walan_design.ai_designer import run as design_run

    results = design_run(briefs, config)
    logger.info(f"完成，共生成 {len(results)} 组设计")
    return results


def run_psd(design_results: list, config: dict) -> list:
    step_header("③ PSD 文件生成")
    from walan_design.psd_generator import run as psd_run

    psd_run(design_results, config)
    count = sum(len(r.get("psd_paths", [])) for r in design_results)
    logger.info(f"完成，共生成 {count} 个 PSD 文件")
    return design_results


def run_check(design_results: list, config: dict) -> list:
    step_header("④ 质量检测")
    from walan_design.quality_checker import run as check_run

    passed = check_run(design_results, config)
    logger.info(f"完成，{len(passed)}/{len(design_results)} 通过")
    return passed


def run_upload(design_results: list, config: dict) -> dict:
    step_header("⑤ 瓦栏上传准备 (ego-lite)")
    from walan_design.walan_uploader import run as upload_run

    upload_info = upload_run(design_results, config)
    if upload_info["status"] == "ready":
        logger.info(f"完成，{upload_info['task_count']} 个上传任务已准备就绪")
        logger.info(f"任务文件: {upload_info['task_file']}")
        logger.info(f"Prompt 文件: {upload_info['prompt_file']}")
        logger.info("下一步：调用 browser_use agent 执行浏览器上传")
    else:
        logger.warning("没有可上传的任务")
    return upload_info


def run_pipeline(config: dict, mode: str = "full") -> dict:
    """
    按指定模式运行流水线。
    mode: full / collect_only / design_only / upload_only
    """
    logger.info(f"流水线启动 | 模式={mode}")
    start = time.time()

    output = {}

    # 尝试加载之前的中间产物（支持 design_only / upload_only 模式）
    briefs = None
    design_results = None

    if mode in ("full", "collect_only", "design_only"):
        briefs = run_collect(config)
        output["briefs"] = briefs

    if mode == "collect_only":
        logger.info("collect_only 模式，流水线结束")
        return output

    if mode in ("full", "design_only"):
        if briefs is None:
            # 用本地 fallback 测试用
            from walan_design.trend_collector import _local_fallback_briefs

            briefs = _local_fallback_briefs(
                config["trend"]["brief_count"],
                config["design"]["style_preferences"],
            )
        design_results = run_design(briefs, config)
        output["design_results"] = design_results

    if mode == "design_only":
        logger.info("design_only 模式，流水线结束")
        return output

    if mode in ("full", "upload_only"):
        if design_results is None:
            logger.error("upload_only 模式需要先运行 design_only 生成设计结果")
            return output

        run_psd(design_results, config)
        # 质量门必须真正过滤：只把通过检测的设计交给上传步骤
        passed_results = run_check(design_results, config)
        upload_info = run_upload(passed_results, config)
        output["upload_info"] = upload_info

    elapsed = time.time() - start
    logger.info("")
    logger.info(f"✓ 流水线完成，耗时 {elapsed:.1f}s")
    return output


def main():
    parser = argparse.ArgumentParser(description="瓦栏 AI 花型设计自动售卖流水线")
    parser.add_argument("--mode", default=None, help="运行模式: full / collect_only / design_only / upload_only")
    parser.add_argument("--count", type=int, default=None, help="覆盖批量数量")
    parser.add_argument("--headless", action="store_true", help="(已废弃，ego-lite 不需要)")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config["pipeline"]["log_level"])

    # CLI 参数覆盖配置
    if args.mode:
        config["pipeline"]["mode"] = args.mode
    if args.count:
        config["pipeline"]["batch_count"] = args.count
        config["trend"]["brief_count"] = args.count

    mode = config["pipeline"]["mode"]

    logger.info("=" * 60)
    logger.info("  瓦栏 AI 花型设计自动售卖流水线")
    logger.info("=" * 60)
    logger.info(f"  模式: {mode}")
    logger.info(f"  批量: {config['pipeline']['batch_count']}")
    logger.info("  配置: config.yaml")

    run_pipeline(config, mode=mode)


if __name__ == "__main__":
    main()
