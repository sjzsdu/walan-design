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
import json
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

    # 断点续跑（仅 full 模式启用）
    ckpt = None
    if mode == "full" and config.get("pipeline", {}).get("resume", True):
        from walan_design.checkpoint import CheckpointManager

        ckpt = CheckpointManager(config)
        logger.info(ckpt.summary())

    def _skip_if_done(step: str) -> bool:
        """如果 checkpoint 显示 step 已完成，跳过并返回 True"""
        if ckpt and ckpt.is_step_done(step):
            logger.info(f"  ↳ {step} 已完成，跳过")
            return True
        return False

    output = {}

    # 尝试加载之前的中间产物（支持 design_only / upload_only 模式）
    briefs = None
    design_results = None

    # 中间产物落盘：checkpoint 只记"做没做完"，产物本身存 JSON，
    # 中断恢复时据此还原（否则 design 标记 done 但内存无结果，后续全卡死）
    temp_dir = Path(config["pipeline"].get("temp_dir", "output/temp"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    def _save_step(name: str, data):
        (temp_dir / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_step(name: str):
        p = temp_dir / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"  {name} 读取失败，将重新生成: {e}")
            return None

    if mode in ("full", "collect_only", "design_only"):
        if not _skip_if_done("collect"):
            briefs = run_collect(config)
            _save_step("briefs.json", briefs)
            output["briefs"] = briefs
            if ckpt:
                ckpt.mark_done("collect", len(briefs), "output/trends/")
        else:
            briefs = _load_step("briefs.json")

    if mode == "collect_only":
        logger.info("collect_only 模式，流水线结束")
        return output

    if mode in ("full", "design_only"):
        if not _skip_if_done("design"):
            if briefs is None:
                # collect 标记 done 但 briefs.json 丢失 → 重新采集（纯本地操作）
                briefs = run_collect(config)
                _save_step("briefs.json", briefs)
                if ckpt:
                    ckpt.mark_done("collect", len(briefs), "output/trends/")
            design_results = run_design(briefs, config)
            _save_step("design_results.json", design_results)
            output["design_results"] = design_results
            if ckpt:
                ckpt.mark_done("design", len(design_results), "output/designs/")
        else:
            design_results = _load_step("design_results.json")

    if mode == "design_only":
        logger.info("design_only 模式，流水线结束")
        return output

    if mode in ("full", "upload_only"):
        if design_results is None:
            if mode == "upload_only":
                logger.error("upload_only 模式需要先运行 design_only 生成设计结果")
                return output
            # full 模式：checkpoint 标 design done 但产物丢失 → 重跑 design 而不是卡死
            logger.warning("  design 结果缺失（checkpoint 标记 done 但产物丢失），重新执行 design 步骤")
            if ckpt:
                ckpt.reset("design")
            if briefs is None:
                briefs = _load_step("briefs.json")
            if briefs is None:
                # collect 标 done 但 briefs.json 也丢失 → 重新采集
                briefs = run_collect(config)
                _save_step("briefs.json", briefs)
                if ckpt:
                    ckpt.mark_done("collect", len(briefs), "output/trends/")
            design_results = run_design(briefs, config)
            _save_step("design_results.json", design_results)
            output["design_results"] = design_results
            ckpt.mark_done("design", len(design_results), "output/designs/")

        if not _skip_if_done("psd"):
            run_psd(design_results, config)
            # psd 步骤会检测 floral_type 写入 result，重新落盘保持一致
            _save_step("design_results.json", design_results)
            if ckpt:
                total_psd = sum(len(r.get("psd_paths", [])) for r in design_results)
                ckpt.mark_done("psd", total_psd, "output/psd/")

        if not _skip_if_done("check"):
            # 质量门必须真正过滤：只把通过检测的设计交给上传步骤
            passed_results = run_check(design_results, config)
            if ckpt:
                ckpt.mark_done("check", len(passed_results))
        else:
            passed_results = design_results

        if not _skip_if_done("upload"):
            upload_info = run_upload(passed_results, config)
            output["upload_info"] = upload_info
            if ckpt:
                ckpt.mark_done("upload", upload_info.get("task_count", 0))

            # 上传准备完成后（浏览器 agent 执行前），把指纹写入历史库
            # 这样下一批跑 dedup 能和上一批比对
            if upload_info.get("status") == "ready":
                try:
                    from walan_design.dedup_checker import record_published
                    record_published(passed_results, config)
                except Exception as e:
                    logger.warning(f"  历史库写入失败（不阻塞）: {e}")

            # 真实上传执行（ego-browser）：auto_upload 开启时逐任务上传到瓦栏
            if (
                upload_info.get("status") == "ready"
                and config.get("pipeline", {}).get("auto_upload", False)
                and upload_info.get("tasks")
            ):
                from walan_design.upload_executor import execute_upload

                upload_results = []
                for task in upload_info["tasks"]:
                    # 上次失败过的任务：文件已在服务器，skip_upload 续跑管理+发布
                    for pr in passed_results:
                        if pr.get("brief", {}).get("title") == task["title"] and pr.get("run_dir"):
                            prev = Path(pr["run_dir"]) / "upload_result.json"
                            if prev.exists():
                                try:
                                    if not json.loads(prev.read_text(encoding="utf-8")).get("ok", False):
                                        task["skip_upload"] = True
                                        logger.info(f"  ↳ {task['title']} 上次上传失败，从已上传列表续跑")
                                except (json.JSONDecodeError, OSError):
                                    pass
                            break
                    r = execute_upload(task, config)
                    upload_results.append({"title": task["title"], **r})
                    if r.get("ok"):
                        logger.info(f"  ✅ 上传成功: {task['title']} → {r.get('detail')}")
                    else:
                        logger.error(f"  ❌ 上传失败: {task['title']} [{r.get('step')}] {r.get('detail')}")
                output["upload_results"] = upload_results
                # 结果也落盘到各花型目录
                for r, res in zip(upload_info["tasks"], upload_results):
                    for pr in passed_results:
                        if pr.get("brief", {}).get("title") == r["title"] and pr.get("run_dir"):
                            (Path(pr["run_dir"]) / "upload_result.json").write_text(
                                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
                            )
                            break
        else:
            upload_info = {}

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
