"""
CLI 入口 — 使用 Typer 构建的现代命令行界面
子命令: run / collect / design / psd / upload / config
"""

import json
import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from walan_design.pipeline import (
    load_config,
    run_collect,
    run_design,
    run_pipeline,
    setup_logging,
)

app = typer.Typer(
    name="walan-design",
    help="瓦栏 AI 花型设计自动售卖流水线 — 从趋势采集到瓦栏上传全自动化",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# =====================================================
# 辅助函数
# =====================================================


def _load_config() -> dict:
    return load_config()


def _setup_logging(level: str):
    setup_logging(level)


def _print_config_value(key_path: str, config: dict):
    """用点分隔的路径访问 config 值，如 'pipeline.batch_count'"""
    parts = key_path.split(".")
    val = config
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            console.print(f"[red]未找到配置项: {key_path}[/red]")
            return
    console.print(f"[dim]{key_path}[/dim] = [bold]{val}[/bold]")


def _set_config_value(key_path: str, new_value: str):
    """用点分隔的路径修改 config.yaml 中的值"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parts = key_path.split(".")
    # 找到倒数第二层
    target = config
    for p in parts[:-1]:
        if p not in target:
            target[p] = {}
        target = target[p]

    old_value = target.get(parts[-1])

    # 尝试智能类型转换
    val = new_value
    if old_value is not None:
        if isinstance(old_value, bool):
            val = new_value.lower() in ("true", "1", "yes")
        elif isinstance(old_value, int):
            try:
                val = int(new_value)
            except ValueError:
                pass
        elif isinstance(old_value, float):
            try:
                val = float(new_value)
            except ValueError:
                pass
        elif isinstance(old_value, list):
            val = [v.strip() for v in new_value.split(",")]

    target[parts[-1]] = val

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] 已更新 config.yaml: [dim]{key_path}[/dim] = [bold]{val}[/bold]")


# =====================================================
# login — 瓦栏登录态保存
# =====================================================


@app.command()
def login(
    headless: bool = typer.Option(False, "--headless", "-h", help="headless 模式（默认 headed 方便手动登录）"),
):
    """打开 headed 浏览器登录瓦栏，保存 storage_state 供后续自动上传复用"""
    from walan_design.upload_executor import save_login_state

    save_login_state(headless=headless)


# =====================================================
# run — 完整流水线
# =====================================================


@app.command()
def run(
    count: int = typer.Option(None, "--count", "-n", help="生成花型数量"),
    mode: str = typer.Option("full", "--mode", "-m", help="运行模式: full / collect_only / design_only / upload_only"),
    log_level: str = typer.Option(None, "--log", "-l", help="日志级别: DEBUG / INFO / WARNING / ERROR"),
):
    """
    运行完整流水线（默认）。也可用 --mode 只跑某一段。

    示例:
      walan-design run                  # 完整流程
      walan-design run -n 3            # 生成 3 个花型
      walan-design run -m collect_only  # 只做趋势采集
      walan-design run -m design_only   # 只做 AI 生图
    """
    config = _load_config()
    _setup_logging(log_level or config["pipeline"]["log_level"])

    if count:
        config["pipeline"]["batch_count"] = count
        config["trend"]["brief_count"] = count

    config["pipeline"]["mode"] = mode

    console.print(
        Panel.fit(
            f"[bold cyan]walan-design v0.1.0[/bold cyan]\n"
            f"模式: [bold]{mode}[/bold]  |  批量: [bold]{config['trend']['brief_count']}[/bold]",
            border_style="green",
        )
    )

    try:
        result = run_pipeline(config, mode=mode)
        console.print("\n[green bold]✓ 完成[/green bold]")
        return result
    except KeyboardInterrupt:
        console.print("\n[yellow]已中断[/yellow]")
        sys.exit(130)


# =====================================================
# collect — 趋势采集
# =====================================================


@app.command()
def collect(
    count: int = typer.Option(None, "--count", "-n", help="生成 Brief 数量"),
):
    """采集流行趋势，生成设计 Brief（保存到 output/trends/）"""
    config = _load_config()
    _setup_logging("INFO")

    if count:
        config["trend"]["brief_count"] = count

    console.print(f"采集趋势 → 生成 {config['trend']['brief_count']} 个设计 Brief...")
    briefs = run_collect(config)

    # 漂亮地展示结果
    table = Table(title="设计 Brief 预览", show_lines=True)
    table.add_column("#", style="dim", justify="center")
    table.add_column("标题", style="bold cyan")
    table.add_column("主题")
    table.add_column("标签", style="yellow")
    for i, b in enumerate(briefs, 1):
        table.add_row(
            str(i),
            b.get("title", ""),
            b.get("theme", ""),
            ", ".join(b.get("tags", [])),
        )
    console.print(table)
    console.print(f"\n[green]✓[/green] 共 {len(briefs)} 个 Brief 已保存")


# =====================================================
# design — AI 生图
# =====================================================


@app.command()
def design(
    input_file: str = typer.Argument(None, help="Brief JSON 文件路径（留空用本地模板）"),
):
    """
    根据 Brief 用 AI 生成花型图片。

    示例:
      walan-design design                                 # 用本地模板 Brief
      walan-design design output/trends/briefs_xxx.json   # 用之前采集的 Brief
    """
    from walan_design.trend_collector import _local_fallback_briefs

    config = _load_config()
    _setup_logging("INFO")

    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        briefs = data.get("briefs", data)
    else:
        briefs = _local_fallback_briefs(config["trend"]["brief_count"], config["design"]["style_preferences"])

    console.print(f"AI 生图: {len(briefs)} 个设计")
    results = run_design(briefs, config)

    # 漂亮展示
    table = Table(title="生成结果")
    table.add_column("设计")
    table.add_column("图片数")
    table.add_column("保存路径")
    for r in results:
        paths = r["image_paths"]
        table.add_row(
            r["brief"].get("title", ""),
            str(len(paths)),
            str(Path(paths[0]).parent) if paths else "-",
        )
    console.print(table)
    console.print(f"\n[green]✓[/green] 共生成 {sum(len(r['image_paths']) for r in results)} 张图片")


# =====================================================
# psd — PSD 文件生成
# =====================================================


@app.command()
def psd(
    input_dir: str = typer.Argument(None, help="PNG 图片目录（留空则重新生成设计）"),
):
    """将 AI 生成的 PNG 转换为符合瓦栏规范的 PSD 文件"""
    from walan_design.psd_generator import run as psd_run

    config = _load_config()
    _setup_logging("INFO")

    if input_dir:
        # 用外部目录做一个临时 result 结构
        pngs = sorted(Path(input_dir).glob("*.png"))
        results = [
            {
                "brief": {"title": p.stem},
                "image_paths": [str(p)],
            }
            for p in pngs
        ]
    else:
        # 需要先有 design 结果，否则用本地模板生成占位
        from walan_design.ai_designer import run as design_run
        from walan_design.trend_collector import _local_fallback_briefs

        briefs = _local_fallback_briefs(2, config["design"]["style_preferences"])
        results = design_run(briefs, config)

    console.print(f"PNG → PSD: {len(results)} 个设计")
    psd_run(results, config)

    total = sum(len(r.get("psd_paths", [])) for r in results)
    console.print(f"[green]✓[/green] 共生成 {total} 个 PSD")


# =====================================================
# upload — 上传准备
# =====================================================


@app.command()
def upload(
    input_dir: str = typer.Argument(None, help="PSD 目录（留空用 output/runs/ 下已有花型）"),
):
    """准备瓦栏上传任务（生成 browser_use prompt，实际浏览器操作由 ego-lite 执行）"""
    from walan_design.walan_uploader import run as upload_run

    config = _load_config()
    _setup_logging("INFO")

    if input_dir:
        psds = sorted(Path(input_dir).glob("*.psd"))
        results = [
            {
                "brief": {"title": p.stem, "tags": []},
                "psd_paths": [str(p)],
            }
            for p in psds
        ]
    else:
        # 从集中目录 output/runs/{标题}/ 读取已生成的花型（每个取 _1.psd 主图）
        runs_dir = Path(config.get("pipeline", {}).get("runs_dir", "output/runs"))
        results = []
        for run_dir in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
            if not run_dir.is_dir():
                continue
            main_psd = run_dir / f"{run_dir.name}_1.psd"
            if main_psd.exists():
                # brief 信息从花型目录里的 brief.json 恢复
                brief = {}
                brief_file = run_dir / "brief.json"
                if brief_file.exists():
                    brief = json.loads(brief_file.read_text(encoding="utf-8"))
                results.append(
                    {
                        "brief": brief,
                        "run_dir": str(run_dir),
                        "psd_paths": [str(main_psd)],
                    }
                )
        if not results:
            console.print("[yellow]output/runs/ 下没有已生成的花型，请先运行 walan-design run[/yellow]")
            return

    console.print("准备瓦栏上传任务...")
    info = upload_run(results, config)

    if info["status"] == "ready":
        console.print(
            Panel.fit(
                f"[bold]✅ 上传任务已准备好[/bold]\n\n"
                f"任务数: [bold]{info['task_count']}[/bold]\n"
                f"任务文件: [dim]{info['task_file']}[/dim]\n"
                f"Prompt 文件: [dim]{info['prompt_file']}[/dim]\n\n"
                f"[yellow]下一步[/yellow]: 用 Trae 打开 prompt 文件，"
                f"让 browser_use agent 执行浏览器上传",
                border_style="green",
            )
        )
    else:
        console.print("[red]没有可上传的任务[/red]")


# =====================================================
# config — 配置管理
# =====================================================

config_app = typer.Typer(help="配置管理（查看/修改 config.yaml）")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(key: str = typer.Argument(None, help="配置键路径，如 pipeline.batch_count（留空显示全部）")):
    """查看配置"""
    config = _load_config()
    if key:
        _print_config_value(key, config)
    else:
        # 漂亮地打印整个 config
        table = Table(title="config.yaml")
        table.add_column("section", style="cyan")
        table.add_column("key")
        table.add_column("value")
        for section, val in config.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    table.add_row(section, k, str(v))
            else:
                table.add_row(section, "-", str(val))
        console.print(table)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="配置键路径，如 walan.price_tier"),
    value: str = typer.Argument(..., help="新值"),
):
    """修改配置（自动保存到 config.yaml）"""
    _set_config_value(key, value)


# =====================================================
# __main__
# =====================================================

if __name__ == "__main__":
    app()
