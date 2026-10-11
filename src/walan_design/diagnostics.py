"""
故障诊断 — 把底层异常翻译成「发生了什么 / 为什么 / 怎么办」

背景：2026-10-11 事故复盘。Stability 余额耗尽（429）后旧逻辑静默用占位图
跑完 5 个设计 25 个 PSD 全是废品，用户在日志里根本看不出哪里坏了。

设计原则（用户是非技术人员，任何故障必须回答三个问题）：
  1. 发生了什么（现象）
  2. 为什么（根因，尽量带证据：HTTP 状态码、余额数字）
  3. 怎么办（可直接复制执行的命令 / URL）

两类故障的处理策略：
  - 硬故障（余额耗尽 / key 失效 / 网络不通）：重试无意义，立即中止流水线，
    CLI 渲染成红色面板（PipelineBlocked）
  - 软故障（单设计超时 / 视觉自评挂了 / 上传登录态过期）：跳过继续跑，
    记录到 output/temp/attention.json，运行结束时 CLI 渲染黄色汇总面板
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineBlocked(RuntimeError):
    """流水线硬故障 — CLI 捕获后渲染成「发生了什么/为什么/怎么办」面板"""

    def __init__(self, title: str, cause: str, solutions: list[str]):
        self.title = title
        self.cause = cause
        self.solutions = solutions
        super().__init__(
            f"{title}\n原因：{cause}\n解决方法：\n" + "\n".join(f"  {i}. {s}" for i, s in enumerate(solutions, 1))
        )


# 错误文本模式 → (原因, [解决方法])
# 覆盖：生图供应商（Stability/本地SD/DALL-E）、LLM（OpenRouter）、瓦栏登录态
_ERROR_PATTERNS: list[tuple[str, str, list[str]]] = [
    (
        r"(?i)(insufficient|not\s+enough|out\s+of).{0,30}(credit|balance)"
        r"|(credit|balance).{0,30}(exhausted|insufficient|too\s+low)"
        r"|余额(不足|耗尽|只剩)",
        "AI 生图账户余额不足，无法继续生成",
        [
            "打开 Stability 充值页: https://platform.stability.ai/account/billing",
            "充值完成后重跑: walan-design run",
        ],
    ),
    (
        r"(?i)401|unauthorized|invalid.{0,25}api.{0,10}key|api.{0,10}key.{0,20}(invalid|expired)|key 未设置",
        "AI 服务 API key 无效或未设置",
        [
            "检查 config.yaml 里的 api_key 配置项，或对应环境变量（STABILITY_API_KEY / OPENROUTER_API_KEY）",
            "Stability key 在 https://platform.stability.ai/account/keys 管理",
        ],
    ),
    (
        r"(?i)connection\s*(error|refused|reset|aborted)|无法连接|连接失败|max\s+retries\s+exceeded|name\s+or\s+service\s+not\s+known",
        "AI 服务连接失败（网络问题，或本地服务未启动）",
        [
            "检查网络/代理是否正常",
            "如用的是本地 SD WebUI，先启动服务再重跑",
            "网络恢复后重跑: walan-design run",
        ],
    ),
    (
        r"(?i)403|forbidden|location\s+not\s+supported",
        "AI 服务拒绝了请求（地区限制或账号权限不足）",
        [
            "部分服务（如 OpenRouter 免费 Gemini）不对中国大陆开放，换用其他模型/供应商",
            "改 config.yaml 对应 provider 的 model 或 engine 配置",
        ],
    ),
    (
        r"登录态|storage_state|重新登录|登录已?失效|登录态文件不存在",
        "瓦栏登录态问题",
        [
            "运行: walan-design login（浏览器里手动登录一次即可）",
            "登录态约 6-7 小时过期，跑流水线前可先 walan-design status 检查",
        ],
    ),
]


def diagnose_error(err_text: str) -> tuple[str, list[str]] | None:
    """按错误文本匹配已知模式，返回 (原因, 解决方法)；未知错误返回 None"""
    for pat, cause, solutions in _ERROR_PATTERNS:
        if re.search(pat, err_text or ""):
            return cause, solutions
    return None


# 硬故障模式：命中即中止整批（重试每个设计都会同样失败）
_HARD_FAILURE_PAT = re.compile(
    r"(?i)(insufficient|not\s+enough|out\s+of).{0,30}(credit|balance)"
    r"|(credit|balance).{0,30}(exhausted|insufficient|too\s+low)"
    r"|余额(不足|耗尽|只剩)"
    r"|401|unauthorized|key 未设置|api.{0,10}key.{0,20}(invalid|expired)"
    r"|connection\s*(error|refused|reset)|无法连接|连接失败|max\s+retries\s+exceeded"
)


def is_hard_failure(err_text: str) -> bool:
    """硬故障判断：余额耗尽 / key 失效 / 网络断 — 中止而非逐设计跳过"""
    return bool(_HARD_FAILURE_PAT.search(err_text or ""))


def provider_blocked(engine: str, err_text: str) -> PipelineBlocked:
    """生图供应商不可用 → 构造带诊断的阻断异常"""
    diagnosed = diagnose_error(err_text)
    if diagnosed:
        cause, solutions = diagnosed
        if err_text:
            cause = f"{cause}（{err_text[:150]}）"
    else:
        cause = f"生图供应商 {engine} 不可用: {(err_text or '')[:150]}"
        solutions = [
            f"检查 config.yaml 中 design.{engine} 相关配置",
            "或切换供应商: walan-design config set design.engine <供应商名>",
        ]
    return PipelineBlocked("AI 生图无法继续，流水线中止", cause, solutions)


# ---------- 中断提示：checkpoint 进度 ----------

_STEP_LABELS = [
    ("collect", "趋势采集"),
    ("design", "AI 生图"),
    ("psd", "PSD 生成"),
    ("check", "质量检测"),
    ("upload", "瓦栏上传"),
]

RESUME_HINT = "处理完成后执行: walan-design run（自动从断点继续，已生成的花型自动跳过、不重复扣费）"


def checkpoint_progress(config: dict) -> str:
    """读 checkpoint.json 生成进度摘要（中断面板用）；无 checkpoint 返回空串"""
    path = Path(config.get("pipeline", {}).get("temp_dir", "output/temp")) / "checkpoint.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    steps = data.get("steps", {})
    parts = []
    for key, label in _STEP_LABELS:
        st = steps.get(key, {}).get("status", "pending")
        icon = {"done": "✅", "failed": "❌"}.get(st, "⏳")
        parts.append(f"{label} {icon}")
    return "  |  ".join(parts)


# ---------- 软故障提醒（attention.json）----------


def _attention_path(config: dict) -> Path:
    temp_dir = Path(config.get("pipeline", {}).get("temp_dir", "output/temp"))
    return temp_dir / "attention.json"


def record_attention(config: dict, stage: str, message: str, solutions: list[str] | None = None):
    """
    记录一条软故障提醒（结束运行时 CLI 汇总渲染）。
    未显式给 solutions 时自动按错误文本诊断；同一 (stage, message) 去重。
    """
    if solutions is None:
        diagnosed = diagnose_error(message)
        solutions = diagnosed[1] if diagnosed else ["处理后重跑: walan-design run"]

    path = _attention_path(config)
    notes = []
    try:
        if path.exists():
            notes = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        notes = []

    if any(n.get("stage") == stage and n.get("message") == message for n in notes):
        return

    notes.append({"stage": stage, "message": message[:300], "solutions": solutions})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.warning(f"⚠ 已记录待处理问题 [{stage}]: {message[:120]}")


def take_attention(config: dict) -> list[dict]:
    """取出全部提醒并清空文件（一次性消费，避免下次误报）"""
    path = _attention_path(config)
    if not path.exists():
        return []
    try:
        notes = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        notes = []
    path.unlink(missing_ok=True)
    return notes
