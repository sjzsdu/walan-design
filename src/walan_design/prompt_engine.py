"""
Prompt 工程引擎 — 把 Brief 升级为专业纺织花型生图 prompt。

核心思路：
1. SD 原始 prompt 只描述主题，缺少纺织行业标准术语与构图约束
2. 这里统一追加：质量增强词 + 构图约束 + 印刷媒介提示 + 负向词强化
3. 所有增强词可在 config.yaml design.prompt 下自定义
"""

import logging

logger = logging.getLogger(__name__)

# 内置默认（config 缺省时兜底）
DEFAULT_QUALITY_BOOSTERS = (
    "professional textile surface pattern design, print-ready artwork, "
    "high detail, crisp clean edges, rich color depth, award-winning surface design"
)
DEFAULT_COMPOSITION = (
    "balanced all-over repeat layout, evenly distributed motifs, "
    "harmonious negative space, flat illustration style, no cropped motifs, "
    "largest motifs kept away from all four image edges, "
    "plain background margin along the borders, nothing touching the image boundary"
)
DEFAULT_MEDIUM = "flat textile print for fabric industry, matte finish, no perspective, no shadows"

# 数码花型专用：强调无套色限制的渐变与丰富色域（2026-10-11，用户反馈数码花型看着像传统花型）
DEFAULT_DIGITAL_MEDIUM = (
    "digital textile print, unlimited color palette, rich smooth color gradients, "
    "watercolor color blends, layered vibrant tones, painterly depth"
)
# 数码花型的构图约束：去掉 flat illustration style（扁平风会压掉渐变表现力）
DEFAULT_DIGITAL_COMPOSITION = (
    "balanced all-over repeat layout, evenly distributed motifs, "
    "harmonious negative space, no cropped motifs, "
    "largest motifs kept away from all four image edges, "
    "plain background margin along the borders, nothing touching the image boundary"
)


def _cfg(config: dict) -> dict:
    return (config or {}).get("design", {}).get("prompt", {})


def build_quality_boosters(config: dict = None) -> str:
    return _cfg(config).get("quality_boosters", DEFAULT_QUALITY_BOOSTERS)


def build_composition_constraints(config: dict = None) -> str:
    return _cfg(config).get("composition_constraints", DEFAULT_COMPOSITION)


def build_medium_hints(config: dict = None) -> str:
    return _cfg(config).get("medium_hints", DEFAULT_MEDIUM)


def build_negative_prompt(config: dict = None) -> str:
    """
    专业负向词：排除文字/水印/人像/3D 感/边框等花型审核常见毙点。
    config.design.common.negative_prompt 存在时以其为基础追加补强项。
    """
    common = (config or {}).get("design", {}).get("common", {})
    base = common.get("negative_prompt", "")
    extra = _cfg(config).get(
        "negative_extra",
        "photorealistic photo, 3d render, human, face, hands, signature, frame, border, "
        "uneven edges, distorted motifs, cluttered, oversaturated, gradient noise",
    )
    parts = [p.strip() for p in (base + ", " + extra).split(",") if p.strip()]
    # 去重保序
    seen, merged = set(), []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            merged.append(p)
    return ", ".join(merged)


# 数码花型 sd_prompt 禁用词（传统平涂/分色词汇）。
# 2026-10-11 事故：brief LLM 给数码花型写出 "flat vector style with fine hand-drawn
# line hatching"，SDXL 对 prompt 前段权重高 → 产出传统花型长相。LLM 指令不可全信，
# 这里按逗号分段整段剔除，做确定性兜底。
_DIGITAL_FORBIDDEN = (
    "flat vector", "vector style", "flat illustration", "flat colors",
    "line hatching", "hatching", "crosshatching", "hand-drawn line",
    "silkscreen", "screen print", "block print", "woodblock",
    "limited palette", "gouache",
)
# 数码质感锚点词：净化后一个都没有则强制补上
_DIGITAL_ANCHORS = (
    "watercolor", "gradient", "painterly", "digital painting", "digital print",
    "ink bleed", "color blend", "smooth color transition", "wash",
)


def _sanitize_digital_core(core: str) -> str:
    """数码花型 sd_prompt 净化：剔除含禁用词的整段，确保至少一个数码质感锚点。"""
    kept, removed = [], []
    for seg in core.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if any(bad in seg.lower() for bad in _DIGITAL_FORBIDDEN):
            removed.append(seg)
        else:
            kept.append(seg)
    out = ", ".join(kept)
    if not any(a in out.lower() for a in _DIGITAL_ANCHORS):
        out = (out + ", watercolor wash, smooth color gradients").strip(", ")
    if removed:
        logger.info(f"  数码 sd_prompt 净化，剔除: {'; '.join(removed)}")
    return out


def is_digital_brief(brief: dict, config: dict = None) -> bool:
    """该 brief 是否按数码花型出图（brief.floral_type 优先，config brief_type 兜底）"""
    floral = (brief.get("floral_type") or "").strip().lower()
    if floral in ("digital", "traditional"):
        return floral == "digital"
    return (config or {}).get("design", {}).get("brief_type", "mixed") == "digital"


def build_sd_prompt(brief: dict, config: dict = None) -> str:
    """
    Brief.sd_prompt → 专业级 prompt。
    结构：主体（LLM 生成）+ 构图约束 + 媒介提示 + 质量增强词。
    数码花型自动切换数码媒介词（渐变/无套色限制），传统花型保持扁平印花风。
    """
    core = (brief.get("sd_prompt") or "").strip().rstrip(",")
    if not core:
        core = brief.get("theme", "seamless floral pattern")
        logger.debug("brief 缺少 sd_prompt，退化为 theme 描述")

    if is_digital_brief(brief, config):
        core = _sanitize_digital_core(core)
        comp = _cfg(config).get("digital_composition_constraints", DEFAULT_DIGITAL_COMPOSITION)
        medium = _cfg(config).get("digital_medium_hints", DEFAULT_DIGITAL_MEDIUM)
    else:
        comp = build_composition_constraints(config)
        medium = build_medium_hints(config)

    sections = [core]
    if comp:
        sections.append(comp)
    if medium:
        sections.append(medium)
    if boosters := build_quality_boosters(config):
        sections.append(boosters)

    prompt = ", ".join(sections)
    logger.debug(f"prompt_engine 输出: {prompt[:120]}...")
    return prompt
