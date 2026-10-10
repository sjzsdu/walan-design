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
    "harmonious negative space, flat illustration style, no cropped motifs"
)
DEFAULT_MEDIUM = "flat textile print for fabric industry, matte finish, no perspective, no shadows"


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


def build_sd_prompt(brief: dict, config: dict = None) -> str:
    """
    Brief.sd_prompt → 专业级 prompt。
    结构：主体（LLM 生成）+ 构图约束 + 媒介提示 + 质量增强词
    """
    core = (brief.get("sd_prompt") or "").strip().rstrip(",")
    if not core:
        core = brief.get("theme", "seamless floral pattern")
        logger.debug("brief 缺少 sd_prompt，退化为 theme 描述")

    sections = [core]
    if comp := build_composition_constraints(config):
        sections.append(comp)
    if medium := build_medium_hints(config):
        sections.append(medium)
    if boosters := build_quality_boosters(config):
        sections.append(boosters)

    prompt = ", ".join(sections)
    logger.debug(f"prompt_engine 输出: {prompt[:120]}...")
    return prompt
