"""
AI 视觉自评闭环 — 生图后用视觉模型扮演"花型设计总监"打分，
低于阈值的图自动带着改进建议重新生成，只让高质量图进入后续流程。

评分维度：
- composition 构图平衡（主次分明、留白和谐）
- detail 细节丰富度（纺织印花可看的细节层次）
- color 配色和谐度（色彩关系是否高级）
- marketability 商业潜力（是否像市场上会被买的布料花型）
- flatness 平面印花感（无 3D/照片感/阴影透视，可做四方连续）

失败策略：LLM 不可用/解析失败 → fail-open（视为通过），不阻塞流水线。
"""

import base64
import io
import json
import logging
import os
import re

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"

CRITIC_SYSTEM_PROMPT = """You are a strict art director at a textile design studio \
reviewing AI-generated fabric patterns for a marketplace (Walalnwalan, Chinese textile platform).

Score the pattern image 0-100 on each dimension:
- composition: balanced layout, clear focal hierarchy, harmonious negative space
- detail: rich layers of detail suitable for fabric printing, not muddy or empty
- color: sophisticated color harmony, no garish or muddy combos
- marketability: would a fashion/home decorator buy fabric with this print
- flatness: flat print aesthetic (NO 3D render, NO photo perspective, NO cast shadows, \
NO frame/border), must look tileable as a seamless repeat

overall = weighted score (composition 25%, detail 20%, color 20%, marketability 20%, flatness 15%).
Be strict: generic blobs, harsh gradients, obvious AI artifacts, or photo-like images should score below 60.

Also write "suggestions": 1-2 short English style phrases describing concrete improvements \
for an image-generation prompt (e.g. "denser motif spacing, softer pastel palette"). \
No Chinese in suggestions.

Respond with ONLY a JSON object:
{"composition": n, "detail": n, "color": n, "marketability": n, "flatness": n, "overall": n, "suggestions": "..."}"""


def _vision_config(config: dict) -> dict:
    return (config or {}).get("design", {}).get("visual_review", {})


def _get_api_key(config: dict) -> str:
    cfg = _vision_config(config)
    return cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")


def _encode_image(image: Image.Image, max_side: int = 768) -> str:
    """压缩为 JPEG base64（控制 token 成本）"""
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _call_vision_model(client, model: str, b64: str, theme: str) -> str:
    """
    单次视觉调用，返回原始文本。
    注意：不用 system 角色 — openrouter/free 路由的部分模型在 system+image
    组合下会原样复述 JSON 模板（字面量 n 占位符）导致解析失败，
    实测 user 单角色 + 带数字示例最稳。
    """
    user_text = (
        f"{CRITIC_SYSTEM_PROMPT}\n\n"
        f"Now review this fabric pattern design. Intended theme: {theme}\n"
        'Respond with ONLY the filled JSON object with numeric scores, '
        'e.g. {"composition": 75, "detail": 60, "color": 82, "marketability": 70, '
        '"flatness": 88, "overall": 73, "suggestions": "denser motif spacing, softer palette"}'
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=2000,  # 推理型免费模型会消耗额外 token，给足否则返回空内容
        temperature=0.2,
    )
    if not response.choices:
        raise ValueError(f"响应无 choices: {response.model_extra or response}")
    message = response.choices[0].message
    content = (message.content or "").strip()
    # 推理型模型可能把答案放进 reasoning 而非 content
    if not content:
        content = (getattr(message, "reasoning", "") or "").strip()
        logger.debug("content 为空，回退解析 reasoning 字段")
    if not content:
        raise ValueError("模型返回空内容（content 与 reasoning 均为空）")
    return content


def _extract_json(content: str) -> dict:
    """容错解析：支持 ```json 包裹、前后说明文字、reasoning 叙述中夹带"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # 依次尝试各候选 {...} 块（贪婪匹配可能抓到模板/多个块，取能解析的）
    candidates = re.findall(r"\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}", content, re.DOTALL)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # 最后兜底：推理文本被截断没写出完整 JSON 时，直接抢救 "overall": 数字
    overall_m = re.search(r'"overall"\s*:\s*(\d+(?:\.\d+)?)', content)
    if overall_m:
        data = {"overall": overall_m.group(1)}
        for key in ("composition", "detail", "color", "marketability", "flatness"):
            m = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', content)
            if m:
                data[key] = m.group(1)
        s_m = re.search(r'"suggestions"\s*:\s*"([^"]*)"', content)
        if s_m:
            data["suggestions"] = s_m.group(1)
        logger.info("  完整 JSON 缺失，已从文本抢救评分键值")
        return data
    raise ValueError(f"无法从模型输出解析 JSON: {content[:150]}")


def critique_image(image: Image.Image, brief: dict, config: dict) -> dict:
    """
    对生成的花型图打分。
    返回: {"overall": 0-100, "breakdown": {...}, "suggestions": str, "passed": bool, "critic_model": str}
    任何异常 → fail-open（passed=True, overall=100）
    """
    vcfg = _vision_config(config)
    threshold = float(vcfg.get("threshold", 70))
    fail_open = {
        "overall": 100.0,
        "breakdown": {},
        "suggestions": "",
        "passed": True,
        "critic_model": "disabled",
    }

    api_key = _get_api_key(config)
    if not api_key:
        logger.warning("视觉自评未配置 API key，跳过评分")
        return fail_open

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=vcfg.get("base_url", "https://openrouter.ai/api/v1"),
        )
        b64 = _encode_image(image)
        theme = brief.get("theme", "") or brief.get("sd_prompt", "")[:120]
        model = vcfg.get("model", DEFAULT_MODEL)

        content = _call_vision_model(client, model, b64, theme)
        try:
            data = _extract_json(content)
        except ValueError:
            # 免费模型输出不稳定，解析失败重试一次
            logger.info("  自评解析失败，重试一次...")
            content = _call_vision_model(client, model, b64, theme)
            data = _extract_json(content)

        overall = float(data.get("overall", 0))
        breakdown = {k: data.get(k) for k in ("composition", "detail", "color", "marketability", "flatness")}
        result = {
            "overall": overall,
            "breakdown": breakdown,
            "suggestions": str(data.get("suggestions", "")),
            "passed": overall >= threshold,
            "critic_model": vcfg.get("model", DEFAULT_MODEL),
        }
        verdict = "通过" if result["passed"] else "不通过"
        logger.info(f"  视觉自评: {overall:.0f} 分 ({verdict}) | {result['suggestions'][:80]}")
        return result
    except Exception as e:
        logger.warning(f"视觉自评失败（fail-open）: {e}")
        return fail_open
