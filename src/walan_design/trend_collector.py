"""
趋势采集模块
从 Pinterest / Behance / Patternbank 抓取流行花型趋势，
用 LLM 分析后生成设计 Brief。
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_pinterest(keywords: str, max_items: int) -> list:
    results = []
    keyword_list = [k.strip() for k in keywords.split(",")]
    for kw in keyword_list:
        url = f"https://www.pinterest.com/search/pins/?q={requests.utils.quote(kw)}"
        try:
            resp = requests.get(
                url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )
            pins = re.findall(r'"description"\s*:\s*"([^"]+)"', resp.text)
            for desc in pins[: max(max_items // len(keyword_list), 1)]:
                results.append({"source": "pinterest", "keyword": kw, "description": desc})
        except Exception as e:
            logger.warning(f"Pinterest 采集失败 [{kw}]: {e}")
    return results


def collect_behance(keywords: str, max_items: int) -> list:
    results = []
    keyword_list = [k.strip() for k in keywords.split(",")]
    for kw in keyword_list:
        url = f"https://www.behance.net/search?search={requests.utils.quote(kw)}"
        try:
            resp = requests.get(
                url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )
            names = re.findall(r'"name"\s*:\s*"([^"]{5,80})"', resp.text)
            for name in names[: max(max_items // len(keyword_list), 1)]:
                results.append({"source": "behance", "keyword": kw, "description": name})
        except Exception as e:
            logger.warning(f"Behance 采集失败 [{kw}]: {e}")
    return results


def collect_patternbank(keywords: str, max_items: int) -> list:
    results = []
    try:
        resp = requests.get(
            "https://patternbank.com/trend-reports/",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        )
        titles = re.findall(r"<h[23][^>]*>([^<]{5,100})</h[23]>", resp.text)
        for title in titles[:max_items]:
            results.append({"source": "patternbank", "keyword": keywords, "description": title.strip()})
    except Exception as e:
        logger.warning(f"Patternbank 采集失败: {e}")
    return results


def _local_fallback_briefs(count: int, style_prefs: dict) -> list:
    templates = [
        {
            "title": "热带花卉",
            "theme": "热带植物花卉，鲜艳活泼",
            "colors": "vibrant tropical, green, coral, yellow",
            "style": "tropical floral botanical",
            "sd_prompt": (
                "seamless pattern, tropical floral design, vibrant colors, green leaves, "
                "coral flowers, highly detailed, tileable, repeat pattern, fabric textile design"
            ),
            "tags": ["花卉", "植物", "田园", "抽象"],
        },
        {
            "title": "极简几何",
            "theme": "现代几何图案，简洁大方",
            "colors": "monochrome, black, white, gold accent",
            "style": "geometric modern minimal",
            "sd_prompt": (
                "seamless pattern, geometric shapes, modern minimal design, "
                "black and white with gold accents, clean lines, tileable, "
                "repeat pattern, fabric textile design"
            ),
            "tags": ["几何", "抽象", "植物"],
        },
        {
            "title": "抽象有机",
            "theme": "流体抽象图案，柔和自然",
            "colors": "soft pastels, pink, lavender, mint",
            "style": "abstract organic fluid",
            "sd_prompt": (
                "seamless pattern, abstract organic shapes, fluid forms, "
                "soft pastel colors, pink lavender mint, elegant, tileable, "
                "repeat pattern, fabric textile design"
            ),
            "tags": ["抽象", "植物", "花卉"],
        },
        {
            "title": "豹纹变奏",
            "theme": "动物纹理，时尚经典",
            "colors": "warm earth tones, brown, beige, black",
            "style": "animal print leopard",
            "sd_prompt": (
                "seamless pattern, leopard print, animal texture, warm earth tones, "
                "brown beige black, fashion, tileable, repeat pattern, fabric textile design"
            ),
            "tags": ["豹纹", "动物", "几何"],
        },
        {
            "title": "田园花园",
            "theme": "英式田园花卉，浪漫温馨",
            "colors": "warm earth tones, rose, sage, cream",
            "style": "floral botanical cottage",
            "sd_prompt": (
                "seamless pattern, cottage garden flowers, roses, sage green, "
                "cream background, romantic vintage floral, tileable, "
                "repeat pattern, fabric textile design"
            ),
            "tags": ["花卉", "田园", "植物", "传统花卉"],
        },
        {
            "title": "蝴蝶花园",
            "theme": "蝴蝶与花卉，灵动自然",
            "colors": "vibrant, purple, blue, yellow, green",
            "style": "butterfly floral nature",
            "sd_prompt": (
                "seamless pattern, butterflies and flowers, vibrant purple blue yellow, "
                "nature inspired, delicate, tileable, repeat pattern, fabric textile design"
            ),
            "tags": ["蝴蝶", "花卉", "植物", "田园"],
        },
    ]
    _ = style_prefs  # 预留：未来可以根据偏好过滤
    return templates[:count]


def analyze_with_llm(trend_data: list, config: dict) -> list:
    llm_cfg = config["trend"]["llm"]
    brief_count = config["trend"]["brief_count"]
    style_prefs = config["design"]["style_preferences"]
    trend_summary = json.dumps(
        [{k: v for k, v in item.items() if k != "src"} for item in trend_data[:50]],
        ensure_ascii=False,
        indent=2,
    )

    system_prompt = f"""你是一位有 15 年经验的纺织花型设计师兼趋势分析师，服务国际面料市场（女装、家纺、童装）。
根据以下真实采集的趋势数据（含 Pinterest 流行花型的图片描述），生成 {brief_count} 个花型设计 Brief。

专业要求：
- 目标市场：{style_prefs["target_market"]}
- 偏好风格：{", ".join(style_prefs["preferred_styles"])}
- 偏好配色：{", ".join(style_prefs["color_schemes"])}
- 花型必须原创、贴合当下真实流行趋势（不要凭空想象过时款式）
- 必须能做四方连续无缝拼接
- sd_prompt 必须用纺织设计行业标准术语，具体描述：主体母题（motif）的形态与排布方式、
  底色与主色（给出具体色彩名）、笔触/质感（如 hand-painted gouache, flat vector, watercolor wash）、
  母题密度（如 dense allover, spaced tossed）、参考印花工艺（如 digital print style）。
  禁止只写 "beautiful pattern" 这类空话。长度 40-70 词。
- 描述里出现 nautical、 Retro 等具体风格词时优先吸收进 sd_prompt

请返回 JSON 数组，每个元素包含：
- title: 花型名称（中文，10字以内）
- theme: 主题描述（中文，30字以内）
- colors: 配色方案（英文逗号分隔的具体颜色名，如 terracotta, sage green, cream）
- style: 风格关键词（英文）
- sd_prompt: Stable Diffusion 的英文 prompt（以 seamless pattern 开头）
- tags: 适合的瓦栏标签（从这些中选择：几何、动物、抽象、植物、花卉、田园、蝴蝶、豹纹、卡通、传统花卉）
"""

    # 根据 provider 选 API key 环境变量名
    provider = llm_cfg.get("provider", "openai")
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = env_var_map.get(provider, "OPENAI_API_KEY")
    api_key = llm_cfg.get("api_key") or os.environ.get(env_var, "")
    base_url = llm_cfg.get("base_url")  # OpenRouter 需要

    if not api_key:
        logger.warning(f"未配置 {env_var}，使用本地规则生成 Brief")
        return _local_fallback_briefs(brief_count, style_prefs)

    # 视觉分析：趋势数据里带真实下载图片时，直接给视觉模型看图
    vision_cfg = config["trend"].get("vision", {})
    image_items = [t for t in trend_data if t.get("image_path")][: int(vision_cfg.get("max_images", 8))]
    model = llm_cfg.get("model", "gpt-4o")
    user_content = f"趋势数据：\n{trend_summary}"
    if image_items:
        model = vision_cfg.get("model", model)
        logger.info(f"  视觉模式：附加 {len(image_items)} 张真实趋势图（{model}）")

    try:
        import base64

        from openai import OpenAI

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        if image_items:
            content_parts = [
                {
                    "type": "text",
                    "text": "以下是当前 Pinterest 上真实流行的花型图片（附描述数据）。"
                    "请逐张分析它们的母题、配色、风格趋势，再结合趋势数据生成 Brief：\n\n" + trend_summary,
                }
            ]
            for t in image_items:
                with open(t["image_path"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ]
            # 部分免费视觉模型不支持 response_format，靠解析兜底
            response = client.chat.completions.create(model=model, messages=messages, temperature=0.8)
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
            )
        content = response.choices[0].message.content

        # 容错解析：模型可能返回 ```json 包裹或前后带说明文字
        result = None
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]|\{.*\}", content, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
        if result is None:
            raise ValueError(f"LLM 返回内容无法解析为 JSON: {content[:200]}")

        # LLM 可能返回 {"briefs": [...]} / {"designs": [...]} / 裸数组 / 单个对象
        if isinstance(result, dict):
            briefs = result.get("briefs") or result.get("designs") or [result]
        elif isinstance(result, list):
            briefs = result
        else:
            briefs = [result]

        # 规范化每个 brief：tags 可能是字符串（"植物、抽象"）或数组
        valid_tags = set(config["walan"]["existing_tags"])
        normalized = []
        for b in briefs:
            if not isinstance(b, dict):
                continue
            tags = b.get("tags", [])
            if isinstance(tags, str):
                # 按 、/,/空格 拆分
                for sep in ["，", ",", " ", "、"]:
                    tags = tags.replace(sep, "、")
                tags = [t.strip() for t in tags.split("、") if t.strip()]
            elif isinstance(tags, list):
                tags = [str(t).strip() for t in tags if str(t).strip()]
            else:
                tags = []
            # 过滤到瓦栏有效标签集合内；不足时从有效集合补齐
            tags = [t for t in tags if t in valid_tags]
            if len(tags) < 3:
                for t in ["花卉", "植物", "几何", "抽象", "田园"]:
                    if len(tags) >= 3:
                        break
                    if t not in tags:
                        tags.append(t)
            b["tags"] = tags[:5]
            normalized.append(b)

        if not normalized:
            raise ValueError("LLM 未返回有效 brief")
        return normalized[:brief_count]
    except Exception as e:
        logger.error(f"LLM 分析失败: {e}")
        return _local_fallback_briefs(brief_count, style_prefs)


def run(config: dict = None) -> list:
    if config is None:
        config = load_config()

    trend_cfg = config["trend"]
    all_data = []

    for source in trend_cfg["sources"]:
        if not source.get("enabled", True):
            continue
        name = source["name"]
        keywords = source["keywords"]
        max_items = source.get("max_items", 20)
        logger.info(f"从 {name} 采集趋势... 关键词: {keywords}")

        if name == "browser":
            # ego-browser 真实浏览器采集（国内可用，返回带本地图片的 items）
            from walan_design.browser_trends import collect_via_browser

            data = collect_via_browser(keywords, config)
        else:
            collector = {
                "pinterest": collect_pinterest,
                "behance": collect_behance,
                "patternbank": collect_patternbank,
            }.get(name)
            data = collector(keywords, max_items) if collector else []
            if collector is None:
                logger.warning(f"未知来源: {name}")
        if data:
            all_data.extend(data)
            logger.info(f"  {name} 采集到 {len(data)} 条")

    logger.info(f"总共采集到 {len(all_data)} 条趋势数据")
    logger.info("使用 LLM 分析趋势，生成设计 Brief...")
    briefs = analyze_with_llm(all_data, config)

    output_dir = Path(trend_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"briefs_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": timestamp, "trend_data_count": len(all_data), "briefs": briefs},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"生成 {len(briefs)} 个设计 Brief，保存到 {output_file}")

    return briefs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    briefs = run()
    for i, b in enumerate(briefs):
        print(f"\n--- Brief {i + 1} ---")
        print(f"  标题: {b.get('title', '')}")
        print(f"  主题: {b.get('theme', '')}")
        print(f"  配色: {b.get('colors', '')}")
        print(f"  风格: {b.get('style', '')}")
        print(f"  标签: {b.get('tags', [])}")
