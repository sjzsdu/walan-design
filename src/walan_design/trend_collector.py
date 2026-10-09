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
    trend_summary = json.dumps(trend_data[:50], ensure_ascii=False, indent=2)

    system_prompt = f"""你是一位专业的花型设计师和趋势分析师。
根据以下从国际时尚网站采集的趋势数据，生成 {brief_count} 个花型设计 Brief。

设计要求：
- 目标市场：{style_prefs["target_market"]}
- 偏好风格：{", ".join(style_prefs["preferred_styles"])}
- 偏好配色：{", ".join(style_prefs["color_schemes"])}
- 花型必须是原创的、容易被市场接受的流行风格
- 花型需要能做成四方连续（无缝拼接）

请返回 JSON 数组，每个元素包含：
- title: 花型名称（中文，10字以内）
- theme: 主题描述（中文，30字以内）
- colors: 配色方案（英文逗号分隔的颜色描述）
- style: 风格关键词（英文）
- sd_prompt: Stable Diffusion 的英文 prompt（包含 seamless pattern, tileable 等关键词）
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

    try:
        from openai import OpenAI

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=llm_cfg.get("model", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"趋势数据：\n{trend_summary}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        briefs = result.get("briefs") or result.get("designs") or ([result] if isinstance(result, dict) else result)
        return briefs[:brief_count]
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

        collector = {
            "pinterest": collect_pinterest,
            "behance": collect_behance,
            "patternbank": collect_patternbank,
        }.get(name)
        if collector:
            data = collector(keywords, max_items)
            all_data.extend(data)
            logger.info(f"  {name} 采集到 {len(data)} 条")
        else:
            logger.warning(f"未知来源: {name}")

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
