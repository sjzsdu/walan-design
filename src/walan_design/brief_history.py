"""
Brief 历史库 — 记录已成功产出的花型 Brief，下次生成时注入 prompt 让 LLM 主动避开重复。

跟 phash_store.json 同级同模式（output/history/），但记录的是语义级主题而非像素指纹。
写入时机：upload 成功后（失败品不算已占用主题，跟 phash_store 一致）。
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STORE = "output/history/brief_store.json"
DEFAULT_MAX = 50  # 历史上限，避免 prompt 膨胀


def _store_path(config: dict = None) -> Path:
    dedup_cfg = (config or {}).get("quality", {}).get("dedup", {})
    path = dedup_cfg.get("brief_history_store", DEFAULT_STORE)
    return Path(path)


def _max_history(config: dict = None) -> int:
    dedup_cfg = (config or {}).get("quality", {}).get("dedup", {})
    return int(dedup_cfg.get("brief_history_size", DEFAULT_MAX))


def load_recent(config: dict = None, n: int = None) -> list[dict]:
    """加载最近 n 条历史 brief（按时间倒序）。n 缺省用 config 的上限。"""
    path = _store_path(config)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, dict):
        records = data.get("briefs", [])
    else:
        records = data
    max_n = n or _max_history(config)
    return records[:max_n]


def append_briefs(briefs: list[dict], config: dict = None):
    """追加新 brief 到历史库（只保留 title/theme/colors/floral_type/tags，省空间）"""
    path = _store_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing = data.get("briefs", []) if isinstance(data, dict) else data
        except (json.JSONDecodeError, OSError):
            existing = []

    new_records = []
    for b in briefs:
        if not isinstance(b, dict):
            continue
        new_records.append(
            {
                "title": b.get("title", ""),
                "theme": b.get("theme", ""),
                "colors": b.get("colors", ""),
                "floral_type": b.get("floral_type", ""),
                "tags": b.get("tags", []),
            }
        )

    merged = new_records + existing
    # 去重（按 title 精确匹配）
    seen_titles = set()
    deduped = []
    for r in merged:
        t = r.get("title", "")
        if t and t not in seen_titles:
            seen_titles.add(t)
            deduped.append(r)

    max_n = _max_history(config)
    deduped = deduped[:max_n]

    payload = {"briefs": deduped, "total": len(deduped)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"brief_history: 写入 {len(new_records)} 条，历史库共 {len(deduped)} 条")


def format_for_prompt(config: dict = None, max_items: int = 20) -> str:
    """格式化为 LLM prompt 可用的排除指令文本。空历史或关闭时返回空字符串。"""
    # brief_history_size=0 表示关闭
    if _max_history(config) == 0:
        return ""
    recent = load_recent(config, n=max_items)
    if not recent:
        return ""

    lines = []
    for r in recent:
        tags = ", ".join(r.get("tags", []))
        theme = r.get("theme", "")[:30]
        line = f"- {r.get('title', '')}（{theme}）"
        if r.get("colors"):
            line += f" 配色: {r['colors']}"
        if r.get("tags"):
            line += f" 标签: {tags}"
        lines.append(line)

    header = (
        f"\n\n【重要排除指令】以下是你最近已生成的 {len(recent)} 个花型，"
        "请**不要重复任何相似的主题、配色、母题组合**——"
        "你必须想出全新的方向（不同的植物/动物/几何母题、不同的色调、不同的排布方式）：\n"
    )
    return header + "\n".join(lines)
