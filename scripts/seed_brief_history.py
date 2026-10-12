"""种子脚本：把 output/runs/ 下已有的 brief.json 导入 brief_store.json"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from walan_design.brief_history import append_briefs

runs_dir = Path("output/runs")
briefs = []
for brief_file in sorted(runs_dir.glob("*/brief.json")):
    try:
        b = json.loads(brief_file.read_text(encoding="utf-8"))
        briefs.append(b)
        print(f"  + {b.get('title', '?')} ({b.get('theme', '')[:30]})")
    except Exception as e:
        print(f"  ! {brief_file}: {e}")

if briefs:
    append_briefs(briefs, config={"quality": {"dedup": {"brief_history_store": "output/history/brief_store.json"}}})
    print(f"\n✓ 已导入 {len(briefs)} 条历史 brief")
else:
    print("output/runs/ 下没有 brief.json")
