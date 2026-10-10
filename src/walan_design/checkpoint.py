"""
Checkpoint 管理器（断点续跑）

功能：流水线每个 step 的产物落盘后记录状态，重启时跳过已完成的 step。

持久化到单个 JSON 文件：output/temp/checkpoint.json

格式：
{
  "pipeline_id": "20261010_173000",
  "started_at": "...",
  "last_step": "psd",
  "steps": {
    "collect":  {"status": "done",  "ts": "...", "output_count": 5, "output_ref": "output/trends/briefs.json"},
    "design":   {"status": "done",  "ts": "...", "output_count": 5, "output_ref": "output/designs/"},
    "psd":      {"status": "done",  "ts": "...", "output_count": 5, "output_ref": "output/psd/"},
    "check":    {"status": "failed","ts": "...", "detail": "psd 生成中断"},
    "upload":   {"status": "pending"}
  }
}

用法（在 pipeline.py 里）：
  ckpt = CheckpointManager(config)
  if ckpt.is_step_done("psd") and config["pipeline"].get("resume", True):
      logger.info("PSD 已完成，跳过")
  else:
      run_psd(results, config)
      ckpt.mark_done("psd", output_count=len(results))
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class CheckpointManager:
    """流水线断点管理器"""

    STEPS = ["collect", "design", "upscale", "recolor", "psd", "check", "upload"]

    def __init__(self, config: dict, pipeline_id: str = None):
        temp_dir = Path(config.get("pipeline", {}).get("temp_dir", "output/temp"))
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.path = temp_dir / "checkpoint.json"
        self.archive_dir = temp_dir / "checkpoint_archive"
        self.data = self._load()

        # 上一批全部完成 → 归档旧 checkpoint，自动开新一批
        # （断点续跑只针对"中断恢复"；跑完的批次不应该挡住下一次运行）
        if self._all_done():
            self._archive()
            self.data = self._new_state()
            self.save()
            logger.info("上一批已全部完成，开启新一批")

        if pipeline_id and self.data.get("pipeline_id") != pipeline_id:
            # 新流水线：覆盖旧状态
            self.data = self._new_state(pipeline_id)
            self.save()

    def _all_done(self) -> bool:
        """所有实际步骤都 done（upscale/recolor 是旧版残留步骤，忽略）"""
        steps = self.data.get("steps", {})
        active = [s for s in self.STEPS if s not in ("upscale", "recolor")]
        return all(steps.get(s, {}).get("status") == "done" for s in active) and bool(steps)

    def _archive(self):
        """把已完成的 checkpoint 归档（留档不删除，供追溯）"""
        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            pid = self.data.get("pipeline_id", "unknown")
            dst = self.archive_dir / f"checkpoint_{pid}.json"
            dst.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"旧 checkpoint 已归档: {dst}")
        except OSError as e:
            logger.warning(f"checkpoint 归档失败（忽略）: {e}")

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        # 首次启动
        return self._new_state()

    def _new_state(self, pipeline_id: str = None) -> dict:
        return {
            "pipeline_id": pipeline_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(),
            "last_step": None,
            "steps": {s: {"status": "pending"} for s in self.STEPS},
        }

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def is_step_done(self, step: str) -> bool:
        return self.data.get("steps", {}).get(step, {}).get("status") == "done"

    def mark_done(self, step: str, output_count: int = 0, output_ref: str = ""):
        self.data["steps"][step] = {
            "status": "done",
            "ts": datetime.now().isoformat(),
            "output_count": output_count,
            "output_ref": output_ref,
        }
        self.data["last_step"] = step
        self.save()

    def mark_failed(self, step: str, detail: str = ""):
        self.data["steps"][step] = {
            "status": "failed",
            "ts": datetime.now().isoformat(),
            "detail": detail,
        }
        self.save()

    def status_of(self, step: str) -> str:
        return self.data.get("steps", {}).get(step, {}).get("status", "unknown")

    def summary(self) -> str:
        lines = [f"Checkpoint ({self.data.get('pipeline_id', '?')}):"]
        for s in self.STEPS:
            st = self.status_of(s)
            icon = {"done": "✅", "failed": "❌", "pending": "⏳", "skipped": "⏭️"}.get(st, "?")
            lines.append(f"  {icon} {s}: {st}")
        return "\n".join(lines)

    def reset(self, step: str = None):
        """重置（全部或指定 step）"""
        if step is None:
            self.data = self._new_state()
        else:
            self.data["steps"][step] = {"status": "pending"}
        self.save()


def resume_pipeline(pipeline_id: str, config: dict) -> dict:
    """
    查询断点状态，返回每个 step 是否需要跳过。
    pipeline_id 为 None 时创建新 checkpoint。
    """
    ckpt = CheckpointManager(config, pipeline_id)
    logger.info(f"\n{ckpt.summary()}")

    resume = {}
    for step in CheckpointManager.STEPS:
        resume[step] = ckpt.is_step_done(step)
    return resume
