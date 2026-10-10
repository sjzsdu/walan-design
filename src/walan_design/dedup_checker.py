"""
防重复检测模块（dedup_checker）

检测维度：
  1. 同批次内跨设计相似度（排除同一 brief 的配色变体）
  2. 与已发布历史作品库的相似度

检测算法：
  - pHash（感知哈希）：比较视觉结构相似度，汉明距离越小越相似
  - 颜色直方图：辅助判断配色是否雷同

为什么不直接比 MD5：AI 生图即使同一 prompt 每次像素都不同，但结构上可能高度相似——
pHash 抓的是"看起来像不像"，而不是"是不是同一文件"。

降级策略：
  - 批次内相似 → 跳过（只保留第一个，其余标 _dedup_skipped=True）
  - 与历史库相似 → 跳过
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# 指纹计算
# ============================================================

def compute_phash(img: Image.Image, hash_size: int = 8) -> str:
    """
    计算图片的感知哈希（pHash）。
    算法：缩放 hash_size*hash_size → 灰度 → DCT → 取低频左上角 → 与均值比较生成 0/1 位。
    返回十六进制字符串（hash_size²/4 字节）。
    """
    try:
        # 缩放到 32x32 计算 DCT 更准确
        small = img.convert("L").resize((32, 32), Image.LANCZOS)
        arr = np.array(small, dtype=np.float32)

        # DCT（numpy 实现）
        dct = _dct2(arr)

        # 取 hash_size x hash_size 低频区域
        low_freq = dct[:hash_size, :hash_size]
        median = np.median(low_freq)

        # 生成 0/1 位
        bits = (low_freq > median).astype(int).flatten().tolist()

        # 转十六进制字符串（每 4 bit 转 1 hex 字符，不足补 0）
        hex_str = ""
        # 补齐到 4 的倍数
        while len(bits) % 4 != 0:
            bits.append(0)
        for i in range(0, len(bits), 4):
            nibble = bits[i:i + 4]
            hex_str += hex(int("".join(str(b) for b in nibble), 2))[2:]
        return hex_str
    except Exception as e:
        logger.warning(f"pHash 计算失败: {e}")
        return ""


def _dct2(a: np.ndarray) -> np.ndarray:
    """二维 DCT-II 纯 numpy 实现（matrix multiply）"""
    n = a.shape[0]
    # DCT-II 变换矩阵: C[k, n] = cos(pi/N * (n+0.5) * k)
    k = np.arange(n).reshape(-1, 1)  # (N, 1)
    x = np.arange(n).reshape(1, -1)  # (1, N)
    C = np.cos(np.pi / n * (x + 0.5) * k)  # (N, N)
    # 对行做 DCT，再对列做 DCT: C @ arr @ C.T
    return C @ a @ C.T


def compute_histogram(img: Image.Image, bins: int = 8) -> list:
    """
    计算颜色直方图（RGB 各通道 bins 个桶，共 bins³ 桶但只返回扁平计数）。
    用于配色相似度的辅助判断。
    """
    try:
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        # 把 [0,255] 映射到 [0, bins-1]，避免 256 溢出
        quantized = np.clip(arr * (bins - 1) / 255.0, 0, bins - 1).astype(int).reshape(-1, 3)
        # 计算三维直方图
        hist, _ = np.histogramdd(quantized, bins=[bins, bins, bins], range=[(0, bins), (0, bins), (0, bins)])
        return hist.flatten().tolist()
    except Exception as e:
        logger.warning(f"直方图计算失败: {e}")
        return []


def hamming_distance(hex_a: str, hex_b: str) -> int:
    """两个十六进制 pHash 之间的汉明距离"""
    if not hex_a or not hex_b or len(hex_a) != len(hex_b):
        return 999  # 无法比较
    a = int(hex_a, 16)
    b = int(hex_b, 16)
    return bin(a ^ b).count("1")


# ============================================================
# 历史作品库（简单 JSON，避免过度设计）
# ============================================================

class HistoryStore:
    """已发布作品的指纹库，持久化到 JSON 文件"""

    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._records = self._load()

    def _load(self) -> list:
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"历史库损坏，重建: {e}")
                return []
        return []

    def save(self):
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    def add(self, record: dict):
        self._records.append(record)
        self.save()

    def all(self) -> list:
        return list(self._records)

    def phashes(self) -> list[str]:
        return [r.get("phash", "") for r in self._records if r.get("phash")]

    def count(self) -> int:
        return len(self._records)


# ============================================================
# 主检测流程
# ============================================================

def extract_fingerprints(design_results: list) -> list[dict]:
    """
    从 design_results 提取每张 PNG 的指纹。
    design_results 结构见 ai_designer.run 的输出——每个元素有 brief 和 image_paths。
    """
    fingerprints = []
    for i, result in enumerate(design_results):
        brief_id = i  # 同一 brief 的变体共享
        brief = result.get("brief", {})
        title = brief.get("title", f"design_{i+1}")
        variants = result.get("image_paths", [])
        # image_paths 里第 0 张是原图，1+ 是配色变体
        for j, img_path in enumerate(variants):
            try:
                img = Image.open(img_path).convert("RGB")
                fp = {
                    "brief_id": brief_id,
                    "title": title,
                    "variant_index": j,
                    "image_path": str(Path(img_path).resolve()),
                    "phash": compute_phash(img),
                    "histogram": compute_histogram(img),
                    "is_variant": j > 0,
                }
                fingerprints.append(fp)
            except Exception as e:
                logger.warning(f"指纹提取失败 {img_path}: {e}")
    return fingerprints


def batch_dedup(
    fingerprints: list[dict],
    phash_threshold: int = 10,
    same_brief_exempt: bool = True,
) -> set[int]:
    """
    批次内相似度检测。
    返回应被跳过的 fingerprint 索引集合。

    参数：
      phash_threshold: pHash 汉明距离阈值。越小越严格。
        - 0-5: 几乎完全相同
        - 6-10: 非常相似（可能是同一素材的变体）
        - 11-20: 中等相似
        - 21+: 差异明显
      same_brief_exempt: 同一 brief 的配色变体是否豁免（推荐 True）
        因为 recolor 变体的结构是一样的，只是配色不同。
    """
    skip_indices = set()
    n = len(fingerprints)

    for i in range(n):
        if i in skip_indices:
            continue
        fp_i = fingerprints[i]
        for j in range(i + 1, n):
            if j in skip_indices:
                continue
            fp_j = fingerprints[j]

            # 同一 brief 的变体 → 豁免
            if same_brief_exempt and fp_i["brief_id"] == fp_j["brief_id"]:
                continue

            # 比 pHash
            dist = hamming_distance(fp_i["phash"], fp_j["phash"])
            if dist <= phash_threshold:
                logger.info(
                    f"  批次内相似 (汉明距离={dist}): "
                    f"[{fp_i['title']} v{fp_i['variant_index']}] vs "
                    f"[{fp_j['title']} v{fp_j['variant_index']}]"
                )
                skip_indices.add(j)  # 保留第一个，跳过后续

    return skip_indices


def history_dedup(
    fingerprints: list[dict],
    history: HistoryStore,
    phash_threshold: int = 8,
) -> set[int]:
    """
    与历史库比对。
    phash_threshold 比批次内更严格（8 vs 10），避免和自己已发布的撞车。
    """
    skip_indices = set()
    history_hashes = history.phashes()

    if not history_hashes:
        logger.info("  历史库为空，跳过历史比对")
        return skip_indices

    for i, fp in enumerate(fingerprints):
        if not fp.get("phash"):
            continue
        for hist_hash in history_hashes:
            dist = hamming_distance(fp["phash"], hist_hash)
            if dist <= phash_threshold:
                logger.info(
                    f"  与历史相似 (汉明距离={dist}): "
                    f"[{fp['title']} v{fp['variant_index']}]"
                )
                skip_indices.add(i)
                break

    return skip_indices


def check_and_mark(
    design_results: list,
    config: dict,
) -> list:
    """
    入口函数：检测 + 在 design_results 上标记 _dedup_skipped。
    返回过滤后的 design_results（去掉被跳过得）。

    config 新增字段（放在 config.yaml quality 下）：
      quality.dedup.enabled: true
      quality.dedup.phash_threshold_batch: 10   # 批次内阈值
      quality.dedup.phash_threshold_history: 8   # 历史库阈值
      quality.dedup.same_brief_exempt: true      # 同 brief 配色变体豁免
      quality.dedup.history_store: "output/history/phash_store.json"
    """
    dedup_cfg = config.get("quality", {}).get("dedup", {})
    if not dedup_cfg.get("enabled", True):
        logger.info("防重复检测已禁用")
        return design_results

    ph_batch = dedup_cfg.get("phash_threshold_batch", 10)
    ph_hist = dedup_cfg.get("phash_threshold_history", 8)
    same_brief_exempt = dedup_cfg.get("same_brief_exempt", True)
    store_path = dedup_cfg.get("history_store", "output/history/phash_store.json")

    logger.info(f"\n=== 防重复检测 ===")
    fingerprints = extract_fingerprints(design_results)
    logger.info(f"  提取指纹: {len(fingerprints)} 张")

    # 批次内
    batch_skip = batch_dedup(fingerprints, ph_batch, same_brief_exempt)
    logger.info(f"  批次内相似跳过: {len(batch_skip)} 张")

    # 历史库
    history = HistoryStore(store_path)
    logger.info(f"  历史库记录: {history.count()} 条")
    hist_skip = history_dedup(fingerprints, history, ph_hist)
    logger.info(f"  历史相似跳过: {len(hist_skip)} 张")

    all_skip = batch_skip | hist_skip
    logger.info(f"  总计跳过: {len(all_skip)}/{len(fingerprints)} 张")

    # 标记 design_results
    # fingerprint 索引映射回 (brief_id, variant_index)
    results_to_skip = set()
    for idx in all_skip:
        fp = fingerprints[idx]
        results_to_skip.add((fp["brief_id"], fp["variant_index"]))

    # 过滤：至少保留每个 brief 的主图（variant_index=0）——如果主图也被相似判过，整个 brief 跳过
    brief_has_kept = set()
    for idx, result in enumerate(design_results):
        all_variants_skipped = all(
            (idx, j) in results_to_skip
            for j in range(len(result.get("image_paths", [])))
        )
        if all_variants_skipped:
            title = result.get("brief", {}).get("title", f"design_{idx}")
            logger.warning(f"  → 设计 [{title}] 全部变体相似，跳过整个设计")
            result["_dedup_skipped"] = True
        else:
            # 只保留未被跳过的变体路径
            kept_paths = [
                p for j, p in enumerate(result.get("image_paths", []))
                if (idx, j) not in results_to_skip
            ]
            result["image_paths"] = kept_paths
            brief_has_kept.add(idx)

    kept = [r for r in design_results if not r.get("_dedup_skipped")]
    skipped_count = len(design_results) - len(kept)
    logger.info(f"  检测完成: {len(kept)}/{len(design_results)} 个设计保留")

    return kept


def record_published(design_results: list, config: dict):
    """
    发布成功后调用：把已发布作品的指纹写入历史库，下次比对用。
    在 pipeline 里放在 upload 成功后。
    """
    dedup_cfg = config.get("quality", {}).get("dedup", {})
    if not dedup_cfg.get("enabled", True):
        return

    store_path = dedup_cfg.get("history_store", "output/history/phash_store.json")
    history = HistoryStore(store_path)

    now = datetime.now().isoformat()
    for i, result in enumerate(design_results):
        brief = result.get("brief", {})
        title = brief.get("title", f"design_{i}")
        # 只存主图（variant_index=0）的指纹——主图代表该设计
        for j, img_path in enumerate(result.get("image_paths", [])):
            if j != 0:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                record = {
                    "brief_id": i,
                    "title": title,
                    "variant_index": j,
                    "image_path": str(Path(img_path).resolve()),
                    "phash": compute_phash(img),
                    "histogram": compute_histogram(img),
                    "tags": brief.get("tags", []),
                    "published_at": now,
                }
                history.add(record)
                logger.info(f"  入库: {title}")
            except Exception as e:
                logger.warning(f"  入库失败 {title}: {e}")


# ============================================================
# CLI 快速测试
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    from walan_design.pipeline import load_config
    from walan_design.trend_collector import _local_fallback_briefs
    from walan_design.ai_designer import run as design_run

    config = load_config()
    briefs = _local_fallback_briefs(3, config["design"]["style_preferences"])
    results = design_run(briefs, config)

    kept = check_and_mark(results, config)
    print(f"\n保留: {len(kept)}/{len(results)} 个设计")
