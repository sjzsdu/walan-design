"""make_seamless 算法、质量门过滤与供应商配置的回归测试"""
import numpy as np
from PIL import Image

from walan_design.ai_designer import make_seamless
from walan_design.quality_checker import check_seamless, run as qc_run
from walan_design.upscaler import upscale_pil


def _random_image(w: int = 384, h: int = 256, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_make_seamless_edges_match_exactly():
    """α 方向修复后：col0 == col(w-1)、row0 == row(h-1)，wrap 差异归零"""
    out = np.array(make_seamless(_random_image()))
    assert np.array_equal(out[:, 0, :], out[:, -1, :])
    assert np.array_equal(out[0, :, :], out[-1, :, :])


def test_make_seamless_after_upscale_passes_tolerance(tmp_path):
    """验收标准：LANCZOS 放大之后再做接回位，最终图 check_seamless(容差=0) 通过（平均边缘差异 ≤0.1）"""
    img = _random_image()
    upscaled = upscale_pil(img, 768, 1152)

    # 未处理的放大图应 FAIL（证明测试有分辨力）
    raw_path = str(tmp_path / "raw.png")
    upscaled.save(raw_path, "PNG", dpi=(300, 300))
    raw_passed, raw_detail = check_seamless(raw_path, tolerance=0)
    assert not raw_passed

    # 放大后接回位 → PASS
    final = make_seamless(upscaled)
    final_path = str(tmp_path / "final.png")
    final.save(final_path, "PNG", dpi=(300, 300))
    passed, detail = check_seamless(final_path, tolerance=0)
    avg_diff = float(detail.split("=")[1].split(" ")[0])
    assert passed
    assert avg_diff <= 0.1


def test_stability_model_normalization():
    """config 里的 SDXL-1.0 等别名应归一化为 sdxl-v1，且初始化即打印 model+endpoint"""
    from walan_design.providers.stability import StabilityProvider

    for raw, expected in [("SDXL-1.0", "sdxl-v1"), ("sdxl-v1", "sdxl-v1"), ("core", "core")]:
        provider = StabilityProvider({"stability": {"model": raw}})
        assert provider.model == expected
        if expected == "sdxl-v1":
            assert provider._endpoint_for(expected).endswith("/stable-diffusion-xl-1024-v1-0/text-to-image")


def _flat_png(path, seam_break=False):
    """生成一张 2:3 的纯色测试 PNG；seam_break 时左右边缘差异巨大"""
    arr = np.full((450, 300, 3), 200, dtype=np.uint8)
    if seam_break:
        arr[:, 0, :] = [10, 20, 30]
    Image.fromarray(arr).save(str(path), "PNG", dpi=(300, 300))


def _qc_config():
    return {
        "quality": {
            "enabled": True,
            "checks": {
                "resolution": True, "seamless": True, "originality": True,
                "no_text": True, "color_mode": True, "color_variants": True,
                "tag_count": True, "dimensions": True,
            },
            "on_fail": "skip",
            "originality_threshold": 0.85,
            "max_retries": 3,
        },
        "psd": {"seamless_tolerance": 0, "color_mode": "RGB"},
        "design": {"color_variant_count": 4},
        "walan": {"tag_min": 3, "tag_max": 5},
    }


def test_quality_gate_filters_failed_design(tmp_path):
    """质量门必须真正过滤：seamless FAIL 的设计不能进入通过列表（finding 4/5）"""
    good = {"brief": {"title": "good", "tags": ["a", "b", "c"]}, "image_paths": [], "psd_paths": []}
    bad = {"brief": {"title": "bad", "tags": ["a", "b", "c"]}, "image_paths": [], "psd_paths": []}
    for r in (good, bad):
        for j in range(4):
            p = tmp_path / f"{r['brief']['title']}_{j + 1}.png"
            _flat_png(p, seam_break=(r is bad))
            r["image_paths"].append(str(p))

    passed = qc_run([good, bad], _qc_config())

    assert [r["brief"]["title"] for r in passed] == ["good"]
    # originality 是显式 no-op，检测记录里必须可见而不是静默缺失
    first_file = list(good["checks"].values())[0]
    assert "originality" in first_file
    assert first_file["originality"][0] is True


def test_pipeline_uploads_only_passed_results(monkeypatch):
    """pipeline 编排：run_upload 只能拿到 run_check 过滤后的设计（finding 4）"""
    from walan_design import pipeline

    r1, r2 = {"id": 1}, {"id": 2}
    captured = {}
    monkeypatch.setattr(pipeline, "run_collect", lambda config: [])
    monkeypatch.setattr(pipeline, "run_design", lambda briefs, config: [r1, r2])
    monkeypatch.setattr(pipeline, "run_psd", lambda results, config: None)
    monkeypatch.setattr(pipeline, "run_check", lambda results, config: [r1])

    def fake_upload(results, config):
        captured["results"] = results
        return {"status": "none"}

    monkeypatch.setattr(pipeline, "run_upload", fake_upload)
    pipeline.run_pipeline({"trend": {"brief_count": 1}, "pipeline": {"batch_count": 1}}, mode="full")
    assert captured["results"] == [r1]
