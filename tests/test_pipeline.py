"""流水线集成测试（占位）"""
import pytest


def test_config_loads():
    from walan_design.trend_collector import load_config
    config = load_config()
    assert "walan" in config
    assert "trend" in config
    assert "design" in config


def test_pipeline_imports():
    from walan_design.pipeline import main
    assert callable(main)
