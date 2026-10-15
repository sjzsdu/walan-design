"""
本地 diffusers 生图供应商（HTTP 客户端）

走常驻的 scripts/diffusers_server.py（A1111 venv python 跑），
服务内部用 diffusers + circular padding，模型只加载一次。
"""

import io
import logging

import requests
from PIL import Image

from .base import GenerationParams, GenerationResult, ImageProvider

logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:7861"


class DiffusersLocalProvider(ImageProvider):
    """本地 diffusers 生图（常驻 HTTP 服务）"""

    name = "diffusers_local"
    max_resolution = (2048, 2048)
    supports_tiling = True  # circular padding 按构造四方连续
    supports_2_3_ratio = True

    def __init__(self, config: dict):
        super().__init__(config)
        sd_cfg = config.get("stable_diffusion", {})
        self.server_url = sd_cfg.get("api_url", API_URL)
        self.timeout = int(sd_cfg.get("timeout_s", 1800))
        # server 端模型注册表 key（juggernaut=美学精调 / base=SDXL 原始底座）
        self.model_key = sd_cfg.get("model_key", "juggernaut")
        logger.info(
            f"diffusers_local provider: server={self.server_url}, model={self.model_key}, timeout={self.timeout}s"
        )

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, params: GenerationParams) -> GenerationResult:
        payload = {
            "prompt": f"{params.prompt}, high quality, detailed, professional textile design",
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "cfg_scale": params.cfg_scale,
            "seed": params.seed,
            "model": self.model_key,
        }
        logger.info(f"  diffusers_local: {params.prompt[:60]}... ({params.width}×{params.height})")
        try:
            resp = requests.post(f"{self.server_url}/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            img = Image.open(result["image_path"])
            logger.info(f"  diffusers_local 成功: {result.get('size')} 用时 {result.get('elapsed_s')}s")
            return GenerationResult(
                images=[img],
                provider=self.name,
                raw_resolution=(params.width, params.height),
                seed_used=params.seed,
                metadata={"elapsed_s": result.get("elapsed_s")},
            )
        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接 diffusers 服务 ({self.server_url})，请先启动 scripts/diffusers_server.py")
            raise RuntimeError(f"diffusers 本地服务不可用: {self.server_url}")
        except Exception as e:
            logger.error(f"diffusers_local 生成失败: {e}")
            raise
