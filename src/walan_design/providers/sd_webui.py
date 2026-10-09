"""
本地 Stable Diffusion WebUI 供应商
API: http://127.0.0.1:7860/sdapi/v1/txt2img

优势:
  - 免费（不花 API 费）
  - ControlNet 支持无缝平铺
  - 完全可控（模型/LoRA/采样器全可调）

劣势:
  - 需要本地 GPU（Mac M1/M2 可跑但慢，8GB VRAM 最低要求）
  - 分辨率受限于显存（512×512 基准，高清修复可达 2048×2048）
  - 需要手动启动 WebUI
"""

import base64
import io
import logging

import requests
from PIL import Image

from .base import GenerationParams, GenerationResult, ImageProvider

logger = logging.getLogger(__name__)


class SDWebUIProvider(ImageProvider):
    """本地 SD WebUI API"""

    name = "stable_diffusion"
    max_resolution = (2048, 2048)
    supports_tiling = True  # 通过 ControlNet tile
    supports_2_3_ratio = True

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_url = config.get("stable_diffusion", {}).get("api_url", "http://127.0.0.1:7860")
        self.checkpoint = config.get("stable_diffusion", {}).get("checkpoint", "")
        self.use_controlnet = config.get("stable_diffusion", {}).get("use_controlnet", True)
        self.controlnet_model = config.get("stable_diffusion", {}).get("controlnet_model", "control_v11p_sd15_tile")
        logger.info(f"SD WebUI provider: api_url={self.api_url}, checkpoint={self.checkpoint or '<server default>'}")

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.api_url}/sdapi/v1/options", timeout=5)
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
            "batch_size": params.batch_size,
            "sampler_name": "DPM++ 2M Karras",
        }

        # ControlNet 无缝平铺
        if params.tiling and self.use_controlnet:
            payload["alwayson_scripts"] = {
                "ControlNet": {
                    "args": [
                        {
                            "model": self.controlnet_model,
                            "module": "tile_resample",
                            "weight": 0.5,
                        }
                    ]
                }
            }

        logger.info(f"  SD WebUI: {params.prompt[:60]}... ({params.width}×{params.height})")

        try:
            resp = requests.post(f"{self.api_url}/sdapi/v1/txt2img", json=payload, timeout=180)
            resp.raise_for_status()
            result = resp.json()

            images = [Image.open(io.BytesIO(base64.b64decode(b))) for b in result.get("images", [])]

            logger.info(f"  SD WebUI 成功: {len(images)} 张图")

            return GenerationResult(
                images=images,
                provider=self.name,
                raw_resolution=(params.width, params.height),
                seed_used=result.get("seed", -1),
                metadata={"checkpoint": self.checkpoint},
            )

        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接 SD WebUI ({self.api_url})")
            raise RuntimeError(f"SD WebUI 不可用: {self.api_url}")
        except Exception as e:
            logger.error(f"SD WebUI 生成失败: {e}")
            raise
