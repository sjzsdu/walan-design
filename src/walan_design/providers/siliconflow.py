"""
硅基流动（SiliconFlow）云生图供应商
API: https://api.siliconflow.cn/v1/images/generations

选型理由（2026-10-12，本地 SDXL 换云）：
  - 本地 M2 16GB 跑 SDXL 26 分钟/张且服务易 down，云端 2-10 秒/张零运维
  - Z-Image-Turbo ¥0.036/张（跑量主力）、FLUX.2 [pro] $0.03/张（精品档）
  - OpenAI 风格 API，国内直连，支付宝充值

注意：
  - 云 API 不支持 UNet 级 circular padding（构造性四方连续），
    四方连续由 quilt 后处理保证（seam_quilt，确定性算法，此前 5 个上架作品即此路线）
  - 生成 URL 仅 1 小时有效，必须立即下载转存 PIL
"""

import logging
import time

import requests
from PIL import Image

from .base import GenerationParams, GenerationResult, ImageProvider

logger = logging.getLogger(__name__)


class SiliconFlowProvider(ImageProvider):
    """硅基流动云生图（Z-Image-Turbo / FLUX / Kolors 等托管模型）"""

    name = "siliconflow"
    max_resolution = (2048, 2048)
    supports_tiling = False  # 无构造性 tiling，靠 quilt 后处理
    supports_2_3_ratio = True

    def __init__(self, config: dict):
        super().__init__(config)
        import os

        sf_cfg = config.get("stable_diffusion", {}).get("siliconflow", {})
        self.api_url = sf_cfg.get("api_url", "https://api.siliconflow.cn/v1/images/generations")
        self.user_info_url = sf_cfg.get("user_info_url", "https://api.siliconflow.cn/v1/user/info")
        self.model = sf_cfg.get("model", "Z-Image-Turbo")
        self.api_key = sf_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", "")
        self.timeout = int(sf_cfg.get("timeout_s", 120))
        self.health_reason = ""
        logger.info(f"SiliconFlow provider: model={self.model}")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # 2026-09-30 起硅基流动生图默认加「AI 生成」显式水印（瓦栏审核毙水印图）。
            # 官方条款：对生成图片做下游进一步处理时可显式传 0，自行承担最终输出的
            # 标识义务——本项目对图片做 quilt/放大/PSD 加工，且瓦栏上传走 AI 设计声明
            "X-Enable-Watermark": "0",
        }

    def health_check(self) -> bool:
        if not self.api_key:
            self.health_reason = "缺少 api_key（config design.stable_diffusion.siliconflow.api_key 或环境变量 SILICONFLOW_API_KEY）"
            return False
        # 2026-10 user/info 端点返回 410 Gone，生成 API 又确实可用。
        # 轻量鉴权探测：调一次模型列表端点（免费、不扣费）
        try:
            resp = requests.get("https://api.siliconflow.cn/v1/models?sub_type=text-to-image&type=image",
                                headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                return True
            self.health_reason = f"硅基流动鉴权失败: HTTP {resp.status_code}"
            return False
        except Exception as e:
            self.health_reason = f"硅基流动不可达: {e}"
            return False

    def generate(self, params: GenerationParams) -> GenerationResult:
        payload = {
            "model": self.model,
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt or None,
            "image_size": f"{params.width}x{params.height}",
            "num_inference_steps": params.steps,
            "seed": params.seed if params.seed >= 0 else None,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        logger.info(f"  SiliconFlow[{self.model}]: {params.prompt[:60]}... ({params.width}×{params.height})")
        t0 = time.time()

        try:
            resp = requests.post(self.api_url, json=payload, headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            images = []
            for item in data.get("images", []):
                url = item.get("url")
                if url:
                    # URL 仅 1 小时有效，立即下载
                    img_resp = requests.get(url, timeout=60)
                    img_resp.raise_for_status()
                    import io
                    images.append(Image.open(io.BytesIO(img_resp.content)).convert("RGB"))

            elapsed = time.time() - t0
            logger.info(f"  SiliconFlow 成功: {len(images)} 张, {elapsed:.1f}s")

            return GenerationResult(
                images=images,
                provider=self.name,
                raw_resolution=(params.width, params.height),
                seed_used=data.get("seed", params.seed),
                metadata={"model": self.model, "elapsed_s": round(elapsed, 1)},
            )

        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("message", "")[:200]
            except Exception:
                detail = str(e)[:200]
            logger.error(f"SiliconFlow 生成失败: {detail}")
            raise RuntimeError(f"SiliconFlow API 错误: {detail}")
        except Exception as e:
            logger.error(f"SiliconFlow 生成失败: {e}")
            raise
