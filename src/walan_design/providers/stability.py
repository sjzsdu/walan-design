"""
Stability AI API 供应商 — 使用最新 v2beta 接口

可用模型:
  - Stable Image Ultra    (SD 3.5, 最高质量, 8 credits/张)
  - Stable Image Core     (SDXL 下一代, 快速便宜, 4 credits/张)
  - SDXL v1               (经典, 有 tile-texture 预设, 2 credits/张)

能力上限: 所有模型直出最大 1048576 像素 (1 MP)
  → 必须超分辨率放大到瓦栏目标 (33.5 MP @300DPI)

新版 v2beta 特性:
  - aspect_ratio 参数: 16:9, 1:1, 3:4, 4:3, 5:4, 21:9
  - 支持 negative_prompt
  - 有 upscale API (conservative / fast) 可直接放大到目标尺寸
  - SDXL v1 有 style_preset: tile-texture（专门针对无缝图案）
"""

import io
import logging
import os

import requests
from PIL import Image

from .base import GenerationParams, GenerationResult, ImageProvider

logger = logging.getLogger(__name__)


# 瓦栏目标比例 2:3 → Stability aspect_ratio = "3:4"（竖版）
# 但 SDXL v1 有固定分辨率列表，其中 2:3 最接近的是 832×1216
SDXL_FIXED_RESOLUTIONS = [
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
    (640, 1536),
    (768, 1344),
    (832, 1216),
    (896, 1152),
]


def _select_2_3_resolution() -> tuple[int, int]:
    """从 SDXL 固定分辨率中选最接近 2:3 (竖版) 的"""
    best = None
    best_diff = float("inf")
    for w, h in SDXL_FIXED_RESOLUTIONS:
        if h > w:  # 竖版
            ratio = w / h
            diff = abs(ratio - 2 / 3)
            if diff < best_diff:
                best_diff = diff
                best = (w, h)
    return best or (832, 1216)


class StabilityProvider(ImageProvider):
    """Stability AI 云端 API（v2beta）"""

    name = "stability"
    max_resolution = (1024, 1536)  # Core/Ultra: 1024px 短边
    supports_tiling = False  # v2beta 去掉了原生 tiling
    supports_2_3_ratio = True  # 通过 aspect_ratio 或 SDXL 固定分辨率

    API_BASE = "https://api.stability.ai/v2beta"

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("stability", {}).get("api_key", "") or os.environ.get("STABILITY_API_KEY", "")
        # 默认用 Core（性价比最高），可设 ultra / core / sdxl-v1
        self.model = config.get("stability", {}).get("model", "core")

    def health_check(self) -> bool:
        """简化版：有 key 就认为可用，真正连通性在 generate 时验证"""
        if not self.api_key:
            return False
        # 异步尝试查余额，但不阻塞——key 存在即返回 True
        try:
            import threading

            def _check():
                try:
                    resp = requests.get(
                        f"{self.API_BASE}/user/balance",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        credits = resp.json().get("credits", "?")
                        logger.info(f"  Stability 余额: {credits} credits")
                    elif resp.status_code == 404:
                        logger.info("  Stability API key 有效（余额端点 404 但 key 没问题）")
                except Exception:
                    pass

            threading.Thread(target=_check, daemon=True).start()
        except Exception:
            pass
        return True

    def generate(self, params: GenerationParams) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("Stability API key 未设置（STABILITY_API_KEY 环境变量或 config）")

        # 构造 prompt
        prompt_text = params.prompt
        if params.tiling:
            # 强调无缝平铺（Core/Ultra 无原生 tiling，靠 prompt 引导）
            prompt_text += ", seamless tileable repeating pattern, edge-to-edge continuous"

        # 根据模型选 API endpoint 和参数
        if self.model == "sdxl-v1":
            endpoint = f"{self.API_BASE}/stable-diffusion-xl-1024-v1-0/text-to-image"
            # SDXL 用固定分辨率列表，选最接近 2:3 的
            w, h = _select_2_3_resolution()
            form_data = {
                "prompt": prompt_text,
                "negative_prompt": params.negative_prompt,
                "width": str(w),
                "height": str(h),
                "samples": str(params.batch_size),
                "steps": str(params.steps),
                "cfg_scale": str(params.cfg_scale),
                "style_preset": "tile-texture",  # 专门针对无缝图案！
            }
        elif self.model == "ultra":
            endpoint = f"{self.API_BASE}/stable-image/generate/ultra"
            form_data = {
                "prompt": prompt_text,
                "negative_prompt": params.negative_prompt,
                "aspect_ratio": "2:3",  # 瓦栏标准 竖版
                "output_format": "png",
            }
        else:  # core
            endpoint = f"{self.API_BASE}/stable-image/generate/core"
            form_data = {
                "prompt": prompt_text,
                "negative_prompt": params.negative_prompt,
                "aspect_ratio": "2:3",
                "output_format": "png",
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*",  # 直接返回图片（不是 base64）
        }

        logger.info(f"  Stability {self.model}: {prompt_text[:60]}...")

        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                data=form_data,
                files={"none": ""},
                timeout=180,
            )

            content_type = resp.headers.get("content-type", "")

            if resp.status_code != 200:
                # 可能是 JSON 错误
                try:
                    err = resp.json()
                    msg = err.get("name", "") + ": " + err.get("errors", [""])[0]
                except Exception:
                    msg = resp.text[:200]
                raise RuntimeError(f"Stability API {resp.status_code}: {msg}")

            # Core/Ultra 返回单张 image/png
            images = []
            if "image" in content_type:
                img = Image.open(io.BytesIO(resp.content))
                images.append(img)
            elif "multipart" in content_type:
                for part in resp.iter_parts():
                    if part.content_type.startswith("image/"):
                        img = Image.open(io.BytesIO(part.content))
                        images.append(img)

            if not images:
                raise RuntimeError("Stability API 未返回有效图片")

            logger.info(f"  Stability 成功: {len(images)} 张, 原始 {images[0].size}")

            return GenerationResult(
                images=images,
                provider=f"stability-{self.model}",
                raw_resolution=images[0].size if images else (0, 0),
                metadata={"model": self.model},
            )

        except requests.exceptions.ConnectionError:
            raise RuntimeError("Stability API 连接失败")

    def upscale_to_target(self, img: Image.Image, target_width: int) -> Image.Image:
        """
        调用 Stability 的 upscale API 直接放大到目标宽度。
        比本地 LANCZOS 质量好很多，接近 ESRGAN 水平。
        """
        if not self.api_key:
            logger.warning("Stability API key 不可用，跳过云端放大")
            return img

        endpoint = f"{self.API_BASE}/stable-image/upscale/conservative"

        # 把 PIL Image 存到内存
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        form_data = {"width": str(target_width), "output_format": "png"}
        files = {"image": ("input.png", buf, "image/png")}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*",
        }

        try:
            resp = requests.post(endpoint, headers=headers, data=form_data, files=files, timeout=300)
            if resp.status_code == 200 and "image" in resp.headers.get("content-type", ""):
                upscaled = Image.open(io.BytesIO(resp.content))
                logger.info(f"  Stability upscale: {img.size} → {upscaled.size}")
                return upscaled
            else:
                logger.warning(f"Stability upscale 失败 ({resp.status_code})，回退本地放大")
                return img
        except Exception as e:
            logger.warning(f"Stability upscale 异常 ({e})，回退本地放大")
            return img
