"""
DALL-E 3 API 供应商
文档: https://platform.openai.com/docs/api-reference/images

优势:
  - 接入极简（用户已有 OpenAI API key）
  - 图像质量好，提示词理解强
  - 1792×1024 支持 3:2 比例

劣势:
  - 不支持 tiling（无无缝平铺）
  - 不支持 negative_prompt
  - 每次只能生成 1 张
  - 分辨率偏低（~1.8 MP），需放大
  - 费用较高 $0.04-0.12/张
"""

import logging
import os

from PIL import Image

from .base import GenerationParams, GenerationResult, ImageProvider

logger = logging.getLogger(__name__)


class DalleProvider(ImageProvider):
    """DALL-E 3 API"""

    name = "dalle"
    max_resolution = (1792, 1024)  # DALL-E 3 最大
    supports_tiling = False
    supports_2_3_ratio = True  # 1792×1024 ≈ 3:2，接近 2:3

    def __init__(self, config: dict):
        super().__init__(config)
        from openai import OpenAI

        api_key = config.get("dalle", {}).get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        base_url = config.get("dalle", {}).get("base_url", None)
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.model = config.get("dalle", {}).get("model", "dall-e-3")

    def health_check(self) -> bool:
        if self.client is None:
            return False
        try:
            # 检查余额或连接（不消耗配额）
            self.client.models.list()
            return True
        except Exception:
            return False

    def generate(self, params: GenerationParams) -> GenerationResult:
        if self.client is None:
            raise RuntimeError("OpenAI API key 未设置")

        # DALL-E 3 只支持 quality=standard/hd，不支持自定义分辨率
        # 用 size 参数选最接近 2:3 的
        w, h = params.width, params.height
        if w > h:
            size = "1792x1024"  # 横版 ~3:2
        else:
            size = "1024x1792"  # 竖版 ~2:3

        # DALL-E 3 不支持 negative_prompt，拼到 prompt 里
        prompt = params.prompt
        if params.negative_prompt:
            prompt += f"\n\nAvoid: {params.negative_prompt}"

        # 强调无缝平铺和纺织品特性
        if params.tiling:
            prompt += "\n\nSeamless tileable pattern, repeating pattern for fabric textile design"

        logger.info(f"  DALL-E 3: {prompt[:60]}... (size={size})")

        images = []
        for i in range(params.batch_size):
            try:
                resp = self.client.images.generate(
                    model=self.model,
                    prompt=prompt,
                    size=size,
                    quality="hd",
                    n=1,  # DALL-E 3 每次只能 1 张
                )
                import base64
                import io

                for data in resp.data:
                    # 新版 API 返回 URL 或 b64_json
                    if hasattr(data, "b64_json") and data.b64_json:
                        img = Image.open(io.BytesIO(base64.b64decode(data.b64_json)))
                    elif hasattr(data, "url") and data.url:
                        import requests as _req

                        img_resp = _req.get(data.url, timeout=60)
                        img = Image.open(io.BytesIO(img_resp.content))
                    else:
                        continue
                    images.append(img)
                    logger.info(f"  DALL-E 3 成功: 第 {i + 1} 张")
            except Exception as e:
                logger.error(f"DALL-E 3 生成失败: {e}")
                # 继续尝试下一张

        return GenerationResult(
            images=images,
            provider=self.name,
            raw_resolution=images[0].size if images else (0, 0),
            metadata={"model": self.model, "size": size},
        )
