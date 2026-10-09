"""
供应商基类 — 所有 AI 生图供应商实现此接口。
统一接口设计：输入 prompt + 参数 → 输出 PIL Image 列表。
"""

import logging
from dataclasses import dataclass, field

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    """生图参数（供应商无关）"""

    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    seed: int = -1  # -1 = 随机
    steps: int = 30
    cfg_scale: float = 7.5
    # 无缝平铺：部分供应商原生支持
    tiling: bool = True


@dataclass
class GenerationResult:
    """生图结果"""

    images: list[Image.Image] = field(default_factory=list)
    provider: str = ""
    raw_resolution: tuple[int, int] = (0, 0)  # 原始直出分辨率
    seed_used: int = -1
    metadata: dict = field(default_factory=dict)


class ImageProvider:
    """供应商基类 — 子类必须实现 generate()"""

    name: str = "base"
    max_resolution: tuple[int, int] = (1024, 1024)
    supports_tiling: bool = False
    supports_2_3_ratio: bool = False

    def __init__(self, config: dict):
        self.config = config

    def generate(self, params: GenerationParams) -> GenerationResult:
        """生成图片 — 子类必须实现"""
        raise NotImplementedError

    def health_check(self) -> bool:
        """检查供应商是否可用（API key、服务在线等）"""
        return False

    def get_effective_resolution(self, params: GenerationParams) -> tuple[int, int]:
        """
        返回实际生成分辨率。
        如果供应商不支持自定义分辨率，返回其最大分辨率。
        """
        return self.max_resolution

    @staticmethod
    def create(provider_name: str, config: dict) -> "ImageProvider":
        """工厂方法 — 根据名称创建供应商"""
        # 延迟导入避免循环依赖
        from .dalle import DalleProvider
        from .sd_webui import SDWebUIProvider
        from .stability import StabilityProvider

        providers = {
            "stable_diffusion": SDWebUIProvider,
            "stability": StabilityProvider,
            "dalle": DalleProvider,
        }
        cls = providers.get(provider_name)
        if cls is None:
            raise ValueError(f"未知供应商: {provider_name}，可选: {', '.join(providers.keys())}")
        return cls(config)
