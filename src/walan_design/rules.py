"""
瓦栏网 · 花型设计作品上传标准
来源: https://www.walanwalan.com/wldata/open/articles/2017-04/f4dd720c-24a1-11e7-9c88-00163e001713.pdf

所有规则均来自 PDF 原文，是审核的硬约束，必须严格遵守。
"""

# =====================================================
# 1. 尺寸标准（按应用领域）
# =====================================================

# 单位：厘米 (cm)
SIZE_STANDARDS = {
    # 服装面料（女装/男装/童装/泳装等）
    "digital_square": {  # 四方连续（数码）
        "w_cm": 40,
        "h_cm": 60,
        "note": "建议 40cm×60cm，比例 2:3",
        "ratio": (2, 3),
    },
    "traditional_square": {  # 四方连续（传统）
        "w_cm": 40,
        "h_cm": 64,
        "note": "高度必须是 32cm 的倍数（工艺限制）",
        "ratio": (5, 8),  # 40:64
    },
    "two_way_position": {  # 定位二方连续
        "options": [
            (50, 90),  # 通用
            (75, 110),  # 真丝面料
            (100, 150),  # 大码女装定位花
        ],
        "note": "水平方向可连续",
    },
    "single_place": {  # 独幅（胸前片定位）
        "w_cm": 60,
        "h_cm": 80,
        "note": "独幅，不需要连续",
    },
    # 家纺
    "home_textile": {  # 沙发布/桌布/墙纸
        "w_cm": 55,
        "h_cm": 180,
        "note": "多为四方连续，参考服装四方连续尺寸",
    },
    "umbrella": {  # 雨伞花型
        "length_cm": 64,
        "note": "经向 64cm 或倍数，宽度随意但必须可连续",
    },
}


# 像素计算: cm → px = cm × DPI / 2.54
# 40cm × 60cm @ 300DPI ≈ 4724 × 7087 px
# 40cm × 60cm @ 200DPI ≈ 3150 × 4724 px
def cm_to_px(cm: float, dpi: int) -> int:
    """厘米转像素"""
    return round(cm * dpi / 2.54)


def get_default_pixel_size(dpi: int = 300, style: str = "digital_square") -> tuple[int, int]:
    """
    获取默认像素尺寸。
    style: digital_square / traditional_square
    """
    s = SIZE_STANDARDS.get(style, SIZE_STANDARDS["digital_square"])
    return (cm_to_px(s["w_cm"], dpi), cm_to_px(s["h_cm"], dpi))


# =====================================================
# 2. 格式标准
# =====================================================
ALLOWED_FORMATS = ["PSD", "TIF", "JPG"]
REQUIRED_FORMAT_FOR_STUDIO = "PSD"  # 签约工作室必须上传 PSD

# =====================================================
# 3. 清晰度标准
# =====================================================
DPI_MIN = {
    "traditional": 300,  # 传统花型分辨率 > 300
    "digital": 200,  # 数码花型分辨率 > 200
}

# =====================================================
# 4. 接回位标准
# =====================================================
SEAMLESS_TOLERANCE = 0  # 相差 1 像素都不行！PDF 明确要求

# =====================================================
# 5. 配色标准：一花四色
# =====================================================
COLOR_VARIANT_COUNT = 4  # 必须 4 个配色方案
COLOR_VARIANT_MIX = (2, 2)  # 2 深底 + 2 浅底
COLOR_HUE_DIFFERENCE = "明显"  # 四个配色要有明显色相区别
SEASONAL_CONSIDERED = True  # 需考虑春夏/秋冬季节色彩

# =====================================================
# 6. 原创性标准
# =====================================================
ORIGINALITY_MODIFICATION_THRESHOLD = 0.30  # 改动幅度 < 30% 算抄袭
ORIGINALITY_BLACKLIST: list[str] = [
    "盗用",
    "模仿",
    "抄袭他人作品",
    "非原创素材修改幅度 < 30%",
    "WGSN",
    "蝶讯",
    "POP素材",  # PDF 里点名的素材源
    "已分色且经市场流通作品",
    "品牌 logo/品牌名",
    "侵犯个人肖像权/电影人物形象",
    "未授权知名卡通形象",
    "侵犯摄影/美术作品版权",
]

# =====================================================
# 7. 标签标准
# =====================================================
TAG_COUNT_MIN = 3
TAG_COUNT_MAX = 5
TAG_RULE = "必须根据花型实际元素/风格正确填写，不允许不相关标签"

# =====================================================
# 8. PSD 分层标准
# =====================================================
LAYER_RULES = [
    "底部 (Background) 与 主花 (Main Flower) 必须分开各一个图层",
    "次花 (Secondary Flower) 单独一个图层",
    "数码照片风格素材，每个素材必须单独分层",
    "如果分层对客户改动没作用，可选择不提供分层稿（但仍建议分层）",
]
LAYER_COUNT_MIN = 2  # 背景层 + 主花层 = 最少 2 层


# =====================================================
# 汇总：所有标准的规则总览
# =====================================================
def print_summary():
    """打印全部规则摘要（方便 CLI 展示）"""
    lines = [
        "瓦栏网花型设计作品上传标准 · 硬约束",
        "=" * 45,
        "① 尺寸: 数码 40×60cm / 传统 40×64cm",
        f"② 格式: {'/'.join(ALLOWED_FORMATS)}（签约必须 PSD）",
        f"③ 清晰度: 传统 > {DPI_MIN['traditional']}DPI / 数码 > {DPI_MIN['digital']}DPI",
        f"④ 接回位: 四方连续水平+垂直无缝 (容差={SEAMLESS_TOLERANCE}px)",
        f"⑤ 配色: 必须 {COLOR_VARIANT_COUNT} 个方案 ({COLOR_VARIANT_MIX[0]}深+{COLOR_VARIANT_MIX[1]}浅)，色相区别明显",
        f"⑥ 原创性: 改动幅度 < {ORIGINALITY_MODIFICATION_THRESHOLD * 100:.0f}% 算抄袭，禁止品牌/IP/知名素材",
        f"⑦ 标签: {TAG_COUNT_MIN}-{TAG_COUNT_MAX} 个，必须与花型相关",
        f"⑧ PSD 分层: 至少 {LAYER_COUNT_MIN} 层（背景+主花）",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(print_summary())
