# walan-design

AI 花型设计自动售卖流水线 — 从趋势采集到瓦栏上传全自动化。

> 基于瓦栏网官方《花型设计作品上传标准》PDF 的合规实现。

---

## ✅ 当前完成度

| 模块 | 状态 | 说明 |
|------|------|------|
| 趋势采集 | 🟡 部分可用 | 本地模板 Brief 正常工作；Pinterest/Behance 爬虫因网络限制暂不可用；LLM 分析（OpenAI）可用 |
| AI 生图 | 🟡 可用 | Stable Diffusion WebUI API 已接入（需本地启动）；SD 不可用时自动回退占位图 |
| **一花四色** | ✅ 完成 | HSV 变换自动生成 2 深底 + 2 浅底共 4 个配色方案 |
| PSD 分层生成 | ✅ 完成 | 强制 Background + Pattern 两层，ResolutionInfo(DPI) 自动注入 |
| **质量检测** | ✅ 完成 | 分辨率、接回位（容差 0px）、色彩模式、尺寸比例 2:3、一花四色、标签数量 3-5 |
| CLI 命令行 | ✅ 完成 | Typer + Rich，子命令模式，config set 直接改配置 |
| 浏览器上传 | ⚪ 待集成 | Python 侧准备 upload_tasks.json + prompts.json；实际上传需 Trae browser_use agent 执行 |
| 前端 Dashboard | ⚪ 未开始 | Phase 2 再做 |

---

## 🚀 快速开始

### 环境要求

- **Python 3.11+**
- **uv**（比 pip 快 10-100 倍）
- **Stable Diffusion WebUI**（可选，本地部署用于真实 AI 生图；没装也能跑，会自动用占位图）

### 安装

```bash
# 1. 克隆项目
cd walan-design

# 2. 用 uv 安装依赖
uv sync

# 3. 查看配置（可选）
uv run walan-design config show
```

### 跑起来

```bash
# 最基本：完整流水线（用本地模板 Brief + 占位图，跑通全流程）
uv run walan-design run -n 1

# 只看趋势采集产出的设计 Brief
uv run walan-design collect -n 3

# 只做 AI 生图（需要本地 SD WebUI 运行在 127.0.0.1:7860）
uv run walan-design design

# 只把已有 PNG 转成 PSD 分层文件
uv run walan-design psd

# 只准备上传任务（生成 upload_tasks.json + prompts.json）
uv run walan-design upload

# 打印瓦栏官方标准（方便对照）
uv run python -m walan_design.rules
```

### 准备好真实 AI 生图

```bash
# 1. 启动本地 SD WebUI（启用 API）
./webui.sh --api --listen

# 2. 修改 config.yaml 里的 checkpoint（你有的模型文件）
uv run walan-design config set design.stable_diffusion.checkpoint "your_model.safetensors"

# 3. 跑真图
uv run walan-design run -n 3
```

---

## 📋 上传到瓦栏

Python 侧的流水线会在最后生成两个文件：

| 文件 | 位置 | 内容 |
|------|------|------|
| `upload_tasks_*.json` | `output/temp/` | PSD 路径、价格、标签 |
| `upload_prompts.json` | `output/temp/` | 给 browser_use agent 的自然语言指令 |

然后：
1. 确认你的 ego-lite 浏览器已登录瓦栏
2. 打开 Trae，让 agent 执行 `output/temp/upload_prompts.json` 里的指令
3. 审核通过后，转至公开区 / VIP 区即可售卖

---

## 🛠️ CLI 速查

```bash
# 完整流水线（支持分段）
walan-design run                          # 全流程
walan-design run -n 3                     # 生成 3 个
walan-design run -m collect_only          # 只采集
walan-design run -m design_only           # 只生图
walan-design run -m upload_only           # 只上传准备

# 单独子命令
walan-design collect                      # 趋势采集
walan-design design [briefs.json]         # AI 生图
walan-design psd [png_dir]                # PNG → PSD
walan-design upload [psd_dir]             # 上传准备

# 配置管理
walan-design config show                  # 显示全部
walan-design config show walan.price_tier # 显示某个值
walan-design config set walan.price_tier custom  # 修改
walan-design config set design.stable_diffusion.width 2048
```

---

## 📂 项目结构

```
walan-design/
├── pyproject.toml              # 项目配置 + 依赖（PEP 621）
├── config.yaml                 # 所有参数配置（改这里就行）
├── .gitignore
├── src/
│   └── walan_design/           # 真正的 Python 包
│       ├── __init__.py
│       ├── rules.py            # 瓦栏 PDF 标准 → 可编程常量
│       ├── trend_collector.py  # ① 趋势采集 + Brief 生成
│       ├── ai_designer.py      # ② AI 生图 + 接回位 + 一花四色
│       ├── psd_generator.py    # ③ PSD 分层文件 + DPI 注入
│       ├── quality_checker.py  # ④ 质量检测
│       ├── walan_uploader.py   # ⑤ 上传准备（生成 browser_use prompt）
│       ├── pipeline.py         # 主流程编排
│       └── cli.py              # Typer CLI 入口
├── tests/
│   └── test_pipeline.py
└── output/                     # .gitignore 排除，运行时生成
    ├── trends/                 # Brief JSON
    ├── designs/                # PNG 图片（原图 + 4 配色变体）
    ├── psd/                    # PSD 分层文件
    └── temp/                   # 上传任务文件
```

---

## 📐 瓦栏硬标准（已全部对齐）

| # | 标准 | 代码位置 |
|---|------|---------|
| ① | 尺寸 40×60cm (2:3) | `config.yaml` → SD 2048×3072，`quality_checker.check_dimensions()` |
| ② | PSD 分层稿 | `psd_generator._make_psd_with_dpi()` 强制 Background + Pattern 两层 |
| ③ | DPI > 300 | `rules.DPI_MIN`，`quality_checker.check_resolution()` |
| ④ | 接回位容差 = 0px | `config.yaml` → `seamless_tolerance: 0` |
| ⑤ | 一花四色 (2深+2浅) | `ai_designer.generate_color_variants()` |
| ⑥ | 原创性 > 30% | `rules.ORIGINALITY_MODIFICATION_THRESHOLD`，黑名单位入 `rules.py` |
| ⑦ | 标签 3-5 个 | `config.yaml` → `tag_min/max`，`quality_checker.check_tag_count()` |
| ⑧ | PSD 分层 ≥ 2 层 | `rules.LAYER_COUNT_MIN`，`psd_generator` 强制分层 |

跑 `uv run python -m walan_design.rules` 打印完整摘要。

---

## 🔧 常用配置

编辑 `config.yaml`：

```yaml
# 改价格档位
walan.price_tier: "tier_399"        # 买断399 下载199
# 或自定义
walan.custom_price: { buyout: 499, download: 249, psd: 50 }

# 改批量
pipeline.batch_count: 10

# 改 SD 模型
design.stable_diffusion.checkpoint: "your_model.safetensors"

# 改一花四色数量
design.color_variant_count: 4

# 改 PSD 分层策略（background_main / multi / single）
psd.layer_strategy: "multi"

# 改接回位容差（瓦栏标准是 0，想松一点可以设 1）
psd.seamless_tolerance: 0
```

或者用 CLI：
```bash
walan-design config set walan.price_tier tier_399
walan-design config set pipeline.batch_count 10
```

---

## 🧪 开发

```bash
# 安装开发依赖
uv sync --extra dev

# Lint + Format
uv run ruff check src/walan_design/
uv run ruff format src/walan_design/

# 跑测试
uv run pytest -v
```

---

## 📅 路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | Python 流水线跑通（采集 → 生图 → PSD → 检测 → 上传准备） | ✅ 80% |
| Phase 2 | 真实瓦栏上传（browser_use agent 执行） | 🔜 下一步 |
| Phase 3 | 真实趋势采集（爬虫 + LLM 自动分析） | 🔜 |
| Phase 4 | React Dashboard | ⚪ |
| Phase 5 | 销量反馈闭环（自动分析什么好卖） | ⚪ |

---

## 📜 参考

- [瓦栏网花型设计作品上传标准 PDF](https://www.walanwalan.com/wldata/open/articles/2017-04/f4dd720c-24a1-11e7-9c88-00163e001713.pdf)
- [Photocraft / ArtCraft](https://github.com/storytold/photocraft) — 纯 Rust 的 Photoshop 重实现（后期参考）
