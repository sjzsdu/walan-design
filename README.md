# walan-design

AI 花型设计自动售卖流水线 — 从流行趋势采集到瓦栏上架售卖的全自动化系统。

> 全流程已端到端实测跑通：采集真实趋势 → AI 生成花型 → 生成合规 PSD → 质量检测 → 自动上传瓦栏并发布到公开区（待审核）。

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ ① 趋势采集 │ → │ ② AI 生图 │ → │ ③ PSD 生成│ → │ ④ 质量检测 │ → │ ⑤ 上传发布 │
│ 瓦栏+POP  │   │ SDXL+自评 │   │ 分层+DPI  │   │ 8 项检查  │   │ ego浏览器 │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## 🧠 实现原理

整个系统是一条五段流水线，每一段都可以单独运行。核心思想：**用 AI 判断流行趋势、用 AI 生成设计、再用 AI 审核质量**，人只在最后确认发布。

### ① 趋势采集（trend_collector + browser_trends）

流行方向不靠拍脑袋，而是"看真实市场上正在卖什么"：

- 通过 **ego-browser**（复用已登录的真实浏览器）抓取两个数据源：
  - **瓦栏公开区** — 平台真实在售花型 + 价格，最强的商业趋势信号
  - **POP 服装趋势网** — 中文服装趋势资讯标题
- 下载 12 张在售花型图，交给**视觉 LLM**（OpenRouter 免费视觉模型）"看图"分析配色、母题、排布风格
- 结合趋势标题，生成结构化的**设计 Brief**（主题、母题形态、排布、密度、笔触、印花工艺、标签建议）

> Pinterest / Behance 因国内网络限制不可用，已在配置中禁用；数据源可在 `config.yaml` 切换。

### ② AI 花型设计（ai_designer + prompt_engine + visual_critic + upscaler + recolor）

这一段是质量的核心，由四个模块协作：

1. **Prompt 工程（prompt_engine）**：把 Brief 升级为专业纺织生图 prompt = 主体描述 + 构图约束（均衡四方连续排布）+ 印刷媒介提示（去照片感/阴影/透视）+ 质量增强词，负向词自动合并去重。全部可在配置中自定义。
2. **多供应商生图**：支持 Stability AI 云端 SDXL（推荐，~$0.03/张，原生无缝平铺）、DALL-E 3、本地 SD WebUI 三种引擎，配置一键切换。
3. **AI 视觉自评闭环（visual_critic）**：生成后由视觉模型扮演"花型设计总监"，从构图、细节、配色、商业潜力、平面印花感五个维度打分（0-100）。低于阈值（默认 70 分）的图会**带着改进建议自动重新生成**；评分服务不可用时 fail-open 放行，不阻塞流水线。
4. **超分辨率放大（upscaler）**：AI 直出 ~1 MP，远低于瓦栏要求的 ~33 MP。放大到 4724×7087 px（40×60cm @ 300DPI），支持 Real-ESRGAN / LANCZOS 两种算法。随后处理**四方连续接回位**（水平 + 垂直无缝）。
5. **一花四色（recolor）**：瓦栏强制"一花四色（2深+2浅）"。原理：median-cut 把图量化为 8 个主色簇 → 专业调色板按明度排序作为"明度→颜色"映射坡道 → 每个簇按明度位置映射到目标色 → 像素级重映射时保留原始明暗纹理。内置 6 套专业配色（深底：墨蓝金/酒红夜曲/松林暮色；浅底：奶油乡村/雾霾粉彩/薄荷清晨）。

最终每个设计产出 5 张图：1 张主图 + 4 个配色变体。

### ③ PSD 分层生成（psd_generator）

瓦栏只收 PSD 分层文件且严格校验 DPI。这里有个关键坑：psd-tools 默认只写 ResolutionInfo(0x3ED) 资源，但部分读取方（包括瓦栏服务器）读不到会回退 72dpi 而拒收。解决方案：**手动 patch 同时注入 ResolutionInfo + XMP(0x0424) 双写 DPI**，并保证资源块 2 字节对齐。分层策略强制 Background + Pattern 两层，符合"分层 ≥ 2 层"的上传标准。

### ④ 质量检测（quality_checker）

对齐瓦栏《花型设计作品上传标准》PDF 的 8 项自动检查：DPI ≥ 300、接回位边缘差异 = 0（容差 0px）、RGB 色彩模式、画面无文字水印、尺寸比例 2:3、一花四色、标签 3-5 个、分层 ≥ 2。任何一项不过即拦截，不让不合格文件流到上传环节。

### ⑤ 上传发布（walan_uploader + ego-browser）

- Python 侧生成 `upload_tasks_*.json`（PSD 路径、标签、价格、目标区）和自然语言 prompt
- 通过 **ego-browser** 驱动已登录瓦栏的真实浏览器，自动完成三步：
  1. **上传**：提交 PSD，服务器校验 DPI（约 170MB 大文件约 2 分钟）
  2. **管理**：设置内容/类别/回位/是否含源文件 + 标签多选 → 保存
  3. **销售**：定价（买断+下载档位）→ 保存 → 转入公开区 → 状态变为"公开区(待审核)"

> 发布序列已实测固化（详见 `project_memory`），包括多个坑：发布前必须先选中作品卡片、select2 标签需用 jQuery 设值、销售表单需先保存价格再发布等。

---

## 📦 安装

### 环境要求

| 依赖 | 用途 | 必需？ |
|------|------|--------|
| **Python 3.11+** | 运行流水线 | ✅ |
| **uv** | Python 包管理（比 pip 快 10-100 倍） | ✅ |
| **OpenRouter API Key** | 趋势分析 LLM + 视觉自评（免费模型可用） | ✅ |
| **Stability AI API Key** | 云端 SDXL 生图（~$0.03/张） | 用 stability 引擎时必需 |
| **ego-browser** | 真实浏览器采集趋势 + 上传发布 | 趋势采集/上传时必需 |

### 安装步骤

```bash
# 1. 进入项目目录
cd walan-design

# 2. 安装依赖（uv 会自动创建虚拟环境）
uv sync

# 3. 验证安装
uv run walan-design --help
```

### 配置 API Key

API key 通过环境变量注入（config.yaml 中对应字段留空即自动读取）：

```bash
# 写入 shell 配置（~/.zshrc），二选一或都配
export OPENROUTER_API_KEY="sk-or-v1-xxxx"    # LLM 分析 + 视觉自评
export STABILITY_API_KEY="sk-xxxx"           # SDXL 生图

# 生效后运行
source ~/.zshrc
```

首次运行前确认 `ego-browser` 可用（浏览器已登录瓦栏账号）：

```bash
ego-browser --help
```

---

## 🚀 使用

### 一键全流程

```bash
# 生成 2 个花型并走完全部流程（趋势采集→生图→PSD→质检→上传准备）
# 实测约 6 分钟 / 2 个设计（含 10 个 PSD）
uv run walan-design run -n 2
```

运行结束后在 `output/` 查看：

| 目录 | 内容 |
|------|------|
| `output/trends/` | 设计 Brief JSON + 采集的趋势图 |
| `output/designs/` | PNG 成品（主图 + 4 配色变体，4724×7087） |
| `output/psd/` | PSD 分层文件（300DPI，可直接上传） |
| `output/temp/` | `upload_tasks_*.json` 上传任务清单 |

### 只跑某一段

```bash
# 只采集趋势、生成设计 Brief（看看 AI 打算设计什么）
uv run walan-design collect -n 3

# 只做 AI 生图（用已有的 Brief 文件）
uv run walan-design design output/trends/briefs_xxx.json

# 只把 PNG 转成 PSD
uv run walan-design psd [png目录]

# 只生成上传任务
uv run walan-design upload [psd目录]

# run 的分段模式（等价写法）
uv run walan-design run -m collect_only
uv run walan-design run -m design_only
uv run walan-design run -m upload_only
```

### 上传发布到瓦栏

流水线跑完后（或直接用 `upload` 子命令生成任务），用 ego-browser 执行发布。打开 Trae 对 AI 说：

> 帮我把 output/temp/ 里最新的 upload_tasks 文件中的 PSD 发布到瓦栏

AI 会按已验证的序列自动执行：**上传 PSD → 填管理表单（分类/标签）保存 → 销售定价保存 → 转入公开区**。成功标志是页面出现"作品ID进入公开区(待审核)"，之后等瓦栏人工审核通过即可售卖。

### 配置管理

所有参数集中在 `config.yaml`，支持命令行直接修改：

```bash
# 查看全部 / 某项配置
uv run walan-design config show
uv run walan-design config show walan.price_tier

# 修改（自动写回 config.yaml）
uv run walan-design config set walan.price_tier tier_399
uv run walan-design config set pipeline.batch_count 10
uv run walan-design config set design.visual_review.threshold 75
```

常用配置项：

```yaml
walan:
  price_tier: "tier_299"          # 价格档位：tier_299(买断299/下载99) / tier_399 / custom
  auto_publish_zone: "public"     # 发布目标：public 公开区 / vip

trend:
  brief_count: 5                  # 每次采集生成几个设计 Brief

design:
  engine: "stability"             # 生图引擎：stability / dalle / stable_diffusion
  visual_review:
    threshold: 70                 # 自评及格线（低于则带建议重生成）
    max_retries: 2
  recolor:
    cluster_count: 8              # 主色聚类数；palettes 可自定义配色方案

upscaler:
  method: "lanczos"               # 放大算法：auto / esrgan / lanczos

pipeline:
  batch_count: 5                  # 默认批量
```

### 开发

```bash
uv sync --extra dev          # 安装开发依赖
uv run ruff check src/       # Lint
uv run pytest -v             # 测试
uv run python -m walan_design.rules   # 打印瓦栏标准摘要
```

---

## 📂 项目结构

```
walan-design/
├── pyproject.toml              # 项目配置 + 依赖（PEP 621）
├── config.yaml                 # 全部参数配置（改这里就行）
├── src/walan_design/
│   ├── rules.py                # 瓦栏 PDF 标准 → 可编程常量
│   ├── trend_collector.py      # ① 趋势分析 + Brief 生成（视觉 LLM 看图模式）
│   ├── browser_trends.py       # ① ego-browser 真实浏览器采集（瓦栏公开区/POP）
│   ├── prompt_engine.py        # ② Prompt 工程（纺织专业术语增强）
│   ├── ai_designer.py          # ② 生图编排 + 接回位 + 一花四色调度
│   ├── visual_critic.py        # ② AI 视觉自评闭环（五维打分/低分重生成）
│   ├── upscaler.py             # ② 超分辨率放大（→ 4724×7087 @300DPI）
│   ├── recolor.py              # ② 一花四色（调色板驱动重着色）
│   ├── psd_generator.py        # ③ PSD 分层 + DPI 双写注入
│   ├── quality_checker.py      # ④ 质量检测（8 项）
│   ├── walan_uploader.py       # ⑤ 上传任务生成
│   ├── pipeline.py             # 主流程编排
│   ├── cli.py                  # Typer CLI 入口
│   └── providers/              # 生图供应商适配（stability / dalle / 本地 SD）
├── tests/
└── output/                     # 运行时生成（gitignore）
    ├── trends/                 # Brief JSON + 趋势图
    ├── designs/                # PNG 成品
    ├── psd/                    # PSD 分层文件
    └── temp/                   # upload_tasks_*.json
```

---

## 📐 瓦栏硬标准（已全部对齐）

| # | 标准 | 实现位置 |
|---|------|---------|
| ① | 尺寸 40×60cm (2:3) | `upscaler.get_target_resolution()` → 4724×7087 @300DPI |
| ② | PSD 分层稿 | `psd_generator` 强制 Background + Pattern 两层 |
| ③ | 传统花型 DPI > 300 | ResolutionInfo + XMP 双写注入，`quality_checker.check_resolution()` |
| ④ | 接回位容差 = 0px | `ai_designer` 接回位处理 + `check_seamless()` |
| ⑤ | 一花四色 (2深+2浅) | `recolor.generate_color_variants()` |
| ⑥ | 标签 3-5 个 | `config.yaml tag_min/max` + `check_tag_count()` |
| ⑦ | 色彩模式 RGB | `check_color_mode()` |
| ⑧ | 画面无文字/水印/色标 | `check_no_text()` |

跑 `uv run python -m walan_design.rules` 打印完整标准摘要。

---

## 📅 路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | Python 流水线（采集 → 生图 → PSD → 检测 → 上传准备） | ✅ 完成 |
| Phase 2 | 质量提升（Prompt 工程 / 视觉自评 / 专业一花四色 / 真实趋势） | ✅ 完成 |
| Phase 3 | 真实上传发布（ego-browser 全自动，已发布作品待审核） | ✅ 完成 |
| Phase 4 | 批量产出验证与调优（多轮跑量、分析过审率） | 🔜 当前 |
| Phase 5 | 销量反馈闭环（自动分析什么好卖，反哺趋势采集） | ⚪ |
| Phase 6 | React Dashboard 可视化面板 | ⚪ |

---

## 📜 参考

- [瓦栏网花型设计作品上传标准 PDF](https://www.walanwalan.com/wldata/open/articles/2017-04/f4dd720c-24a1-11e7-9c88-00163e001713.pdf)
- [Photocraft / ArtCraft](https://github.com/storytold/photocraft) — 纯 Rust 的 Photoshop 重实现（后期参考）
