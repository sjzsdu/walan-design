# walan-design

AI 花型设计自动售卖流水线 — 从流行趋势采集到瓦栏上架售卖的全自动化系统。

> 全流程已端到端实测跑通：采集真实趋势 → AI 生成花型（本地 SDXL + 构造性四方连续）→ 生成合规 PSD → 质量检测 → 自动上传瓦栏并发布到公开区（待审核）。
> 已产出 7 个花型、5 个成功发布上架。

```
┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ ① 趋势采集 │ → │ ② AI 生图      │ → │ ③ PSD 生成│ → │ ④ 质量检测 │ → │ ⑤ 上传发布 │
│ 瓦栏+POP  │   │ SDXL+circular │   │ 分层+DPI  │   │ 8 项检查  │   │ ego浏览器 │
│ Brief 历史│   │ padding+自评  │   │           │   │ pHash去重 │   │          │
└──────────┘   └──────────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## 🧠 实现原理

整个系统是一条五段流水线，每一段都可以单独运行。核心思想：**用 AI 判断流行趋势、用 AI 生成设计、再用 AI 审核质量**，人只在最后确认发布。

### ① 趋势采集（trend_collector + browser_trends + brief_history）

流行方向不靠拍脑袋，而是"看真实市场上正在卖什么"：

- 通过 **ego-browser**（复用已登录的真实浏览器）抓取两个数据源：
  - **瓦栏公开区** — 平台真实在售花型 + 价格，最强的商业趋势信号（按日期翻页轮换，每天换不同页）
  - **POP 服装趋势网** — 中文服装趋势资讯标题
- 下载在售花型图（pHash 去重，重复图跳过），交给**视觉 LLM**（OpenRouter 免费视觉模型）"看图"分析配色、母题、排布风格
- 结合趋势标题，生成结构化的**设计 Brief**（主题、母题形态、排布、密度、笔触、印花工艺、标签建议）
- **Brief 历史库去重**：生成时注入最近 20 条已产出 Brief，让 LLM 主动避开重复主题/母题/配色组合（语义级去重，比事后 pHash 拦截更前置，不浪费 26 分钟生图时间）。历史库上限 50 条，可在 config 关闭。

> Pinterest / Behance 因国内网络限制不可用，已在配置中禁用；数据源可在 `config.yaml` 切换。

### ② AI 花型设计（ai_designer + prompt_engine + visual_critic + upscaler + recolor）

这一段是质量的核心，由五个模块协作：

1. **Prompt 工程（prompt_engine）**：把 Brief 升级为专业纺织生图 prompt = 主体描述 + 构图约束（均衡四方连续排布）+ 印刷媒介提示（去照片感/阴影/透视）+ 质量增强词，负向词自动合并去重。**数码花型净化器**：自动剔除含 `flat vector`/`line hatching`/`silkscreen`/`gouache` 等传统平涂词的段落，确保至少一个数码质感锚点（watercolor/gradient/digital painting）。全部可在配置中自定义。
2. **本地 SDXL 生图（diffusers + circular padding）**：当前主力引擎。核心机制：给 UNet **+ VAE** 的所有 Conv2d 打 `padding_mode='circular'` 补丁——让卷积在画布边缘环绕读取对侧像素，**生成即构造性四方连续**（母题绕边自然延续，无切花）。这跟 A1111 的 `payload["tiling"]=true` 是同一个机制，但 A1111 依赖的 Stability-AI/stablediffusion 仓库 2026 年已被删除（GitHub 404），已放弃。走常驻 HTTP 服务（`scripts/diffusers_server.py`，端口 7861），模型只加载一次（首次 ~25s），后续每次生图复用。
   > **关键教训**：只改 UNet（40 个 Conv2d）不够——UNet 在 latent 空间生成（latent 边缘连续），但 VAE 解码 latent→像素时用默认 zeros padding，把边缘连续性打断（实测比值 6.7 / 3.4，边缘断裂）。加上 VAE 的 55 个 Conv2d 后比值 0.52 / 0.66（完美连续）。
   >
   > 其他引擎仍可用：Stability AI 云端 SDXL（~$0.03/张）、DALL-E 3。
3. **AI 视觉自评闭环（visual_critic）**：生成后由视觉模型扮演"花型设计总监"，从构图、细节、配色、商业潜力、平面印花感五个维度打分（0-100）。低于阈值（默认 70 分）的图会**带着改进建议自动重新生成**；评分服务不可用时 fail-open 放行，不阻塞流水线。
4. **超分辨率放大（upscaler）**：AI 直出 ~1-2 MP，远低于瓦栏要求的 ~33 MP。放大到 4724×7087 px（40×60cm @ 300DPI），支持 Real-ESRGAN / LANCZOS 两种算法。随后处理**四方连续接回位**（水平 + 垂直无缝）。
5. **一花四色（recolor）**：瓦栏强制"一花四色（2深+2浅）"。原理：median-cut 把图量化为 8 个主色簇 → 专业调色板按明度排序作为"明度→颜色"映射坡道 → 每个簇按明度位置映射到目标色 → 像素级重映射时保留原始明暗纹理。内置 6 套专业配色（深底：墨蓝金/酒红夜曲/松林暮色；浅底：奶油乡村/雾霾粉彩/薄荷清晨）。

最终每个设计产出 5 张图：1 张主图 + 4 个配色变体。

### ③ PSD 分层生成（psd_generator）

瓦栏只收 PSD 分层文件且严格校验 DPI。这里有个关键坑：psd-tools 默认只写 ResolutionInfo(0x3ED) 资源，但部分读取方（包括瓦栏服务器）读不到会回退 72dpi 而拒收。解决方案：**手动 patch 同时注入 ResolutionInfo + XMP(0x0424) 双写 DPI**，并保证资源块 2 字节对齐。分层策略强制 Background + Pattern 两层，符合"分层 ≥ 2 层"的上传标准。

### ④ 质量检测（quality_checker + dedup_checker）

对齐瓦栏《花型设计作品上传标准》PDF 的 8 项自动检查：DPI ≥ 300、接回位边缘差异 = 0（容差 0px）、RGB 色彩模式、画面无文字水印、尺寸比例 2:3、一花四色、标签 3-5 个、分层 ≥ 2。

额外的 pHash 去重：批次内跨设计比对（阈值 10）+ 历史库比对（阈值 8，更严格），避免撞自己已发布的作品。历史库在上传准备好时自动写入（失败品不算）。

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
| **Python 3.10** | 本地 diffusers + torch 2.3.1（torch 无 py3.14 wheel） | ✅（本地生图时） |
| **uv** | Python 包管理（比 pip 快 10-100 倍） | ✅ |
| **Apple M1/M2 + 16GB RAM** | 本地 SDXL MPS 推理 | 用 diffusers_local 引擎时必需 |
| **SDXL base 1.0 safetensors** | 本地生图模型（6.94GB） | 用 diffusers_local 引擎时必需 |
| **OpenRouter API Key** | 趋势分析 LLM + 视觉自评（免费模型可用） | ✅ |
| **ego-browser** | 真实浏览器采集趋势 + 上传发布 | 趋势采集/上传时必需 |

### 安装步骤

```bash
# 1. 进入项目目录
cd walan-design

# 2. 安装依赖（uv 会自动创建虚拟环境）
uv sync

# 3. 本地 SDXL 生图引擎需要额外准备：
#    3a. 下载 SDXL base 1.0 模型（约 6.94GB）
#        放到 ~/.local/share/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors
#    3b. 在 A1111 遗留 venv 里装 diffusers（Python 3.10 + torch 2.3.1 已在里面）
#        /Users/juzhongsun/.local/share/stable-diffusion-webui/venv/bin/python -m pip install "diffusers==0.30.3" "transformers==4.44.2"
#    3c. 启动常驻 HTTP 服务（模型懒加载，首次 ~25s）
#        nohup /Users/juzhongsun/.local/share/stable-diffusion-webui/venv/bin/python \
#          scripts/diffusers_server.py > /tmp/diffusers_server.log 2>&1 &

# 4. 验证安装
uv run walan-design --help
curl -s http://127.0.0.1:7861/health    # 检查 diffusers server
```

### 配置 API Key

API key 通过环境变量注入（config.yaml 中对应字段留空即自动读取）：

```bash
# 写入 shell 配置（~/.zshrc）
export OPENROUTER_API_KEY="sk-or-v1-xxxx"    # LLM 分析 + 视觉自评
# STABILITY_API_KEY 留空即可（已切本地 diffusers）

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
# 生成 1 个数码花型并走完全部流程（趋势采集→生图→PSD→质检→上传）
# Mac M2 本地 SDXL 1024×1536/30步约 26 分钟/张
uv run walan-design run -n 1 -t digital

# 传统花型
uv run walan-design run -n 1 -t traditional
```

运行结束后在 `output/` 查看：

| 目录 | 内容 |
|------|------|
| `output/trends/` | 设计 Brief JSON + 采集的趋势图 |
| `output/runs/<花型名>/` | 每个花型一个子目录：brief.json + design_*.png + design_result.json + PSD |
| `output/psd/` | PSD 分层文件（300DPI，可直接上传） |
| `output/history/` | `phash_store.json`（像素指纹）+ `brief_store.json`（语义主题）|
| `output/temp/` | checkpoint + `upload_tasks_*.json` 上传任务清单 |

### 只跑某一段

```bash
# 只采集趋势、生成设计 Brief（看看 AI 打算设计什么，会自动避开历史主题）
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

`run -m full` 模式下 auto_upload=true 时自动执行上传+发布，无需手动干预。也可以只准备上传任务：

```bash
uv run walan-design run -m upload_only   # 生成 upload_tasks_*.json
```

然后打开 Trae 对 AI 说：

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
pipeline:
  mode: "full"                    # 运行模式
  auto_upload: true               # true=自动上传发布；false=只准备任务文件
  batch_count: 1                  # 默认批量

design:
  engine: "diffusers_local"       # 生图引擎：diffusers_local / stability / dalle
  brief_type: "digital"           # brief 类型：digital / traditional / mixed
  stable_diffusion:
    api_url: "http://127.0.0.1:7861"
    width: 1024
    height: 1536
    steps: 30
    timeout_s: 2400
  visual_review:
    enabled: true
    threshold: 70                 # 自评及格线（低于则带建议重生成）
    max_retries: 2

quality:
  dedup:
    brief_history_store: "output/history/brief_store.json"
    brief_history_size: 50       # Brief 历史库上限（条）；0=关闭

walan:
  price_tier: "tier_299"          # 价格档位：tier_299(买断299/下载99) / tier_399 / custom
  auto_publish_zone: "public"     # 发布目标：public 公开区 / vip
```

### 开发

```bash
uv sync --extra dev          # 安装开发依赖
uv run ruff check src/       # Lint
uv run pytest -v             # 测试
uv run python -m walan_design.rules   # 打印瓦栏标准摘要

# 诊断脚本
python scripts/diag_circular.py     # MPS circular padding + Conv2d 统计
python scripts/test_tiling_v2.py    # 快速验证四方连续（UNet+VAE）
python scripts/edge_diff.py         # 边缘连续性测量
python scripts/seed_brief_history.py # 首次导入历史 brief
```

---

## 📂 项目结构

```
walan-design/
├── pyproject.toml              # 项目配置 + 依赖（PEP 621）
├── config.yaml                 # 全部参数配置（改这里就行）
├── scripts/
│   ├── diffusers_server.py     # 常驻 HTTP 生图服务（端口 7861，UNet+VAE circular padding）
│   ├── test_diffusers_tiling.py # circular padding 完整验证脚本
│   ├── test_tiling_v2.py       # 512×768 快速验证（含边缘连续性测量）
│   ├── diag_circular.py        # MPS circular padding + Conv2d 数量统计
│   ├── edge_diff.py            # 边缘像素差异测量
│   ├── verify_tiling.py        # 周期窗口搜索 + 2×2 平铺目检
│   └── seed_brief_history.py   # 首次导入历史 brief 到 brief_store.json
├── src/walan_design/
│   ├── rules.py                # 瓦栏 PDF 标准 → 可编程常量
│   ├── trend_collector.py      # ① 趋势分析 + Brief 生成（视觉 LLM 看图模式 + 历史排除）
│   ├── browser_trends.py       # ① ego-browser 真实浏览器采集（瓦栏公开区/POP）+ pHash 去重
│   ├── brief_history.py        # ① Brief 历史库（语义级去重，比 pHash 更前置）
│   ├── prompt_engine.py        # ② Prompt 工程（纺织专业术语增强 + 数码花型净化器）
│   ├── ai_designer.py          # ② 生图编排 + circular padding 接回位 + 一花四色调度
│   ├── visual_critic.py        # ② AI 视觉自评闭环（五维打分/低分重生成）
│   ├── upscaler.py             # ② 超分辨率放大（→ 4724×7087 @300DPI）
│   ├── recolor.py              # ② 一花四色（调色板驱动重着色）
│   ├── seam_quilt.py           # ② 确定性四方连续接缝（quilt 算法）
│   ├── psd_generator.py        # ③ PSD 分层 + DPI 双写注入
│   ├── quality_checker.py      # ④ 质量检测（8 项）
│   ├── dedup_checker.py        # ④ pHash 去重（批次内 + 历史库）
│   ├── checkpoint.py           # 断点续跑管理
│   ├── walan_uploader.py       # ⑤ 上传任务生成
│   ├── upload_executor.py      # ⑤ ego-browser 真实上传执行
│   ├── pipeline.py             # 主流程编排
│   ├── cli.py                  # Typer CLI 入口
│   └── providers/              # 生图供应商适配
│       ├── base.py             # ImageProvider 基类 + GenerationParams/Result
│       ├── diffusers_local.py  # 本地 diffusers HTTP 客户端（主力）
│       ├── sd_webui.py         # A1111 WebUI（2026 仓库已删，已弃用）
│       ├── stability.py        # Stability AI 云端
│       └── dalle.py            # DALL-E 3
├── tests/
└── output/                     # 运行时生成（gitignore）
    ├── trends/                 # Brief JSON + 趋势图
    ├── runs/<花型名>/           # 每个花型独立目录（brief + PNG + PSD + 上传记录）
    ├── psd/                    # PSD 分层文件
    ├── history/                # phash_store.json + brief_store.json
    └── temp/                   # checkpoint + upload_tasks_*.json
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

## 🔑 关键设计决策（踩过的坑）

| 决策 | 原因 |
|------|------|
| **放弃 A1111，改 diffusers circular padding** | A1111 v1.10.1 依赖的 `Stability-AI/stablediffusion` 仓库 2026 年被删除（GitHub 404），6 个国内镜像全失败。diffusers 走 PyPI 装包，完全避开已删仓库。 |
| **UNet + VAE 都要改 padding_mode='circular'** | 只改 UNet 时 latent 边缘连续但 VAE 解码用 zeros padding 打断（比值 6.7），加 VAE 后比值 0.52（完美连续）。 |
| **diffusers 0.30.x 而非 0.41.x** | diffusers 0.41 依赖 `torch.accelerator`（torch 2.4+ 才有），torch 2.3.1 没有。0.30.3 + transformers 4.44.2 兼容。 |
| **用 A1111 遗留 venv（Python 3.10）** | 项目 venv 是 Python 3.14，torch 无对应 wheel。A1111 venv 已有 Python 3.10 + torch 2.3.1 + MPS，直接装 diffusers 纯 Python 包即可。 |
| **常驻 HTTP server 而非每次加载模型** | SDXL 6.94GB，从磁盘加载到 MPS 内存需 25-30 秒。常驻服务模型只加载一次，后续每次生图直接复用。 |
| **sandbox 会阻断 MPS GPU** | 进程 RSS=0 卡死。启动 diffusers_server 必须用 `dangerouslyDisableSandbox: true`。 |
| **circular padding 是构造性四方连续** | 让 UNet/VAE 生成时看到环形连续内容，按数学保证母题绕边延续。不是事后补缝（quilt 算法是兜底，circular padding 是根治）。 |
| **Brief 历史库在 upload ready 时写入** | 失败品不算"已占用主题"，跟 pHash 历史库写入时机保持一致。 |
| **数码花型净化器（prompt_engine）** | LLM 给数码 brief 写出 `flat vector style with fine hand-drawn line hatching` 导致产出传统花型长相。LLM 指令不可全信，按逗号分段整段剔除 + 补数码锚点。 |
| **瓦栏翻页按天轮换** | 同一天内 page 固定，翻页让不同批次看到不同在售花型，降低采集重复率。 |

---

## 📅 路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | Python 流水线（采集 → 生图 → PSD → 检测 → 上传准备） | ✅ 完成 |
| Phase 2 | 质量提升（Prompt 工程 / 视觉自评 / 专业一花四色 / 真实趋势） | ✅ 完成 |
| Phase 3 | 真实上传发布（ego-browser 全自动，已发布作品待审核） | ✅ 完成 |
| Phase 4 | 构造性四方连续（diffusers circular padding，根除切花） | ✅ 完成 |
| Phase 5 | 数码花型风格修复 + Brief 历史库语义去重 | ✅ 完成 |
| Phase 6 | 批量产出验证与调优（多轮跑量、分析过审率） | 🔜 当前 |
| Phase 7 | 销量反馈闭环（自动分析什么好卖，反哺趋势采集） | ⚪ |
| Phase 8 | React Dashboard 可视化面板 | ⚪ |

---

## 📜 参考

- [瓦栏网花型设计作品上传标准 PDF](https://www.walanwalan.com/wldata/open/articles/2017-04/f4dd720c-24a1-11e7-9c88-00163e001713.pdf)
- [Photocraft / ArtCraft](https://github.com/storytold/photocraft) — 纯 Rust 的 Photoshop 重实现（后期参考）
