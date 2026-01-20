# 图像转 DrawIO (XML) 转换器 (Image to DrawIO Converter)
一键将静态图表（流程图、架构图、技术原理图）转换为**可编辑的 DrawIO (mxGraph) XML 文件**。依托 SAM 3 与多模态大模型，实现高保真度重构，保留原图细节与逻辑关系，便于极速二次编辑。

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Image2DrawIO-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/XiangjianYi/Image2DrawIO)
[![CUDA Required](https://img.shields.io/badge/GPU-CUDA%20Recommended-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-downloads)

---

访问 `https://db121-img2xml.cn/`，上传图片即可完成转换，一键导出 DrawIO XML 文件。

[English README](README.md)

## 📸 效果演示
### 高清输入输出对比 (3个典型场景)
为了直观展示高保真转换效果，以下提供 3 组“原始静态图片”与“DrawIO 可编辑重建结果”的一对一对比。所有元素均可独立拖拽、设置样式和修改。

| 场景编号 | 原始静态图片 (输入 · 不可编辑) | DrawIO 重建结果 (输出 · 完全可编辑) |
|--------------|-----------------------------------------------|--------------------------------------------------------|
| 场景 1: 基础流程图 | <img src="/static/demo/original_1.jpg" width="400" alt="原始图表 1" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_1.png" width="400" alt="重建结果 1" style="border: 1px solid #eee; border-radius: 4px;"/> |
| 场景 2: 多层架构图 | <img src="/static/demo/original_2.png" width="400" alt="原始图表 2" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_2.png" width="400" alt="重建结果 2" style="border: 1px solid #eee; border-radius: 4px;"/> |
| 场景 3: 技术原理图 | <img src="/static/demo/original_3.jpg" width="400" alt="原始图表 3" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_3.png" width="400" alt="重建结果 3" style="border: 1px solid #eee; border-radius: 4px;"/> |
| 场景 4: 科学公式图 | <img src="/static/demo/original_4.jpg" width="400" alt="原始图表 4" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_4.png" width="400" alt="重建结果 4" style="border: 1px solid #eee; border-radius: 4px;"/> |

> ✨ 转换亮点:
> 1.  保留原图的布局逻辑、配色方案和元素层级
> 2.  1:1 还原形状描边/填充及箭头样式（虚线/粗细）
> 3.  精准的文本识别，支持直接的后续编辑和格式调整
> 4.  所有元素均可独立选中，支持原生的 DrawIO 模版替换和布局优化

## 核心功能

*   **先进分割技术 (Advanced Segmentation)**: 采用 **SAM 3 (Segment Anything Model 3)** 模型，对图表元素（基础形状、箭头、图标）进行 SOTA 级别的精准分割。
*   **固定四轮迭代扫描 (Fixed 4-Round VLM Scanning)**: 引入 **多模态大模型 (Qwen-VL/GPT-4V)** 进行四轮结构化扫描，彻底杜绝元素遗漏：
    1.  **初始全量提取**: 识别基础形状与图标。
    2.  **单词查漏 (Single Word Round)**: 扫描未识别区域的单一物体。
    3.  **双词精修 (Two-Word Round)**: 针对特定属性或罕见物体进行提取。
    4.  **短语补全 (Phrase Round)**: 识别复杂组合或长描述物体。
*   **高质量 OCR 与公式识别**:
    *   **Azure Document Intelligence**: 提供工业级的精准文本定位（Bounding Box）。
    *   **Fallback Mechanism**: 如果 Azure 服务不可用，自动切换到基于 VLM 的端到端 OCR。
    *   **Mistral Vision/MLLM**: 专门用于校对文本内容，能够将复杂的数学公式精确转换为 **LaTeX** 格式（例如 $\int f(x) dx$），并在 DrawIO 中完美渲染。
    *   **局部裁剪策略 (Crop-Guided Strategy)**: 将文本/公式区域裁剪为高清小图发送给 LLM，从根本上解决了小字号模糊和公式乱码问题。
*   **智能背景移除 (Smart Background Removal)**: 集成 **RMBG-2.0** 模型，自动对图标、图片和箭头进行精细抠图（去背），确保它们在 DrawIO 中可以完美叠加，无白色背景干扰。
*   **高保真箭头处理**: 摒弃了不稳定的矢量化路径生成，将箭头作为透明图像提取。这种方法能完美保留虚线、曲线、复杂的路由走向和端点样式，实现了视觉上的绝对一致。
*   **矢量形状恢复**: 标准几何形状会被识别并转换为原生的 DrawIO 矢量对象，并自动提取填充色和描边色。
    *   **支持形状**: 矩形、圆角矩形、菱形(Decision)、椭圆(Start/End)、圆柱(Database)、云、六边形、三角形、平行四边形、小人(Actor)、标题栏(Title Bar)、文本气泡(Text Bubble)、分组框(Section Panel)。
*   **用户系统**: 
    *   **注册**: 新用户注册即送 **30 免费积分**。
    *   **积分系统**: 按次付费模式，防止资源滥用。
*   **多用户并发支持 (Multi-User Concurrency)**: 通过 **全局锁 (Global Lock)** 和 **LRU 缓存 (LRU Cache)** 机制，实现线程安全的 GPU 资源管理。系统能高效处理多用户并发请求，复用图像特征编码，并在保证显存安全的同时显著提升响应速度。
*   **全栈 Web 界面**: 提供基于 React 的现代化前端和 FastAPI 后端，支持拖拽上传、进度实时显示和在线编辑预览。

## 架构流程

1.  **输入**: 图像文件 (PNG/JPG)。
2.  **分割 (SAM3)**:
    *   首轮提取：使用标准提示词（rectangle, arrow, icon）进行全图扫描。
    *   迭代循环：计算未识别区域比例 -> 请求 MLLM 观察掩码图 -> 获取新提示词 -> 重新运行 SAM3 解码器。
3.  **元素处理**:
    *   **矢量形状**: 提取颜色（填充/描边），映射为 DrawIO XML 几何体。
    *   **图像元素 (图标/箭头)**: 坐标裁剪 -> 智能 Padding -> Mask 过滤 -> RMBG-2.0 去背 -> Base64 编码。
4.  **文本提取 (并行处理)**:
    *   Azure OCR 检测文本包围盒。
    *   对每个文本区域进行高清裁剪。
    *   Mistral/LLM 识别内容并判断是否为公式（转 LaTeX）。
5.  **XML 生成**:
    *   合并 SAM3 的空间数据与 OCR 的文本数据。
    *   应用 Z-Index 层级排序（大面积形状置底，文字和连线置顶）。
    *   生成最终的 `.drawio.xml` 文件。

## 项目结构

```
sam3_workflow/
├── config/                 # 配置文件
├── flowchart_text/         # OCR 与文本提取模块
│   ├── src/                # OCR 核心代码 (Azure, Mistral, 文本对齐)
│   └── main.py             # OCR 入口程序
├── frontend/               # React 前端应用
├── input/                  # [需手动创建] 输入图片目录
├── models/                 # [需手动创建] 模型权重目录
│   └── rmbg/               # [需手动创建] RMBG模型
├── output/                 # [需手动创建] 输出结果目录
├── sam3/                   # SAM3 模型库
├── scripts/                # 核心处理脚本
│   ├── sam3_extractor.py   # 分割与图像提取逻辑 (SAM3 + RMBG)
│   ├── merge_xml.py        # XML 合并与流程编排
│   └── run_all.py          # 命令行入口 (CLI)
├── server.py               # FastAPI 后端服务
└── requirements.txt        # Python 依赖列表
```

## 🛠️ 安装与部署指南 (Installation Guide)

### 1. 环境准备
*   **Python 3.10+**
*   **Node.js & npm** (前端运行需要)
*   **NVIDIA GPU** (强烈推荐，用于加速 SAM3 和 RMBG 推理)

### 2. 克隆代码仓库
```bash
git clone https://github.com/XiangjianYi/Image2DrawIO.git
cd Image2DrawIO

conda create --name img2xml python=3.12 -y
conda activate img2xmls
```

### 3. 初始化文件夹结构 (Initialize Folders)
由于 Git 忽略了大文件和临时目录，**拉取代码后必须手动创建以下文件夹**：

```bash
# 创建输入输出目录
mkdir -p input
mkdir -p output
mkdir -p sam3_output

# 创建模型存放目录
mkdir -p models/rmbg
```

### 4. 下载模型权重 (Download Models)
请下载对应的模型文件并放入指定目录：

| 模型名称 (Model) | 用途 | 下载链接 | 目标路径 (Target Path) |
| :--- | :--- | :--- | :--- |
| **RMBG-2.0** | 背景移除 (去底) | [RMBG-2.0](https://modelscope.cn/models/AI-ModelScope/RMBG-2.0/tree/master/onnx) | `models/rmbg/model.onnx` |
| **SAM 3** | 图像分割 | https://modelscope.cn/models/facebook/sam3 | `models/sam3.pt` (需在配置中指定) |

> ⚠️ **注意**: `models/rmbg` 文件夹下必须包含 `model.onnx` 文件。

```bash
pip install modelscope
modelscope download --model AI-ModelScope/RMBG-2.0 onnx/model.onnx --local_dir ./models/rmbg
mv ./models/rmbg/onnx/model.onnx  ./models/rmbg/model.onnx 
```

### 5. 安装依赖 (Dependencies)

**后端:**
```bash
pip install -r requirements.txt

# 国内镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**pytorch-gpu安装**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# Linux + Cuda：13.0，可查阅：https://pytorch.org/get-started/locally/
```

**前端:**
```bash
cd frontend
npm install
cd ..
```

### 6. 配置文件 (Configuration)

**1. 复制配置文件**
```bash
cp config/config.yaml.example config/config.yaml
```

**2. 配置环境变量 (.env)**
在项目根目录创建 `.env` 文件，填入必要的 API 密钥：
```env
AZURE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_API_KEY=your_azure_key
# 其他常用 Key
# OPENAI_API_KEY=...
# DASHSCOPE_API_KEY=...
```

## 使用指南

### 1. Web 界面 (推荐)

启动后端服务:
```bash
python server.py
# 服务运行在 http://localhost:8000
```

启动前端界面:
```bash
cd frontend
npm install
npm run dev
# 界面运行在 http://localhost:5173
```
打开浏览器访问前端地址，上传图片即可查看转换结果。

### 2. 命令行工具 (CLI)

处理单张图片:

```bash
python scripts/run_all.py --image input/test_diagram.png
```
生成的 XML 文件将保存在 `output/` 目录下。

## 配置说明 `config.yaml`

您可以在 `config/config.yaml` 中自定义流水线的行为：
*   **sam3**: 调整置信度阈值 (score_threshold)、NMS 重叠阈值、最大迭代次数。
*   **paths**: 设置输入/输出文件夹路径。
*   **dominant_color**: 微调颜色提取的敏感度和策略。

## 📌 开发路线图 (Development Roadmap)
| 功能模块 (Feature Module) | 状态 (Status) | 说明 (Description) |
|--------------------------|--------------|-------------------|
| 核心转换流水线 (Core Conversion Pipeline) | ✅ 已完成 | 完整的分割、重建与 OCR 流程 |
| 智能箭头连接 (Intelligent Arrow Connection) | ⚠️ 开发中 | 自动建立箭头与目标形状的逻辑连接 |
| DrawIO 模版适配 (DrawIO Template Adaptation) | 📍 计划中 | 支持导入自定义模版样式 |
| 批量导出优化 (Batch Export Optimization) | 📍 计划中 | 批量导出为 .drawio 源文件 |
| 本地 LLM 适配 (Local LLM Adaptation) | 📍 计划中 | 支持本地部署 VLM，摆脱 API 依赖 |

## 🤝 贡献指南 (Contribution Guidelines)
欢迎各种形式的贡献（代码提交、Bug 反馈、功能建议）：
1.  Fork 本仓库
2.  创建特性分支 (`git checkout -b feature/xxx`)
3.  提交您的修改 (`git commit -m 'feat: add xxx'`)
4.  推送到该分支 (`git push origin feature/xxx`)
5.  提交 Pull Request (PR)

Bug 反馈: [Issues](https://github.com/XiangjianYi/Image2DrawIO/issues)
功能建议: [Discussions](https://github.com/XiangjianYi/Image2DrawIO/discussions)

## 🤩 贡献者 (Contributors)
感谢所有为项目做出贡献并推动其迭代的开发者！

| 姓名/ID | 邮箱 |
|---------|-------|
| Chai Chengliang | ccl@bit.edu.cn |
| Zhang Chi | zc315@bit.edu.cn |
| Rao Sijing |  |
| Yi Xiangjian |  |
| Li Jianhui |  |
| Xu Haochen |  |
| Yang Haotian |  |
| An Minghao |  |
| Yu Mingjie |  |

## 📄 开源协议 (License)
本项目基于 [Apache License 2.0](LICENSE) 协议开源，允许商业使用与二次开发（需保留版权声明）。

---
> 🌟 如果本项目对您有帮助，请给个 Star 以示支持！
> 
> [![GitHub stars](https://img.shields.io/github/stars/XiangjianYi/Image2DrawIO?style=social)](https://github.com/XiangjianYi/Image2DrawIO/stargazers)

