# 图像转 DrawIO (XML) 转换器 (Image to DrawIO Converter)

本项目实现了一套精密的自动化流程，利用先进的计算机视觉模型和多模态大语言模型，将静态图像（如流程图、架构图、技术绘图）转换为可编辑的 DrawIO (mxGraph) XML 文件，实现高保真的逆向工程与重建。

[English README](README.md)

## 核心功能

*   **先进分割技术 (Advanced Segmentation)**: 采用 **SAM 3 (Segment Anything Model 3)** 模型，对图表元素（基础形状、箭头、图标）进行 SOTA 级别的精准分割。
*   **智能迭代细化 (Intelligent Refinement)**: 引入 **多模态大模型 (GPT-4o/Gemini/Claude)** 引导的迭代提取机制。系统会自动分析未识别区域，生成补充提示词，确保遗漏的微小元素也能被捕获。
*   **高质量 OCR 与公式识别**:
    *   **Azure Document Intelligence**: 提供工业级的精准文本定位（Bounding Box）。
    *   **Mistral Vision/MLLM**: 专门用于校对文本内容，能够将复杂的数学公式精确转换为 **LaTeX** 格式（例如 $\int f(x) dx$），并在 DrawIO 中完美渲染。
    *   **局部裁剪策略 (Crop-Guided Strategy)**: 将文本/公式区域裁剪为高清小图发送给 LLM，从根本上解决了小字号模糊和公式乱码问题。
*   **智能背景移除 (Smart Background Removal)**: 集成 **RMBG-2.0** 模型，自动对图标、图片和箭头进行精细抠图（去背），确保它们在 DrawIO 中可以完美叠加，无白色背景干扰。
*   **高保真箭头处理**: 摒弃了不稳定的矢量化路径生成，将箭头作为透明图像提取。这种方法能完美保留虚线、曲线、复杂的路由走向和端点样式，实现了视觉上的绝对一致。
*   **矢量形状恢复**: 标准几何形状（矩形、菱形、椭圆、圆柱体等）会被识别并转换为原生的 DrawIO 矢量对象，并自动提取填充色和描边色。
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
├── inputs/                 # 输入图片目录
├── models/                 # 模型权重目录 (RMBG 等)
├── output/                 # 输出结果目录
├── sam3/                   # SAM3 模型库
├── scripts/                # 核心处理脚本
│   ├── sam3_extractor.py   # 分割与图像提取逻辑 (SAM3 + RMBG)
│   ├── merge_xml.py        # XML 合并与流程编排
│   └── run_all.py          # 命令行入口 (CLI)
├── server.py               # FastAPI 后端服务
└── requirements.txt        # Python 依赖列表
```

## 安装指南

### 前置要求
*   Python 3.10+
*   Node.js & npm (若需运行前端)
*   支持 CUDA 的 GPU (推荐用于加速 SAM3 和 RMBG 推理)

### 安装步骤

1.  **安装 Python 依赖**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **模型准备**:
    请确保将以下模型文件放置在 `models/` 目录下：
    *   `models/rmbg/model.onnx` (RMBG-2.0 权重)
    *   SAM3 的 Checkpoint 文件 (在 `config/config.yaml` 中配置路径)

3.  **环境配置**:
    在 `flowchart_text/` 目录下或项目根目录创建 `.env` 文件，填入必要的 API 密钥：
    ```env
    AZURE_ENDPOINT=your_azure_endpoint
    AZURE_API_KEY=your_azure_key
    MISTRAL_API_KEY=your_mistral_key
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

