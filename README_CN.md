<p align="center">
  <img src="/static/banana.jpg" width="180" alt="Edit Banana Logo"/>
</p>

<h1 align="center">🍌 Edit Banana</h1>
<h3 align="center">通用内容重编缉器：让不可编辑变为可编辑</h3>

<p align="center">
打破静态格式的边界。让原本固定的内容，摇身一变为可灵活编辑的数字化资产。
依托 SAM 3 与多模态大模型技术，我们提供高保真重构能力，完美还原图表的精细细节与逻辑脉络。
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white" alt="License"/></a>
  <a href="https://developer.nvidia.com/cuda-downloads"><img src="https://img.shields.io/badge/GPU-CUDA%20Recommended-76B900?style=flat-square&logo=nvidia" alt="CUDA"/></a>
  <a href="#-加入微信群"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?style=flat-square&logo=wechat&logoColor=white" alt="WeChat"/></a>
  <a href="https://github.com/BIT-DataLab/Edit-Banana/stargazers"><img src="https://img.shields.io/github/stars/BIT-DataLab/Edit-Banana?style=flat-square&logo=github" alt="GitHub stars"/></a>
</p>

---

<h3 align="center">立即体验！</h3>
<p align="center">
  <a href="https://editbanana.anxin6.cn/">
    <img src="https://img.shields.io/badge/🚀%20在线体验-editbanana.anxin6.cn-FF6B6B?style=for-the-badge&logoColor=white" alt="Try Online Demo"/>
  </a>
</p>

<p align="center">
  👆 <b>点击上方或访问 https://editbanana.anxin6.cn/ 在线体验 Edit Banana！</b> 上传图片，只需几秒即可获得<b>可编辑的 DrawIO (XML)</b> 文件。
  <b>请注意</b>：我们 GitHub 仓库的代码目前落后于 Web 端服务。为了获得最新功能和最佳性能，我们建议您使用我们的在线平台。
</p>

## 💬 加入微信讨论群

欢迎加入我们的微信群进行讨论交流！

<p align="center">
  <img src="/static/wechatGroup.jpg" width="70%" alt="WeChat Group QR Code"/>
  <br/>
  <em>扫码加入 Edit Banana 交流群</em>
</p>

> 💡 如果二维码已过期，请提交 [Issue](https://github.com/BIT-DataLab/Edit-Banana/issues) 申请更新。

---

## 📸 效果演示
### 高清输入输出对比（3个典型场景）
为了展示高保真的转换效果，我们提供了“原始静态格式”与“可编辑重建结果”之间的 1:1 对比。所有元素都可以独立拖拽、设置样式和修改。

#### 场景 1：图片转 DrawIO (XML, SVG)

| 示例编号 | 原始静态图（输入 · 不可编辑） | DrawIO 重建结果（输出 · 完全可编辑） |
|--------------|-----------------------------------------------|--------------------------------------------------------|
| 示例 1：基础流程图 | <img src="/static/demo/original_1.jpg" width="400" alt="Original Diagram 1" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_1.png" width="400" alt="Reconstruction Result 1" style="border: 1px solid #eee; border-radius: 4px;"/> |
| 示例 2：多层架构图 | <img src="/static/demo/original_2.png" width="400" alt="Original Diagram 2" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_2.png" width="400" alt="Reconstruction Result 2" style="border: 1px solid #eee; border-radius: 4px;"/> |
| 示例 3：技术原理图 | <img src="/static/demo/original_3.jpg" width="400" alt="Original Diagram 3" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_3.png" width="400" alt="Reconstruction Result 3" style="border: 1px solid #eee; border-radius: 4px;"/> |
| 示例 4：科学公式图 | <img src="/static/demo/original_4.jpg" width="400" alt="Original Diagram 4" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_4.png" width="400" alt="Reconstruction Result 4" style="border: 1px solid #eee; border-radius: 4px;"/> |

#### 场景 2：人机协同修改 (Human in the Loop Modification)

> ✨ 转换亮点：
> 1. 保留原图的布局逻辑、配色方案和元素层级
> 2. 1:1 还原形状描边/填充及箭头样式（虚线/粗细）
> 3. 准确的文本识别，支持直接编辑和格式调整
> 4. 所有元素独立可选，支持原生 DrawIO 模板替换和布局优化

## 核心特性

*   **高级分割**：使用我们微调的 **SAM 3** 对图表元素进行精准分割。
*   **固定多轮 VLM 扫描**：由**多模态大模型** 引导的提取分析流程。
*   **文本识别**：
    *   **本地 OCR (Tesseract)** 用于文本定位；易于安装（`pip install pytesseract` + 系统 `tesseract-ocr`），支持离线运行。
    *   **Pix2Text** 用于数学公式识别和 **LaTeX** 转换。
    *   **裁剪引导策略**：提取文本/公式区域，并将高分辨率的裁剪图像发送给公式识别引擎以提升准确率。
*   **用户系统**： 
    *   **注册福利**：新用户可获得 **10 个免费额度**。
    *   **积分系统**：按需付费模式以防止资源滥用。
*   **多用户并发**：使用**全局锁** 机制分配线程安全的 GPU 访问权限，并结合 **LRU缓存**来持久化跨请求的图像嵌入特征，保障并发环境下的高性能和稳定性。

## 架构流水线

1.  **输入**：图像 (PNG/JPG/BMP/TIFF/WebP)。
2.  **分割 (SAM3)**：使用我们微调后的 SAM3 掩码解码器。
3.  **文本提取（并行处理）**：
    *   本地 OCR (Tesseract) 检测文本边界框。
    *   文本/公式区域的高清截图发送到 Pix2Text 进行 LaTeX 转换。
4.  **DrawIO XML 生成**：融合 SAM3 提取的空间数据与 OCR 文本结果。

## 项目结构

```
Edit-Banana/
├── config/                 # 配置文件（复制 config.yaml.example → config.yaml）
├── flowchart_text/         # OCR & 文本提取模块（独立入口）
│   ├── src/
│   └── main.py             # 仅 OCR 的入口点
├── input/                  # [手动创建] 输入图像目录
├── models/                 # [手动创建] 模型权重 (SAM3) 和可选的 BPE 词表
├── output/                 # [手动创建] 结果目录
├── sam3/                   # SAM3 库（详见安装说明：从 facebookresearch/sam3 安装）
├── sam3_service/           # SAM3 HTTP 服务（可选，用于多进程部署）
├── scripts/
│   ├── setup_sam3.sh       # 安装 SAM3 库并将 BPE 词表复制到 models/
│   ├── setup_rmbg.py       # 从魔搭 (ModelScope) 下载 RMBG 模型到 models/rmbg/
│   └── merge_xml.py        # XML 合并工具
├── main.py                 # CLI 入口（模块化流水线）
├── server_pa.py            # FastAPI 后端服务
└── requirements.txt        # Python 依赖
```

## 安装与环境配置

请按照以下步骤在本地部署该项目。

### 1. 前置要求
*   **Python 3.10+**
*   **支持 CUDA 的 GPU**（强烈建议）

### 2. 克隆仓库
```bash
git clone https://github.com/BIT-DataLab/Edit-Banana.git
cd Edit-Banana
```

### 3. 初始化目录结构
克隆完成后，您必须**手动创建**以下资源目录（Git 默认会忽略）：

```bash
# 创建输入/输出目录
mkdir -p input
mkdir -p output
mkdir -p sam3_output
```

### 3.1 需要下载的模型和资产（请勿提交到代码库）

以下**大文件未包含在本代码库中**。请自行下载并放置在以下路径。仓库通过 `.gitignore` 排除了 `models/`、`sam3_src/` 等文件夹。**请不要将这些文件提交到 Git 中。**

| 资产 | 描述 | 目标路径 | 获取方式 |
|-------|-------------|-------------|------------|
| **SAM3 权重** | 分割模型参数（必须是 `.pt` 格式） | 根据配置设定，默认 `models/sam3_ms/sam3.pt` | [ModelScope](https://modelscope.cn/models/facebook/sam3) (推荐) 或 [Hugging Face](https://huggingface.co/facebook/sam3) |
| **BPE 词表** | SAM3 文本编码器词表 | `models/bpe_simple_vocab_16e6.txt.gz` | 执行 `scripts/setup_sam3.sh` 时会自动从克隆的 `sam3_src` 复制；也可以从 [facebookresearch/sam3](https://github.com/facebookresearch/sam3) 仓库手动下载 |
| **RMBG 模型** (可选) | 用于图标/箭头的背景去除 | `models/rmbg/model.onnx` | `pip install modelscope && python scripts/setup_rmbg.py` 或从 [魔搭 RMBG-2.0](https://www.modelscope.cn/models/AI-ModelScope/RMBG-2.0/files) 下载 |

详细步骤请参阅下面的 **5. 安装 SAM3 库**、**6. 下载模型权重** 以及 **可选 — RMBG** 章节。

### 4. 安装 PyTorch（运行 SAM3 所需）
安装支持 CUDA（推荐）或纯 CPU 版本的 PyTorch。CUDA 11.8 的示例安装命令：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
如需其他 CUDA 版本或 CPU 版本，请访问 [pytorch.org](https://pytorch.org/get-started/locally/)。

### 5. 安装 SAM3 库并获取 BPE 词表
本项目依赖 SAM3 的 Python API，该代码不在本仓库内。**详细步骤：** [docs/SETUP_SAM3.md](docs/SETUP_SAM3.md)。

**快捷方式（在仓库根目录运行，并确保您的虚拟环境已激活）：**
```bash
bash scripts/setup_sam3.sh
```
该脚本会将 [facebookresearch/sam3](https://github.com/facebookresearch/sam3) 克隆到 `sam3_src`，执行 `pip install -e sam3_src`，并将 BPE 词库复制至 `models/bpe_simple_vocab_16e6.txt.gz`。

验证安装：`python -c "from sam3.model_builder import build_sam3_image_model; print('OK')"`

### 6. 下载模型权重
获取 SAM 3 的模型权重文件（checkpoint）并将其放置在 `models/` 下：
- **魔搭 ModelScope（推荐，无需申请权限）：** [modelscope.cn/models/facebook/sam3](https://modelscope.cn/models/facebook/sam3)
- **Hugging Face：** [facebook/sam3](https://huggingface.co/facebook/sam3) — 需先申请访问权限。

具体的下载命令和 `config.yaml` 配置` 请参阅 [docs/SETUP_SAM3.md](docs/SETUP_SAM3.md)。

### 7. 安装 Python 依赖

**后端（必须）：**
```bash
pip install -r requirements.txt
```

**Tesseract（默认文本 OCR；请在 Tesseract 或 PaddleOCR 中选择安装一种）：** 在您的系统上安装 Tesseract 引擎。Ubuntu 示例：
```bash
sudo apt install tesseract-ocr tesseract-ocr-chi-sim
```
如果您使用 PaddleOCR (`ocr.engine: "paddleocr"`)，Tesseract 是可选的，但建议作为回退方案。

**可选 — PaddleOCR（更适合中英混合文本）：** 使用 **PaddlePaddle 3.2.x + PaddleOCR 3.x**（推荐 3.2.2 版本；3.3.0+ 在 CPU 上有 oneDNN bug，会自动回退到 Tesseract）：
```bash
pip uninstall paddleocr paddlepaddle paddlepaddle-gpu paddlex -y
pip install paddlepaddle==3.2.2 paddleocr   # CPU 版；避免 3.3.0 的 oneDNN 缺陷
# GPU 版: pip install paddlepaddle-gpu==3.2.2 paddleocr
```
然后在 `config/config.yaml` 中设置 `ocr.engine: "paddleocr"`。

**可选 — 公式识别 (Pix2Text)：** 用于 LaTeX 数学公式识别，安装命令：
```bash
pip install pix2text
# 若有 GPU: pip install onnxruntime-gpu
```

**可选 — RMBG（去除图标/箭头的背景）：** 供 IconPictureProcessor 模块使用：
1. 安装 runtime: `pip install onnxruntime` (或 `onnxruntime-gpu`)。
2. 将 RMBG-2.0 模型下载到 `models/rmbg/model.onnx`（此脚本支持从魔搭自动下载）：
   ```bash
   pip install modelscope
   python scripts/setup_rmbg.py
   ```
   或者手动下载：从 [魔搭 RMBG-2.0](https://www.modelscope.cn/models/AI-ModelScope/RMBG-2.0/files) 下载 `model.onnx` 并存入 `models/rmbg/`。

### 8. 配置

1. **配置文件（首次运行前必备）：**
    ```bash
    cp config/config.yaml.example config/config.yaml
    ```
    编辑 `config/config.yaml`：将 `sam3.checkpoint_path` 和 `sam3.bpe_path` 指向您的 `models/` 目录路径。如果需要还可以配置 `ocr.engine: "paddleocr"`。
2.  **环境变量（可选）：** 如果您使用了外部 API 密钥或自定义服务端点，请在根目录下创建一个 `.env` 文件进行设置。

### 9. 注意事项与故障排除

**推荐版本参考**

| 组件 | 推荐版本 | 备注 |
|-----------|---------|-------|
| Python | 3.10+ | 需兼容 PyTorch 和 Paddle |
| PyTorch | 2.x + 匹配 GPU 的 CUDA | 新架构 GPU (如 Blackwell sm_120) 可能需要 cu128；或者在配置页设置 `sam3.device: "cpu"` |
| SAM3 权重 | `sam3.pt` (非 safetensors) | 在配置文件设置 `config.sam3.checkpoint_path`，如 `models/sam3_ms/sam3.pt` |
| PaddleOCR | PaddlePaddle **3.2.2** + PaddleOCR 3.x | 3.3.0+ 存在 CPU oneDNN bug，会导致流水线强制退回 Tesseract |
| Tesseract | 操作系统级安装 | Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-chi-sim` |
| RMBG | onnxruntime + `models/rmbg/model.onnx` | 可选；使用 `scripts/setup_rmbg.py` 或魔搭进行下载 |

**首次运行前检查清单**

- [ ] 将 `config/config.yaml.example` 复制为 `config/config.yaml` 并且配置好 `sam3.checkpoint_path`、`sam3.bpe_path`
- [ ] 将 SAM3 权重（例如 `models/sam3_ms/sam3.pt`）和 BPE (`models/bpe_simple_vocab_16e6.txt.gz`) 置于 `models/` 目录下
- [ ] 执行 `scripts/setup_sam3.sh` 或参照 [docs/SETUP_SAM3.md](docs/SETUP_SAM3.md) 完成了 SAM3 库的本地安装
- [ ] 全局系统已安装 Tesseract，或配置并安装了 PaddleOCR 且将配置设为 `ocr.engine: "paddleocr"`

**常见问题**

- **"no kernel image is available for execution on the device"（设备上无可执行的核心镜像）** — GPU 架构与 PyTorch CUDA 版本不匹配。请在 `config.yaml` 中把 `sam3.device` 改为 `"cpu"`，或升级 PyTorch 至匹配的 CUDA 版本（如 cu128）。
- **"Model file not found at .../models/rmbg/model.onnx"** — RMBG 属于可选项，如果您不需要背景抠取，大可忽略。如需开启：运行 `pip install modelscope && python scripts/setup_rmbg.py`，或手动从魔搭拉取放入 `models/rmbg/` 目录。
- **"PaddleOCR inference failed…fallback to Tesseract"** — Paddle/oneDNN 存在兼容性问题。使用 `paddlepaddle==3.2.2` + `paddleocr`，或者在配置中设回 `ocr.engine: "tesseract"`。
- **"Please install PaddleOCR" / "pytesseract not installed"** — 未安装相应的 OCR 栈；若是只用 Tesseract，请确保系统已安装 `tesseract-ocr` 并执行了 `pip install pytesseract`。
- **命令行停滞在 "Checking connectivity to the model hosters"** — `main.py` 默认已配置 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`；如果仍有这种情况，可以在启动前手动运行 `export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`。

## 使用方法

### 命令行接口 (CLI)

支持处理图像文件 (PNG, JPG, BMP, TIFF, WebP)。处理单张图像的操作如下：

```bash
python main.py -i input/test_diagram.png
```
生成的 XML 结果将保存在 `output/` 文件夹下。如果是批量处理，可以将多张图片放入 `input/` 后直接运行 `python main.py`（不要带 `-i`）。

### 本地运行和测试

1. **首次设置**
   ```bash
   git clone https://github.com/BIT-DataLab/Edit-Banana.git && cd Edit-Banana
   python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS; Windows环境: .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # 或安装 CPU 版
   pip install -r requirements.txt
   sudo apt install tesseract-ocr tesseract-ocr-chi-sim   # OCR 引擎 (或其他 OS 等效安装方案)
   ```
   安装 SAM3 库（参考 [安装 SAM3 库](#5-安装-sam3-库并获取-bpe-词表)）并下载相关的 [模型权重与 BPE](#6-下载模型权重)。然后：
   ```bash
   mkdir -p input output
   cp config/config.yaml.example config/config.yaml
   # 编辑 config/config.yaml: 配置 sam3.checkpoint_path 等参数指向您的 models/ 目录
   ```

2. **CLI 测试**
   ```bash
   # 在 input/ 放置一张图片，比如 input/test.png
   python main.py -i input/test.png
   # 最终输出结果（含 DrawIO XML 及中间产物）会出现在 output/<image_stem>/ 路径下
   ```

3. **可选：测试 Web API**
   ```bash
   python server_pa.py
   # 在另外一个终端窗口进行推送测试：
   curl -X POST http://localhost:8000/convert -F "file=@input/test.png"
   # 或在浏览器打开 http://localhost:8000/docs，并使用 /convert 接口上传图片进行交互式测试
   ```

## 配置文件 `config.yaml`

在 `config/config.yaml` 中，您可以自定义流水线的控制行为：
*   **sam3**：调整置信度评分阈值、NMS (非极大值抑制) 阈值，以及最大迭代循环数。
*   **paths**：指定输入、输出文件目录。
*   **dominant_color**：微调颜色特征提取的灵敏度。

## 📌 开发路线图
| 功能模块           | 状态       | 描述                     |
|--------------------------|--------------|---------------------------------|
| 核心转换流水线 | ✅ 已完成 | 包括分割、重构、OCR 及文字融合的完整流 |
| 智能箭头连接 | ⚠️ 开发中 | 自动识别箭头锚点并与其他形状做精准连接 |
| DrawIO 模板适配 | 📍 规划中 | 支持用户导入自定义的 DrawIO 样式模板库 |
| 批量导出优化 | 📍 规划中 | 支持原生批处理并导出 DrawIO (.drawio) 标准格式文件 |
| 本地 LLM 适配 | 📍 规划中 | 兼容并支持本地自主部署的 VLM 引擎以替代在线 API |

## 🤝 贡献指南
我们欢迎各种形式的代码贡献（如代码提交、Bug 反馈及功能建议等）：
1.  Fork 本仓库 (Fork this repository)
2.  创建您的功能分支 (`git checkout -b feature/xxx`)
3.  提交您的修改 (`git commit -m 'feat: add xxx'`)
4.  推送至分支 (`git push origin feature/xxx`)
5.  提交合并请求 (Pull Request)

提交 Bug: [Issues](https://github.com/BIT-DataLab/Edit-Banana/issues)  
需求及特性建议: [Discussions](https://github.com/BIT-DataLab/Edit-Banana/discussions)

## 🤩 贡献者
感谢所有为本项目贡献代码及推动其不断迭代的开发者！

| 姓名 / ID | 邮箱 |
|---------|-------|
| Chai Chengliang | ccl@bit.edu.cn |
| Zhang Chi | zc315@bit.edu.cn |
| Deng Qiyan |  |
| Rao Sijing |  |
| Yi Xiangjian |  |
| Li Jianhui |  |
| Shen Chaoyuan |  |
| Zhang Junkai |  |
| Han Junyi |  |
| You Zirui |  |
| Xu Haochen |  |
| An Minghao |  |
| Yu Mingjie |  |
| Yu Xinjiang|  |
| Chen Zhuofan|  |
| Li Xiangkun|  |

## 📄 许可证
本项目在 [Apache License 2.0](LICENSE) 协议下开源，允许商业应用及二次开发（请保留版权声明）。

---
## 🌟 Star 历史

🌟 如果本项目对您有帮助，请给个 Star 以示支持！

[![Star History Chart](https://api.star-history.com/svg?repos=bit-datalab/edit-banana&type=date&legend=top-left)](https://www.star-history.com/#bit-datalab/edit-banana&type=date&legend=top-left)
