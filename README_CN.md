<p align="center">
  <img src="/static/banana.jpg" width="180" alt="Edit Banana Logo"/>
</p>

<h1 align="center">🍌 Edit Banana</h1>
<p align="center">
  中文 | <a href="README.md">English</a>
</p>
<h3 align="center">全场景内容重构器：让“不可编辑”成为过去式</h3>

<p align="center">
打破静态格式的边界。让原本固定的内容，摇身一变为可灵活编辑的数字化资产。
依托 SAM 3 与多模态大模型技术，我们提供高保真重构能力，完美还原图表的精细细节与逻辑脉络。
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white" alt="License"/></a>
  <a href="https://developer.nvidia.com/cuda-downloads"><img src="https://img.shields.io/badge/GPU-CUDA%20推荐-76B900?style=flat-square&logo=nvidia" alt="CUDA"/></a>
  <a href="#-加入微信群"><img src="https://img.shields.io/badge/WeChat-加入群聊-07C160?style=flat-square&logo=wechat&logoColor=white" alt="WeChat"/></a>
  <a href="https://github.com/BIT-DataLab/Edit-Banana/stargazers"><img src="https://img.shields.io/github/stars/BIT-DataLab/Edit-Banana?style=flat-square&logo=github" alt="GitHub stars"/></a>
  <a href="https://atomgit.com/BIT-DataLab/Edit-Banana/stargazers"><img src="https://atomgit.com/BIT-DataLab/Edit-Banana/star/badge.svg" alt="AtomGit Star"/></a >
</p>

---

<h3 align="center">立即体验！</h3>
<p align="center">
  <a href="https://www.editbanana.net/">
    <img src="https://img.shields.io/badge/🚀%20在线体验-www.editbanana.net-FF6B6B?style=for-the-badge&logoColor=white" alt="Try Online Demo"/>
  </a>
</p>

<p align="center">
  👆 <b>点击上方或访问 https://www.editbanana.net/ 在线体验 Edit Banana！</b> 上传图片，几秒钟即可获得<b>可编辑的 DrawIO (XML) 文件</b>
</p>

> [!WARNING]
> **请注意**：我们的 GitHub 仓库代码目前落后于 Web 服务端。为了体验最新功能和最佳性能，我们强烈建议您使用我们的在线网页平台。
## 💬 加入微信讨论群

欢迎加入我们的微信群讨论与交换意见！可以扫描下方二维码进群：

<p align="center">
  <img src="/static/wechatGroup.jpg" width="70%" alt="WeChat Group QR Code"/>
  <br/>
  <em>扫码加入 Edit Banana 交流群</em>
</p>

> [!TIP]
> 如果二维码已过期，请提交 [Issue](https://github.com/BIT-DataLab/Edit-Banana/issues) 申请更新。

## 📮联系我们
如需学术合作、技术对接、商用授权、项目定制及其他商务咨询，欢迎通过邮件与我们取得联系:
> **邮箱: ccl@bit.edu.cn**

---

---
## 📑 目录

- [📸 效果展示](#-效果展示)
- [🚀 核心特性](#-核心特性)
- [🛠️ 架构流程](#️-架构流程)
- [📂 项目结构](#-项目结构)
- [📦 安装与设置](#-安装与设置)
- [🔤 使用方法](#-使用方法)
- [⚙️ 配置](#️-配置)
- [📌 开发路线图](#-开发路线图)
- [💬 加入微信群](#-加入微信群)
- [🤝 贡献指南](#-贡献指南)
- [🤩 贡献者](#-贡献者)
- [📄 许可证](#-许可证)
- [🌟 Star 历史](#-star-历史)

---

## 📸 效果展示

### 高清输入-输出对比（4个典型场景）

为了演示高保真转换的效果，我们提供了4个场景下的“原始静态格式”与“可编辑重构结果”的 1:1 对比。所有元素都可以独立拖拽、设置样式和修改。

#### 场景 1：图片转 DrawIO

| 🔒 原始静态图（输入 · 不可编辑） | 🔓 DrawIO 重建结果（输出 · 完全可编辑） |
|:---:|:---:|
| <br><b>示例 1：基础流程图</b><br><br><img src="/static/demo/original_1.jpg" width="450" alt="原始图 1" style="border: 1px solid #eee; border-radius: 8px;"/> | <br><b>✨ 可编辑流程图</b><br><br><img src="/static/demo/recon_1.png" width="450" alt="重建结果 1" style="border: 1px solid #eee; border-radius: 8px;"/> |
| <br><b>示例 2：多层级架构图</b><br><br><img src="/static/demo/original_2.png" width="450" alt="原始图 2" style="border: 1px solid #eee; border-radius: 8px;"/> | <br><b>✨ 可编辑架构图</b><br><br><img src="/static/demo/recon_2.png" width="450" alt="重建结果 2" style="border: 1px solid #eee; border-radius: 8px;"/> |
| <br><b>示例 3：技术原理图</b><br><br><img src="/static/demo/original_3.jpg" width="450" alt="原始图 3" style="border: 1px solid #eee; border-radius: 8px;"/> | <br><b>✨ 可编辑原理图</b><br><br><img src="/static/demo/recon_3.png" width="450" alt="重建结果 3" style="border: 1px solid #eee; border-radius: 8px;"/> |
| <br><b>示例 4：科学公式</b><br><br><img src="/static/demo/original_4.jpg" width="450" alt="原始图 4" style="border: 1px solid #eee; border-radius: 8px;"/> | <br><b>✨ 可编辑公式</b><br><br><img src="/static/demo/recon_4.png" width="450" alt="重建结果 4" style="border: 1px solid #eee; border-radius: 8px;"/> |

#### 场景 2：交互式修改

<div align="center">

<br>
<img src="static/demo/cut.gif" width="90%"/>
<br><sub>✨ 手动修复</sub>

<br><br>
<img src="static/demo/save.gif" width="90%"/>
<br><sub>✨ 保存到本地</sub>

</div>

> [!NOTE]
> **✨ 转换技术亮点：**
> 1. 保留原始图表的排版逻辑、色彩搭配和元素层级关系。
> 2. 1:1 完美还原形状的边框/填充，以及箭头的样式（如虚线/粗细）。
> 3. 精确的文本识别，支持直接进行后续文字编辑和格式调整。
> 4. 所有元素均可独立选中，支持替换原生 DrawIO 模板以及布局优化。

## 🚀 核心特性

- **高级图像分割**：使用我们微调的 **SAM 3 (Segment Anything Model 3)** 实现图表元素的精准分割。
- **固定多轮 VLM 扫描**：由 **多模态大语言模型 (Multimodal LLMs)** 引导的内容提取流程。
- **文本识别**：
  - **本地 OCR** 用于文本定位；安装简单，完全离线运行。
  - **Pix2Text** 用于数学公式识别并转换为 **LaTeX** 格式。
  - **裁剪引导策略**：精准提取文本/公式区域的高清截图，送入公式引擎识别。
- **用户系统**：
  - **注册福利**：新用户可获得 **10 个免费额度**。
  - **额度系统**：按量付费模式，防止资源滥用。
  - **多用户并发**：内置并发用户会话支持，通过**全局锁机制 (Global Lock)**实现线程安全的 GPU 访问；并使用 **LRU缓存**（最近最少使用）在请求之间持久化图像特征（Embeddings），确保高性能与稳定性。

---

## 🛠️ 架构流程

1.  **输入**：图片格式 (PNG/JPG/BMP/TIFF/WebP)。
2.  **分割 (SAM3)**：使用我们微调的 SAM3 掩码解码器 (Mask Decoder)。
3.  **文本提取 (并行)**：
    *   本地 OCR (Tesseract) 检测文字边界框。
    *   文字/公式区域的高清截图被发送至 Pix2Text 进行 LaTeX 提取。
4.  **DrawIO XML 生成**：将 SAM3 提取的空间位置数据与 OCR 文本结果进行融合合并。

---

## 📂 项目结构

  <details>

  <summary><b>点击展开项目结构 </b></summary>

  ```text
  Edit-Banana/
  ├── config/               # 配置文件 (将 config.yaml.example 复制为 config.yaml)
  ├── flowchart_text/       # OCR与文本提取模块 (独立入口)
  │   ├── src/
  │   └── main.py             # 纯OCR功能入口点
  ├── input/                # [手动创建] 输入图像存放目录
  ├── models/               # [手动创建] 模型权重 (SAM3) 及可选的 BPE 词表
  ├── output/               # [手动创建] 结果输出目录
  ├── sam3/                 # SAM3 库 (见安装说明：从 facebookresearch/sam3 安装)
  ├── sam3_service/         # SAM3 HTTP 服务引擎 (可选，用于多进程部署)
  ├── scripts/              # 环境部署及实用脚本
  │   ├── setup_sam3.sh       # 安装 SAM3 库并将 BPE 复制到 models/
  │   ├── setup_rmbg.py       # 从 ModelScope 下载 RMBG 模型
  │   └── merge_xml.py        # XML 合并工具
  ├── main.py               # 命令行入口 (模块化管道)
  ├── server_pa.py          # FastAPI 后端服务端
  └── requirements.txt      # Python 依赖清单
  ```

  </details>

  ---

## 📦 安装与设置

请按照以下核心阶段在本地进行项目设置。

### 阶段 1：环境与基础设置

配置您的基础环境和目录结构。

#### 1. 前置要求

- Python 3.10+** 及支持 CUDA 的 GPU (强烈推荐)
- 安装支持 CUDA 的 PyTorch (以 CUDA 11.8 为例)：

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

#### 2. 克隆仓库 & 初始化目录
  ```bash
  git clone https://github.com/BIT-DataLab/Edit-Banana.git
  cd Edit-Banana
  mkdir -p input output sam3_output
  ```

### 阶段 2：模型与核心依赖

接下来，安装所需的 Python 包并下载必要的模型权重（权重文件应放在 `models/` 目录下，该目录不应被提交到 Git）。

#### 1. 基础依赖

```bash
pip install -r requirements.txt
```

#### 2. SAM3 & 模型资产

- SAM3 库与 BPE 词表：
运行 `bash scripts/setup_sam3.sh` 安装该库，并将 BPE 词表复制到 `models/` 中。
通过以下命令验证：
  ```bash
  python -c "from sam3.model_builder import build_sam3_image_model; print('OK')"
  ```
- SAM3 权重：从 [ModelScope (魔搭)](https://modelscope.cn/models/facebook/sam3) 或 [Hugging Face](https://huggingface.co/facebook/sam3) 下载 `sam3.pt`，并放置到 `models/sam3_ms` 目录下。

- 文本本地 OCR (Tesseract)：
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-chi-sim
  ```

<details> <summary><b>🧩 进阶可选功能 (OCR 引擎、公式、去背景 RMBG) - 点击展开</b></summary>

- PaddleOCR (替代方案/对于混合文本效果更好)：使用 `paddlepaddle==3.2.2` 以避免 3.3.0 版本的 Bug。
  ```bash
  pip install paddlepaddle==3.2.2 paddleocr
  ```
- 公式 (Pix2Text)：
  ```bash
  pip install pix2text onnxruntime-gpu
  ```
- 去除背景 (RMBG)：运行 `pip install onnxruntime modelscope` 然后执行 `python scripts/setup_rmbg.py`。
</details>

### 阶段 3：配置与故障排查

#### 1. 完成配置

复制配置文件示例并调整资产路径：

  ```bash
  cp config/config.yaml.example config/config.yaml
  ```

编辑 `config.yaml` 确保 `sam3.checkpoint_path` 和 `sam3.bpe_path` 跟您的 `models/` 下的文件路径保持一致。

<details> <summary><b>🛠️ 首次运行前检查清单与故障排查 - 点击展开</b></summary>

**检查清单**:
- [ ] 配置文件已复制，且 `config.yaml` 中的模型路径已设置
- [ ] SAM3 权重 (`sam3.pt`) 和 BPE 词表已放置到 `models/` 下
- [ ] 通过 `scripts/setup_sam3.sh` 提取了 SAM3 库
- [ ] 成功安装 Tesseract 或 PaddleOCR

**常见问题**:
- "no kernel image is available..."：GPU 架构不匹配。尝试升级 PyTorch，或者设置 `sam3.device: "cpu"`。
- "Model file not found at ...rmbg/..."：RMBG 模块是可选的，如果你需要，请通过脚本下载启用。
- "PaddleOCR inference failed..."：请使用 `paddlepaddle==3.2.2` 版本或回退到使用 Tesseract 识别。
</details>

---

## 🔤 使用方法

### 命令行交互接口 (CLI)

支持常见的图像格式 (PNG, JPG, BMP, TIFF, WebP)。如果需要处理单张图片：

```bash
python main.py -i input/test_diagram.png
```

最终生成的 XML 文件将会保存在 `output/` 文件夹中。若要进行批量处理，请把所有图片直接放入 `input/`，并在不加 `-i` 参数的情况下运行 `python main.py`。

### 本地运行测试

1. **一次性设置**

   ```bash
   git clone https://github.com/BIT-DataLab/Edit-Banana.git && cd Edit-Banana
   python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS; 若是 Windows: .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # 或者使用纯 CPU 版本
   pip install -r requirements.txt
   sudo apt install tesseract-ocr tesseract-ocr-chi-sim   # OCR (或根据系统安装对应软件)
   ```

   安装 SAM3 库并下载模型权重与 BPE 词表，接着：

   ```bash
   mkdir -p input output
   cp config/config.yaml.example config/config.yaml
   # 编辑 config/config.yaml: 将 sam3.checkpoint_path 与 sam3.bpe_path 修改为对应的 models/ 路径
   ```

2. **使用 CLI 测试**

   ```bash
   # 在 input/ 中放入图表图片，如 input/test.png
   python main.py -i input/test.png
   # 输出文件及中间结果将存放在 output/<图像文件名>/ 文件夹下（包括 DrawIO XML文件）
   ```

3. **可选：测试 Web 服务 API**

   ```bash
   python server_pa.py
   # 开启另一个终端：
   curl -X POST http://localhost:8000/convert -F "file=@input/test.png"
   # 或打开 http://localhost:8000/docs 使用包含文件上传体验测试的 /convert 接口
   ```

---

## ⚙️ 配置

可在 `config/config.yaml` 中自定义管道行为：

- **sam3**：可调整分数阈值 (score thresholds)、NMS (非极大值抑制) 阈值，以及最大迭代循环数。
- **paths**：设置输入和输出目录的路径。
- **dominant_color**：微调颜色特征提取的灵敏度。

---

## 📌 开发路线图

| 功能模块 | 状态 | 描述 |
|----------|------|------|
| 核心转换工作流 | ✅ 已完成 | 实现了包含图像分割、重构推理与 OCR 的完整管道 |
| 智能箭头连接 | ⚠️ 开发中 | 将箭头与关联形状进行智能自动连接吸附 |
| DrawIO 模板适配 | 📍 计划中 | 支持自定义 DrawIO 模板导入 |
| 批量导出优化 | 📍 计划中 | 支持批量将结果直接导出为 DrawIO 文件格式 (.drawio) |
| 本地脱机 LLM 适配 | 📍 计划中 | 支持本地部署小视觉语言模型 (VLM)，不再强依赖外部 API |

---



## 🤝 贡献指南

我们非常欢迎各种形式的贡献（包括代码提交、错误反馈、新功能建议）：

1. Fork 本仓库
2. 创建一个新的特性分支 (`git checkout -b feature/xxx`)
3. 提交你的更改 (`git commit -m 'feat: add xxx'`)
4. 将更改推送到远程分支 (`git push origin feature/xxx`)
5. 开启一个 Pull Request (合并请求)

提交 Bug 反馈：[Issues](https://github.com/BIT-DataLab/Edit-Banana/issues)
期待新功能/建议：[Discussions](https://github.com/BIT-DataLab/Edit-Banana/discussions)

---

## 🤩 贡献者

非常感谢所有为项目演进做出贡献与反馈的开发者！

<details>
<summary><b>查看贡献者名单</b></summary>

| Name/ID | Email |
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

</details>

---

## 📄 许可证

本项目为基于 [Apache License 2.0](LICENSE) 协议的开源项目。允许保留版权声明条件下的商业使用与二次开发。

---

## 🌟 Star 历史

🌟 如果这个项目对你有帮助，请给个 Star 来支持我们！

[![Star History Chart](https://api.star-history.com/svg?repos=bit-datalab/edit-banana&type=date&legend=top-left)](https://www.star-history.com/#bit-datalab/edit-banana&type=date&legend=top-left)
