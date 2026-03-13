<p align="center">
  <img src="/static/banana.jpg" width="180" alt="Edit Banana Logo"/>
</p>

<h1 align="center">🍌 Edit Banana</h1>
<h3 align="center">Universal Content Re-Editor: Make the Uneditable, Editable</h3>

<p align="center">
Break free from static formats. Our platform empowers you to transform fixed content into fully manipulatable assets.
Powered by SAM 3 and multimodal large models, it enables high-fidelity reconstruction that preserves the original diagram details and logical relationships.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white" alt="License"/></a>
  <a href="https://developer.nvidia.com/cuda-downloads"><img src="https://img.shields.io/badge/GPU-CUDA%20Recommended-76B900?style=flat-square&logo=nvidia" alt="CUDA"/></a>
  <a href="#-join-wechat-group"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?style=flat-square&logo=wechat&logoColor=white" alt="WeChat"/></a>
  <a href="https://github.com/BIT-DataLab/Edit-Banana/stargazers"><img src="https://img.shields.io/github/stars/BIT-DataLab/Edit-Banana?style=flat-square&logo=github" alt="GitHub stars"/></a>
</p>

---

<h3 align="center">Try It Now!</h3>
<p align="center">
  <a href="https://editbanana.anxin6.cn/">
    <img src="https://img.shields.io/badge/🚀%20Try%20Online%20Demo-editbanana.anxin6.cn-FF6B6B?style=for-the-badge&logoColor=white" alt="Try Online Demo"/>
  </a>
</p>

<p align="center">
  👆 <b>Click above or https://editbanana.anxin6.cn/ to try Edit Banana online!</b> Upload an image to get <b>editable DrawIO (XML)</b> in seconds. 
  <b>Please note</b>: Our GitHub repository currently trails behind our web-based service. For the most up-to-date features and performance, we recommend using our web platform.
</p>

## 💬 Join WeChat Group

Welcome to join our WeChat group to discuss and exchange ideas! Scan the QR code below to join:

<p align="center">
  <img src="/static/wechat_20260309.png" width="70%" alt="WeChat Group QR Code"/>
  <br/>
  <em>Scan to join the Edit Banana community</em>
</p>

> 💡 If the QR code has expired, please submit an [Issue](https://github.com/BIT-DataLab/Edit-Banana/issues) to request an updated one.

---

## 📸 Effect Demonstration
### High-Definition Input-Output Comparison (3 Typical Scenarios)
To demonstrate the high-fidelity conversion effect, we provides one-to-one comparisons between 3 scenarios of "original static formats" and "editable reconstruction results". All elements can be individually dragged, styled, and modified.

#### Scenario 1: Figures to DrawIO (XML, SVG)

| Example No. | Original Static Diagram (Input · Non-editable) | DrawIO Reconstruction Result (Output · Fully Editable) |
|--------------|-----------------------------------------------|--------------------------------------------------------|
| Example 1: Basic Flowchart | <img src="/static/demo/original_1.jpg" width="400" alt="Original Diagram 1" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_1.png" width="400" alt="Reconstruction Result 1" style="border: 1px solid #eee; border-radius: 4px;"/> |
| Example 2: Multi-level Architecture Diagram | <img src="/static/demo/original_2.png" width="400" alt="Original Diagram 2" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_2.png" width="400" alt="Reconstruction Result 2" style="border: 1px solid #eee; border-radius: 4px;"/> |
| Example 3: Technical Schematic | <img src="/static/demo/original_3.jpg" width="400" alt="Original Diagram 3" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_3.png" width="400" alt="Reconstruction Result 3" style="border: 1px solid #eee; border-radius: 4px;"/> |
| Example 4: Scientific Formula Diagram | <img src="/static/demo/original_4.jpg" width="400" alt="Original Diagram 4" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_4.png" width="400" alt="Reconstruction Result 4" style="border: 1px solid #eee; border-radius: 4px;"/> |

#### Scenario 2: Human in the Loop Modification

> ✨ Conversion Highlights:
> 1.  Preserves the layout logic, color matching, and element hierarchy of the original diagram
> 2.  1:1 restoration of shape stroke/fill and arrow styles (dashed lines/thickness)
> 3.  Accurate text recognition, supporting direct subsequent editing and format adjustment
> 4.  All elements are independently selectable, supporting native DrawIO template replacement and layout optimization

## Key Features

*   **Advanced Segmentation**: Using our fine-tuned **SAM 3 (Segment Anything Model 3)** for segmentation of diagram elements.
*   **Fixed Multi-Round VLM Scanning**: An extraction process guided by **Multimodal LLMs (Qwen-VL/GPT-4V)**.
*   **Text Recognition**:
    *   **Local OCR (Tesseract)** for text localization; easy to install (`pip install pytesseract` + system `tesseract-ocr`), runs offline.
    *   **Pix2Text** for mathematical formula recognition and **LaTeX** conversion ($\int f(x) dx$).
    *   **Crop-Guided Strategy**: Extracts text/formula regions and sends high-res crops to the formula engine.
*   **User System**: 
    *   **Registration**: New users receive **10 free credits**.
    *   **Credit System**: Pay-per-use model prevents resource abuse.
*   **Multi-User Concurrency**: Built-in support for concurrent user sessions using a **Global Lock** mechanism for thread-safe GPU access and an **LRU Cache** (Least Recently Used) to persist image embeddings across requests, ensuring high performance and stability.

## Architecture Pipeline

1.  **Input**: Image (PNG/JPG/BMP/TIFF/WebP).
2.  **Segmentation (SAM3)**: Using our fine-tuned SAM3 mask decoder.
3.  **Text Extraction (Parallel)**:
    *   Local OCR (Tesseract) detects text bounding boxes.
    *   High-res crops of text/formula regions are sent to Pix2Text for LaTeX conversion.
4.  **DrawIO XML Generation**: Merging spatial data from SAM3 and text OCR results.

## Project Structure

```
Edit-Banana/
├── config/                 # Configuration files (copy config.yaml.example → config.yaml)
├── flowchart_text/         # OCR & Text Extraction Module (standalone entry)
│   ├── src/
│   └── main.py             # OCR-only entry point
├── input/                  # [Manual] Input images directory
├── models/                 # [Manual] Model weights (SAM3) and optional BPE vocab
├── output/                 # [Manual] Results directory
├── sam3/                   # SAM3 library (see Installation: install from facebookresearch/sam3)
├── sam3_service/           # SAM3 HTTP service (optional, for multi-process deployment)
├── scripts/
│   ├── setup_sam3.sh       # Install SAM3 lib and copy BPE to models/
│   ├── setup_rmbg.py       # Download RMBG model from ModelScope to models/rmbg/
│   └── merge_xml.py        # XML merge utilities
├── main.py                 # CLI entry (modular pipeline)
├── server_pa.py            # FastAPI backend server
└── requirements.txt       # Python dependencies
```

## Installation & Setup

Follow these steps to set up the project locally.

### 1. Prerequisites
*   **Python 3.10+**
*   **CUDA-capable GPU** (Highly recommended)

### 2. Clone Repository
```bash
git clone https://github.com/BIT-DataLab/Edit-Banana.git
cd Edit-Banana
```

### 3. Initialize Directory Structure
After cloning, you must **manually create** the following resource directories (ignored by Git):

```bash
# Create input/output directories
mkdir -p input
mkdir -p output
mkdir -p sam3_output
```

### 3.1 需用户自行下载的模型与资源（请勿提交到仓库）

以下大文件**不包含在本仓库中**，请自行下载并放到指定目录。仓库已通过 `.gitignore` 排除 `models/`、`sam3_src/` 等，**请勿将这些文件提交到 Git**。

| 资源 | 说明 | 存放路径 | 获取方式 |
|------|------|----------|----------|
| **SAM3 权重** | 分割模型权重（须为 `.pt` 格式） | `models/sam3_ms/sam3.pt` 或按 config 配置 | [ModelScope](https://modelscope.cn/models/facebook/sam3)（推荐）或 [Hugging Face](https://huggingface.co/facebook/sam3) |
| **BPE 词表** | SAM3 文本编码词表 | `models/bpe_simple_vocab_16e6.txt.gz` | 运行 `scripts/setup_sam3.sh` 时会从克隆的 `sam3_src` 复制；或从 [facebookresearch/sam3](https://github.com/facebookresearch/sam3) 仓库 assets 获取 |
| **RMBG 模型**（可选） | 背景移除，用于图标/箭头 | `models/rmbg/model.onnx` | `pip install modelscope && python scripts/setup_rmbg.py` 或 [ModelScope RMBG-2.0](https://www.modelscope.cn/models/AI-ModelScope/RMBG-2.0/files) 手动下载 |

详细步骤见下文「5. Install SAM3 library」「6. Download model weights」与「Optional — RMBG」。

### 4. Install PyTorch (required for SAM3)
Install PyTorch with CUDA support (recommended) or CPU-only. Example for CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
For other CUDA versions or CPU, see [pytorch.org](https://pytorch.org/get-started/locally/).

### 5. Install SAM3 library and get BPE
This project uses the SAM3 Python API; the code is not in this repo. **Detailed steps:** [docs/SETUP_SAM3.md](docs/SETUP_SAM3.md).

**Quick path (from repo root, with venv activated):**
```bash
bash scripts/setup_sam3.sh
```
This clones [facebookresearch/sam3](https://github.com/facebookresearch/sam3) into `sam3_src`, runs `pip install -e sam3_src`, and copies the BPE vocab to `models/bpe_simple_vocab_16e6.txt.gz`.

Verify: `python -c "from sam3.model_builder import build_sam3_image_model; print('OK')"`

### 6. Download model weights
Get the SAM 3 checkpoint and place it under `models/`:
- **ModelScope (recommended, no access request):** [modelscope.cn/models/facebook/sam3](https://modelscope.cn/models/facebook/sam3)
- **Hugging Face:** [facebook/sam3](https://huggingface.co/facebook/sam3) — request access first.

See [docs/SETUP_SAM3.md](docs/SETUP_SAM3.md) for download commands and `config.yaml` setup.

### 7. Install Python dependencies

**Backend (required):**
```bash
pip install -r requirements.txt
```

**Tesseract (default text OCR; install one of Tesseract or PaddleOCR):** Install the Tesseract engine on your system. Example on Ubuntu:
```bash
sudo apt install tesseract-ocr tesseract-ocr-chi-sim
```
If you use PaddleOCR (`ocr.engine: "paddleocr"`), Tesseract is optional but recommended as fallback.

**Optional — PaddleOCR (better for mixed Chinese/English text):** Use **PaddlePaddle 3.2.x + PaddleOCR 3.x** (recommend 3.2.2; 3.3.0+ has a CPU oneDNN bug and will auto-fallback to Tesseract):
```bash
pip uninstall paddleocr paddlepaddle paddlepaddle-gpu paddlex -y
pip install paddlepaddle==3.2.2 paddleocr   # CPU; avoids 3.3.0 oneDNN bug
# GPU: pip install paddlepaddle-gpu==3.2.2 paddleocr
```
Then in `config/config.yaml` set `ocr.engine: "paddleocr"`.

**Optional — formula recognition (Pix2Text):** For LaTeX formula recognition, install:
```bash
pip install pix2text
# GPU: pip install onnxruntime-gpu
```

**Optional — RMBG (background removal for icons/arrows):** For IconPictureProcessor:
1. Install runtime: `pip install onnxruntime` (or `onnxruntime-gpu`).
2. Download RMBG-2.0 model (e.g. from ModelScope) to `models/rmbg/model.onnx`:
   ```bash
   pip install modelscope
   python scripts/setup_rmbg.py
   ```
   Or manually: [ModelScope RMBG-2.0](https://www.modelscope.cn/models/AI-ModelScope/RMBG-2.0/files) — download `model.onnx` into `models/rmbg/`.

### 8. Configuration

1.  **Config file (required before first run):**
    ```bash
    cp config/config.yaml.example config/config.yaml
    ```
    Edit `config/config.yaml`: set `sam3.checkpoint_path` and `sam3.bpe_path` to your `models/` paths. Optionally set `ocr.engine: "paddleocr"` to use PaddleOCR for text.
2.  **Environment variables (optional):** Create a `.env` file in the root if you use API keys or custom endpoints.

### 9. Notes & Troubleshooting

**Recommended versions**

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | Must be compatible with PyTorch and Paddle |
| PyTorch | 2.x + CUDA to match GPU | Newer GPUs (e.g. Blackwell sm_120) may need cu128; or set `sam3.device: "cpu"` |
| SAM3 weights | `sam3.pt` (not safetensors) | Set `config.sam3.checkpoint_path` to e.g. `models/sam3_ms/sam3.pt` |
| PaddleOCR | PaddlePaddle **3.2.2** + PaddleOCR 3.x | 3.3.0+ has CPU oneDNN bug; pipeline will auto-fallback to Tesseract |
| Tesseract | System install | Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-chi-sim` |
| RMBG | onnxruntime + `models/rmbg/model.onnx` | Optional; use `scripts/setup_rmbg.py` or ModelScope to download |

**Before first run**

- [ ] Copy `config/config.yaml.example` to `config/config.yaml` and set `sam3.checkpoint_path`, `sam3.bpe_path`
- [ ] Place SAM3 weights (e.g. `models/sam3_ms/sam3.pt`) and BPE (`models/bpe_simple_vocab_16e6.txt.gz`) under `models/`
- [ ] Run `scripts/setup_sam3.sh` or follow [docs/SETUP_SAM3.md](docs/SETUP_SAM3.md) to install the SAM3 library
- [ ] Install Tesseract system-wide, or install PaddleOCR and set `ocr.engine: "paddleocr"`

**Common issues**

- **"no kernel image is available for execution on the device"** — GPU arch does not match PyTorch CUDA. Set `sam3.device: "cpu"` in `config.yaml` or upgrade PyTorch to a matching CUDA build (e.g. cu128).
- **"Model file not found at .../models/rmbg/model.onnx"** — RMBG is optional; safe to ignore if you do not need background removal. To enable: `pip install modelscope && python scripts/setup_rmbg.py` or download from [ModelScope RMBG-2.0](https://www.modelscope.cn/models/AI-ModelScope/RMBG-2.0/files) into `models/rmbg/model.onnx`.
- **"PaddleOCR 推理失败…已回退到 Tesseract"** — Paddle/oneDNN incompatibility. Use `paddlepaddle==3.2.2` + `paddleocr`, or set `ocr.engine: "tesseract"`.
- **"请安装 PaddleOCR" / "pytesseract 未安装"** — Install the corresponding OCR stack; for Tesseract only, install system `tesseract-ocr` and `pip install pytesseract`.
- **"Checking connectivity to the model hosters" hangs** — `main.py` sets `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` by default; if it still appears, run `export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` before starting.

## Usage

### Command Line Interface (CLI)

Supports image files (PNG, JPG, BMP, TIFF, WebP). To process a single image:

```bash
python main.py -i input/test_diagram.png
```
The output XML will be saved in the `output/` directory. For batch processing, put images in `input/` and run `python main.py` without `-i`.

### Run and test locally

1. **One-time setup**
   ```bash
   git clone https://github.com/BIT-DataLab/Edit-Banana.git && cd Edit-Banana
   python3 -m venv .venv && source .venv/bin/activate   # Linux/macOS; Windows: .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # or CPU build
   pip install -r requirements.txt
   sudo apt install tesseract-ocr tesseract-ocr-chi-sim   # OCR (or equivalent on your OS)
   ```
   Install the SAM3 library (see [Install SAM3 library](#5-install-sam3-library)) and download [model weights + BPE](#6-download-model-weights-and-bpe-vocab). Then:
   ```bash
   mkdir -p input output
   cp config/config.yaml.example config/config.yaml
   # Edit config/config.yaml: set sam3.checkpoint_path and sam3.bpe_path to your models/ paths
   ```

2. **Test with CLI**
   ```bash
   # Put a diagram image in input/, e.g. input/test.png
   python main.py -i input/test.png
   # Output appears under output/<image_stem>/ (DrawIO XML and intermediates)
   ```

3. **Optional: test the web API**
   ```bash
   python server_pa.py
   # In another terminal:
   curl -X POST http://localhost:8000/convert -F "file=@input/test.png"
   # Or open http://localhost:8000/docs and use the /convert endpoint with a file upload
   ```

## Configuration `config.yaml`

Customize the pipeline behavior in `config/config.yaml`:
*   **sam3**: Adjust score thresholds, NMS (Non-Maximum Suppression) thresholds, max iteration loops.
*   **paths**: Set input/output directories.
*   **dominant_color**: Fine-tune color extraction sensitivity.

## 📌 Development Roadmap
| Feature Module           | Status       | Description                     |
|--------------------------|--------------|---------------------------------|
| Core Conversion Pipeline | ✅ Completed | Full pipeline of segmentation, reconstruction and OCR |
| Intelligent Arrow Connection | ⚠️ In Development | Automatically associate arrows with target shapes |
| DrawIO Template Adaptation | 📍 Planned | Support custom template import |
| Batch Export Optimization | 📍 Planned | Batch export to DrawIO files (.drawio) |
| Local LLM Adaptation | 📍 Planned | Support local VLM deployment, independent of APIs |

## 🤝 Contribution Guidelines
Contributions of all kinds are welcome (code submissions, bug reports, feature suggestions):
1.  Fork this repository
2.  Create a feature branch (`git checkout -b feature/xxx`)
3.  Commit your changes (`git commit -m 'feat: add xxx'`)
4.  Push to the branch (`git push origin feature/xxx`)
5.  Open a Pull Request

Bug Reports: [Issues](https://github.com/BIT-DataLab/Edit-Banana/issues)
Feature Suggestions: [Discussions](https://github.com/BIT-DataLab/Edit-Banana/discussions)



## 🤩 Contributors
Thanks to all developers who have contributed to the project and promoted its iteration!

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

## 📄 License
This project is open-source under the [Apache License 2.0](LICENSE), allowing commercial use and secondary development (with copyright notice retained).

---
## 🌟 Star History

🌟 If this project helps you, please star it to show your support!

[![Star History Chart](https://api.star-history.com/svg?repos=bit-datalab/edit-banana&type=date&legend=top-left)](https://www.star-history.com/#bit-datalab/edit-banana&type=date&legend=top-left)
