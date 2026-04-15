<p align="center">
  <img src="/static/banana.jpg" width="180" alt="Edit Banana Logo"/>
</p>

<h1 align="center">🍌 Edit Banana</h1>
<p align="center">
  <a href="README_CN.md">中文</a> | English
</p>
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
</p>

> [!WARNING]
> **Please note**: Our GitHub repository currently trails behind our web-based service. For the most up-to-date features and performance, we recommend using our web platform.

> [!NOTE]
> **Known limitation — multi-panel / schematic figures.** Recent updates (v2 prompts: 19 → 78; new hierarchical layer assignment; [Stage 8 vector export to SVG/PDF](https://github.com/Sdamirsa/Edit-Banana/commit/d6c445a)) significantly improved single-element extraction, but the pipeline still under-understands complex **multi-panel scientific schematics**. Detection is per-element; the global semantics — which arrow connects which box across panel boundaries, which legend swatch labels which plot — is not yet modeled.
>
> **Roadmap directions on this challenge:**
> 1. **Two-pass extraction with panel splitting.** First detect sub-figure panels (Stage 8's `section_detector` already produces panel candidates), split the source image into per-panel crops, then recursively run the full pipeline on each crop. The orchestration loop is the missing piece.
> 2. **Smart per-element-type margin padding** around cropped rasters. Tight bboxes clip strokes; loose ones bleed neighbors. A per-type heuristic (icon vs. photo vs. schematic illustration) would help, but the logic is hard to pin down cleanly.
---
## 💬 Join WeChat Group

Welcome to join our WeChat group to discuss and exchange ideas! Scan the QR code below to join:

<p align="center">
  <img src="/static/wechatGroup.jpg" width="70%" alt="WeChat Group QR Code"/>
  <br/>
  <em>Scan to join the Edit Banana community</em>
</p>

> [!TIP]
> If the QR code has expired, please submit an [Issue](https://github.com/BIT-DataLab/Edit-Banana/issues) to request an updated one.

---
## 📑 Table of Contents

- [📸 Effect Demonstration](#-effect-demonstration)
- [🚀 Key Features](#-key-features)
- [🛠️ Architecture Pipeline](#️-architecture-pipeline)
- [📂 Project Structure](#-project-structure)
- [📦 Installation & Setup](#-installation--setup)
- [🔤 Usage](#-usage)
- [⚙️ Configuration](#️-configuration)
- [📌 Development Roadmap](#-development-roadmap)
- [💬 Join WeChat Group](#-join-wechat-group)
- [🤝 Contribution Guidelines](#-contribution-guidelines)
- [🤩 Contributors](#-contributors)
- [📄 License](#-license)
- [🌟 Star History](#-star-history)

---

## 📸 Effect Demonstration

### High-Definition Input-Output Comparison (4 Typical Scenarios)

To demonstrate the high-fidelity conversion effect, we provides one-to-one comparisons between 4 scenarios of "original static formats" and "editable reconstruction results". All elements can be individually dragged, styled, and modified.

#### Scenario 1: Figures to DrawIO

| 🔒 Original Static Diagram (Input · Non-editable) | 🔓 DrawIO Reconstruction Result (Output · Fully Editable) |
|:---:|:---:|
| <br><b>Example 1: Basic Flowchart</b><br><br><img src="/static/demo/original_1.jpg" width="450" alt="Original Diagram 1" style="border: 1px solid #eee; border-radius: 8px;"/> | <br><b>✨ Editable Flowchart</b><br><br><img src="/static/demo/recon_1.png" width="450" alt="Reconstruction Result 1" style="border: 1px solid #eee; border-radius: 8px;"/> |
| <br><b>Example 2: Multi-level Architecture</b><br><br><img src="/static/demo/original_2.png" width="450" alt="Original Diagram 2" style="border: 1px solid #eee; border-radius: 8px;"/> | <br><b>✨ Editable Architecture</b><br><br><img src="/static/demo/recon_2.png" width="450" alt="Reconstruction Result 2" style="border: 1px solid #eee; border-radius: 8px;"/> |
| <br><b>Example 3: Technical Schematic</b><br><br><img src="/static/demo/original_3.jpg" width="450" alt="Original Diagram 3" style="border: 1px solid #eee; border-radius: 8px;"/> | <br><b>✨ Editable Schematic</b><br><br><img src="/static/demo/recon_3.png" width="450" alt="Reconstruction Result 3" style="border: 1px solid #eee; border-radius: 8px;"/> |
| <br><b>Example 4: Scientific Formula</b><br><br><img src="/static/demo/original_4.jpg" width="450" alt="Original Diagram 4" style="border: 1px solid #eee; border-radius: 8px;"/> | <br><b>✨ Editable Formula</b><br><br><img src="/static/demo/recon_4.png" width="450" alt="Reconstruction Result 4" style="border: 1px solid #eee; border-radius: 8px;"/> |

#### Scenario 2: Human in the Loop Modification

<div align="center">

<br>
<img src="static/demo/cut.gif" width="90%"/>
<br><sub>✨ Manual repair</sub>

<br><br>
<img src="static/demo/save.gif" width="90%"/>
<br><sub>✨ Save locally</sub>

</div>

> [!NOTE]
> **✨ Conversion Highlights:**
> 1. Preserves the layout logic, color matching, and element hierarchy of the original diagram.
> 2. 1:1 restoration of shape stroke/fill and arrow styles (dashed lines/thickness).
> 3. Accurate text recognition, supporting direct subsequent editing and format adjustment.
> 4. All elements are independently selectable, supporting native DrawIO template replacement and layout optimization.

## 🚀 Key Features

- **Advanced Segmentation**: Using our fine-tuned **SAM 3 (Segment Anything Model 3)** for segmentation of diagram elements.
- **Fixed Multi-Round VLM Scanning**: An extraction process guided by **Multimodal LLMs**.
- **Text Recognition**:
  - **Local OCR** for text localization; easy to install, runs offline.
  - **Pix2Text** for mathematical formula recognition and **LaTeX** conversion .
  - **Crop-Guided Strategy**: Extracts text/formula regions and sends high-res crops to the formula engine.

- **User System**:

  - **Registration**: New users receive **10 free credits**.
  - **Credit System**: Pay-per-use model prevents resource abuse.
  - **Multi-User Concurrency**: Built-in support for concurrent user sessions using a **Global Lock** mechanism for thread-safe GPU access and an **LRU Cache** (Least Recently Used) to persist image embeddings across requests, ensuring high performance and stability.

---

## 🛠️ Architecture Pipeline

1.  **Input**: Image (PNG/JPG/BMP/TIFF/WebP).
2.  **Segmentation (SAM3)**: Using our fine-tuned SAM3 mask decoder.
3.  **Text Extraction (Parallel)**:
    *   Local OCR (Tesseract) detects text bounding boxes.
    *   High-res crops of text/formula regions are sent to Pix2Text for LaTeX conversion.
4.  **DrawIO XML Generation**: Merging spatial data from SAM3 and text OCR results.

---

## 📂 Project Structure

  <details>

  <summary><b>Click to expand project structure </b></summary>

  ```text
  Edit-Banana/
  ├── config/               # Configuration files (copy config.yaml.example → config.yaml)
  ├── flowchart_text/       # OCR & Text Extraction Module (standalone entry)
  │   ├── src/
  │   └── main.py             # OCR-only entry point
  ├── input/                # [Manual] Input images directory
  ├── models/               # [Manual] Model weights (SAM3) and optional BPE vocab
  ├── output/               # [Manual] Results directory
  ├── sam3/                 # SAM3 library (see Installation: install from facebookresearch/sam3)
  ├── sam3_service/         # SAM3 HTTP service (optional, for multi-process deployment)
  ├── scripts/              # Setup and utility scripts
  │   ├── setup_sam3.sh       # Install SAM3 lib and copy BPE to models/
  │   ├── setup_rmbg.py       # Download RMBG model from ModelScope
  │   └── merge_xml.py        # XML merge utilities
  ├── main.py               # CLI entry (modular pipeline)
  ├── server_pa.py          # FastAPI backend server
  └── requirements.txt      # Python dependencies
  ```

  </details>

  ---

## 📦 Installation & Setup

Follow these core phases to set up the project locally.

### Phase 1: Environment & Base Setup

Configure your base environment and directory structure.

#### 1. Prerequisites & Environment

- Python 3.10+** & CUDA-capable GPU (Highly recommended)
- Install PyTorch with CUDA support (e.g., for CUDA 11.8):

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

#### 2. Clone Repository & Init Directories

  ```bash
  git clone https://github.com/BIT-DataLab/Edit-Banana.git
  cd Edit-Banana
  mkdir -p input output sam3_output
  ```

### Phase 2: Models & Core Dependencies

Next, install the required packages and download necessary model weights (which should be placed in models/ and not committed).

#### 1. Base Dependencies

```bash
pip install -r requirements.txt
```

#### 2. SAM3 & Model Assets

- SAM3 Library & BPE: 
Run `bash scripts/setup_sam3.sh`to install the lib and copy the BPE vocab to `models/`.
Verify with:

  ```bash
  python -c "from sam3.model_builder import build_sam3_image_model; print('OK')"
  ```

- SAM3 Weights: Download sam3.pt from [ModelScope](https://modelscope.cn/models/facebook/sam3)  or [Hugging Face](https://huggingface.co/facebook/sam3) and place it under `models/sam3_ms`.

- Text Local OCR (Tesseract): 

  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-chi-sim
  ```

<details> <summary><b>🧩 Optional Capabilities (OCR Engine, Formula, RMBG) - Click to expand</b></summary>

- PaddleOCR (Alternative/Better for mixed text): Use paddlepaddle==3.2.2 (avoiding 3.3.0 bug).

  ```bash
  pip install paddlepaddle==3.2.2 paddleocr.
  ```

- Formula (Pix2Text): 

  ```bash
  pip install pix2text onnxruntime-gpu.
  ```

- Background Removal (RMBG): `pip install onnxruntime modelscope` then run `python scripts/setup_rmbg.py`.
</details>

### Phase 3: Configuration & Troubleshooting

#### 1. Final Configuration

Copy the example config and adjust the asset paths:

  ```bash
  cp config/config.yaml.example config/config.yaml
  ```

Edit `config.yaml` to ensure `sam3.checkpoint_path` and `sam3.bpe_path` match your `models/ locations`.

<details> <summary><b>🛠️ Before First Run Checklist & Troubleshooting - Click to expand</b></summary>

**Checklist**:

- [ ]  Config files copied and model paths set in `config.yaml`
- [ ] SAM3 weights (`sam3.pt`) and BPE vocab placed under `models/`
- [ ] Extracted SAM3 library via `scripts/setup_sam3.sh`
 Tesseract or PaddleOCR installed

**Common Issues**:

- "no kernel image is available...": GPU arch mismatch. Upgrade PyTorch or set `sam3.device: "cpu"`.
- "Model file not found at ...rmbg/...": RMBG is optional. Enable by downloading via script.
- "PaddleOCR inference failed...": Use `paddlepaddle==3.2.2` or fallback to Tesseract.

</details>

---

## 🔤 Usage

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

   Install the SAM3 library and download model weights + BPE. Then:

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

---

## ⚙️ Configuration

Customize the pipeline behavior in `config/config.yaml`:

- **sam3**: Adjust score thresholds, NMS (Non-Maximum Suppression) thresholds, max iteration loops.

- **paths**: Set input/output directories.

- **dominant_color**: Fine-tune color extraction sensitivity.

---

## 📌 Development Roadmap

| Feature Module           | Status       | Description                     |
|--------------------------|--------------|---------------------------------|
| Core Conversion Pipeline | ✅ Completed | Full pipeline of segmentation, reconstruction and OCR |
| Intelligent Arrow Connection | ⚠️ In Development | Automatically associate arrows with target shapes |
| DrawIO Template Adaptation | 📍 Planned | Support custom template import |
| Batch Export Optimization | 📍 Planned | Batch export to DrawIO files (.drawio) |
| Local LLM Adaptation | 📍 Planned | Support local VLM deployment, independent of APIs |

---


## 🤝 Contribution Guidelines

Contributions of all kinds are welcome (code submissions, bug reports, feature suggestions):

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/xxx`)
3. Commit your changes (`git commit -m 'feat: add xxx'`)
4. Push to the branch (`git push origin feature/xxx`)
5. Open a Pull Request

Bug Reports: [Issues](https://github.com/BIT-DataLab/Edit-Banana/issues)
Feature Suggestions: [Discussions](https://github.com/BIT-DataLab/Edit-Banana/discussions)

---

## 🤩 Contributors

Thanks to all developers who have contributed to the project and promoted its iteration!

<details>
<summary><b>View Contributors List</b></summary>

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

## 📄 License

This project is open-source under the [Apache License 2.0](LICENSE), allowing commercial use and secondary development (with copyright notice retained).

---

## 🌟 Star History

🌟 If this project helps you, please star it to show your support!

[![Star History Chart](https://api.star-history.com/svg?repos=bit-datalab/edit-banana&type=date&legend=top-left)](https://www.star-history.com/#bit-datalab/edit-banana&type=date&legend=top-left)
