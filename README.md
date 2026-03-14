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
  👆 <b>Click above or https://editbanana.anxin6.cn/ to try Edit Banana online!</b> Upload an image or pdf, get <b>editable DrawIO (XML) or PPTX</b> in seconds. 
  <b>Please note</b>: Our GitHub repository currently trails behind our web-based service. For the most up-to-date features and performance, we recommend using our web platform.
</p>

## 💬 Join WeChat Group

Welcome to join our WeChat group to discuss and exchange ideas! Scan the QR code below to join:

<p align="center">
  <img src="/static/wechat_20260309.png" width="70%" alt="WeChat Group QR Code"/>
  <br/>
  <em>Scan to join the Edit Banana community</em>
</p>

> 💡 If the QR code has expired, please submit an [Issue](https://github.com/XiangjianYi/Image2DrawIO/issues) to request an updated one.

---

## 📸 Effect Demonstration
### High-Definition Input-Output Comparison (3 Typical Scenarios)
To demonstrate the high-fidelity conversion effect, we provides one-to-one comparisons between 3 scenarios of "original static formats" and "editable reconstruction results". All elements can be individually dragged, styled, and modified.

#### Scenario 1: Figures to Drawio(xml, svg, pptx)

| Example No. | Original Static Diagram (Input · Non-editable) | DrawIO Reconstruction Result (Output · Fully Editable) |
|--------------|-----------------------------------------------|--------------------------------------------------------|
| Example 1: Basic Flowchart | <img src="/static/demo/original_1.jpg" width="400" alt="Original Diagram 1" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_1.png" width="400" alt="Reconstruction Result 1" style="border: 1px solid #eee; border-radius: 4px;"/> |
| Example 2: Multi-level Architecture Diagram | <img src="/static/demo/original_2.png" width="400" alt="Original Diagram 2" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_2.png" width="400" alt="Reconstruction Result 2" style="border: 1px solid #eee; border-radius: 4px;"/> |
| Example 3: Technical Schematic | <img src="/static/demo/original_3.jpg" width="400" alt="Original Diagram 3" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_3.png" width="400" alt="Reconstruction Result 3" style="border: 1px solid #eee; border-radius: 4px;"/> |
| Example 4: Scientific Formula Diagram | <img src="/static/demo/original_4.jpg" width="400" alt="Original Diagram 4" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_4.png" width="400" alt="Reconstruction Result 4" style="border: 1px solid #eee; border-radius: 4px;"/> |

#### Scenario 2: PDF to PPTX


#### Scenario 3: Human in the Loop Modification

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

1.  **Input**: Image (PNG/JPG) or PDF.
2.  **Segmentation (SAM3)**: Using our fine-tuned SAM3 mask decoder.
4.  **Text Extraction (Parallel)**:
    *   Local OCR (Tesseract) detects text bounding boxes.
    *   High-res crops of text/formula regions are sent to Pix2Text for LaTeX conversion.
5.  **XML/PPTX Generation**: Merging spatial data from our fine-tuned SAM3 and text OCR results.

## Project Structure

```
sam3_workflow/
├── config/                 # Configuration files
├── flowchart_text/         # OCR & Text Extraction Module
│   ├── src/                # OCR source (local Tesseract, Pix2Text, alignment)
│   └── main.py             # OCR Entry point
├── input/                  # [Manual] Input images directory
├── models/                 # [Manual] Model weights (SAM3)
├── output/                 # [Manual] Results directory
├── sam3/                   # SAM3 Model Library
├── scripts/                # Utility Scripts
│   └── merge_xml.py        # XML Merging & Orchestration
├── main.py                 # CLI Entry point (Modular Pipeline)
├── server_pa.py            # FastAPI Backend Server (Service-based)
└── requirements.txt        # Python dependencies
```

## Installation & Setup

Follow these steps to set up the project locally.

### 1. Prerequisites
*   **Python 3.10+**
*   **CUDA-capable GPU** (Highly recommended)

### 2. Clone Repository
```bash
git clone https://github.com/BIT-DataLab/Edit-Banana.git
cd Image2DrawIO
```

### 3. Initialize Directory Structure
After cloning, you must **manually create** the following resource directories (ignored by Git):

```bash
# Create input/output directories
mkdir -p input
mkdir -p output
mkdir -p sam3_output
```

### 4. Download Model Weights
Download the required models and place them in the correct paths:

| Model | Download | Target Path |
| :--- | :--- | :--- |
| **SAM 3** | https://modelscope.cn/models/facebook/sam3 | `models/sam3.pt` (or as configured) |

> **Note**: For SAM 3 (or the specific segmentation checkpoint used), place the `.pt` file in `models/` and update `config.yaml`.

### 5. Install Dependencies

**Backend:**
```bash
pip install -r requirements.txt
```

**Tesseract (for text OCR):** Install the Tesseract engine on your system (required for local text recognition). Example on Ubuntu:
```bash
sudo apt install tesseract-ocr tesseract-ocr-chi-sim
```

### 6. Configuration

1.  **Config File**: Copy the example config.
    ```bash
    cp config/config.yaml.example config/config.yaml
    ```
2.  **Environment Variables** (optional): Create a `.env` file in the root directory if your setup requires API keys or endpoints.

## Usage

### Command Line Interface (CLI)

To process a single image:

```bash
python main.py -i input/test_diagram.png
```
The output XML will be saved in the `output/` directory.

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

Bug Reports: [Issues](https://github.com/XiangjianYi/Image2DrawIO/issues)
Feature Suggestions: [Discussions](https://github.com/XiangjianYi/Image2DrawIO/discussions)



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

![Star History Chart](https://api.star-history.com/svg?repos=bit-datalab/edit-banana&type=date&legend=top-left)(https://www.star-history.com/#bit-datalab/edit-banana&type=date&legend=top-left)
