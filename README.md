<p align="center">
  <img src="/static/Edit Banana_Logo.png" width="180" alt="Edit Banana Logo"/>
</p>

<h1 align="center">🍌 Edit Banana</h1>
<h3 align="center">Image to DrawIO (XML) Converter</h3>

<p align="center">
One-click conversion of static diagrams (flowcharts, architecture diagrams, technical schematics) into <b>editable DrawIO (mxGraph) XML files</b>.<br/>
Powered by SAM 3 and multimodal large models, it enables high-fidelity reconstruction that preserves the original diagram details and logical relationships.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white" alt="License"/></a>
  <a href="https://github.com/XiangjianYi/Image2DrawIO"><img src="https://img.shields.io/badge/GitHub-Image2DrawIO-24292F?style=flat-square&logo=github&logoColor=white" alt="GitHub"/></a>
  <a href="https://developer.nvidia.com/cuda-downloads"><img src="https://img.shields.io/badge/GPU-CUDA%20Recommended-76B900?style=flat-square&logo=nvidia" alt="CUDA"/></a>
  <a href="#-join-wechat-group"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?style=flat-square&logo=wechat&logoColor=white" alt="WeChat"/></a>
</p>

---

<p align="center">
  <a href="https://editbanana.anxin6.cn/">
    <img src="https://img.shields.io/badge/🚀%20Try%20Online%20Demo-editbanana.anxin6.cn-FF6B6B?style=for-the-badge&logoColor=white" alt="Try Online Demo"/>
  </a>
</p>

<p align="center">
  👆 <b>Click above or `https://editbanana.anxin6.cn/` to try Edit Banana online!</b> Upload an image, get editable DrawIO XML in seconds.
</p>

---

## 📸 Effect Demonstration
### High-Definition Input-Output Comparison (3 Typical Scenarios)
To intuitively demonstrate the high-fidelity conversion effect, the following provides a one-to-one comparison between 3 groups of "original static images" and "DrawIO editable reconstruction results". All elements can be individually dragged, styled, and modified.

| Scenario No. | Original Static Diagram (Input · Non-editable) | DrawIO Reconstruction Result (Output · Fully Editable) |
|--------------|-----------------------------------------------|--------------------------------------------------------|
| Scenario 1: Basic Flowchart | <img src="/static/demo/original_1.jpg" width="400" alt="Original Diagram 1" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_1.png" width="400" alt="Reconstruction Result 1" style="border: 1px solid #eee; border-radius: 4px;"/> |
| Scenario 2: Multi-level Architecture Diagram | <img src="/static/demo/original_2.png" width="400" alt="Original Diagram 2" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_2.png" width="400" alt="Reconstruction Result 2" style="border: 1px solid #eee; border-radius: 4px;"/> |
| Scenario 3: Technical Schematic | <img src="/static/demo/original_3.jpg" width="400" alt="Original Diagram 3" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_3.png" width="400" alt="Reconstruction Result 3" style="border: 1px solid #eee; border-radius: 4px;"/> |
| Scenario 4: Scientific Formula Diagram | <img src="/static/demo/original_4.jpg" width="400" alt="Original Diagram 4" style="border: 1px solid #eee; border-radius: 4px;"/> | <img src="/static/demo/recon_4.png" width="400" alt="Reconstruction Result 4" style="border: 1px solid #eee; border-radius: 4px;"/> |

> ✨ Conversion Highlights:
> 1.  Preserves the layout logic, color matching, and element hierarchy of the original diagram
> 2.  1:1 restoration of shape stroke/fill and arrow styles (dashed lines/thickness)
> 3.  Accurate text recognition, supporting direct subsequent editing and format adjustment
> 4.  All elements are independently selectable, supporting native DrawIO template replacement and layout optimization

## Key Features

*   **Advanced Segmentation**: Uses **SAM 3 (Segment Anything Model 3)** for state-of-the-art segmentation of diagram elements (shapes, arrows, icons).
*   **Fixed 4-Round VLM Scanning**: A structured, iterative extraction process guided by **Multimodal LLMs (Qwen-VL/GPT-4V)** ensuring no element is left behind:
    1.  **Initial Generic Extraction**: Captures standard shapes and icons.
    2.  **Single Word Scan**: VLM scans blank areas for single objects.
    3.  **Two-Word Scan**: Refines extraction for specific attributes.
    4.  **Phrase Scan**: Captures complex descriptions or grouped objects.
*   **High-Quality OCR**:
    *   **Azure Document Intelligence** for precise text localization.
    *   **Fallback Mechanism**: Automatically switches to VLM-based end-to-end OCR if Azure services are unreachable.
    *   **Mistral Vision/MLLM** for correcting text and converting mathematical formulas to **LaTeX** ($\int f(x) dx$).
    *   **Crop-Guided Strategy**: Extracts text/formula regions and sends high-res crops to LLMs for pixel-perfect recognition.
*   **Smart Background Removal**: Integrated **RMBG-2.0** model to automatically remove backgrounds from icons, pictures, and arrows, ensuring they layer correctly in DrawIO.
*   **Arrow Handling**: Arrows are extracted as transparent images (rather than complex vector paths) to guarantee visual fidelity, handling dashed lines, curves, and complex routing without error.
*   **Vector Shape Recovery**: Standard shapes are converted to native DrawIO vector shapes with accurate fill and stroke colors.
    *   **Supported Shapes**: Rectangle, Rounded Rectangle, Diamond (Decision), Ellipse (Start/End), Cylinder (Database), Cloud, Hexagon, Triangle, Parallelogram, Actor, Title Bar, Text Bubble, Section Panel.
*   **User System**: 
    *   **Registration**: New users receive **30 free credits**.
    *   **Credit System**: Pay-per-use model prevents resource abuse.
*   **Multi-User Concurrency**: Built-in support for concurrent user sessions using a **Global Lock** mechanism for thread-safe GPU access and an **LRU Cache** (Least Recently Used) to persist image embeddings across requests, ensuring high performance and stability.
*   **Web Interface**: A React-based frontend + FastAPI backend for easy uploading and editing.

## Architecture Pipeline

1.  **Input**: Image (PNG/JPG).
2.  **Segmentation (SAM3)**:
    *   Initial pass with standard prompts (rectangle, arrow, icon).
    *   Iterative loop: Analyze unrecognized regions -> Ask MLLM for visual prompts -> Re-run SAM3 mask decoder.
3.  **Element Processing**:
    *   **Vector Shapes**: Color extraction (Fill/Stroke) + Geometry mapping.
    *   **Image Elements (Icons/Arrows)**: Crop -> Padding -> Mask Filtering -> RMBG-2.0 Background Removal -> Base64 Encoding.
4.  **Text Extraction (Parallel)**:
    *   Azure OCR detects text bounding boxes.
    *   High-res crops of text regions are sent to Mistral/LLM.
    *   Latex conversion for formulas.
5.  **XML Generation**:
    *   Merges spatial data from SAM3 and Text OCR.
    *   Applies Z-Index sorting (Layers).
    *   Generates `.drawio.xml` file.

## Project Structure

```
sam3_workflow/
├── config/                 # Configuration files
├── flowchart_text/         # OCR & Text Extraction Module
│   ├── src/                # OCR Source Code (Azure, Mistral, Alignment)
│   └── main.py             # OCR Entry point
├── frontend/               # React Web Application
├── input/                  # [Manual] Input images directory
├── models/                 # [Manual] Model weights (RMBG, SAM3)
│   └── rmbg/               # [Manual] RMBG-2.0
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
*   **Node.js & npm** (for the frontend)
*   **CUDA-capable GPU** (Highly recommended)

### 2. Clone Repository
```bash
git clone https://github.com/XiangjianYi/Image2DrawIO.git
cd Image2DrawIO
```

### 3. Initialize Directory Structure
After cloning, you must **manually create** the following resource directories (ignored by Git):

```bash
# Create input/output directories
mkdir -p input
mkdir -p output
mkdir -p sam3_output

# Create model directories
mkdir -p models/rmbg
```

### 4. Download Model Weights
Download the required models and place them in the correct paths:

| Model | Download | Target Path |
| :--- | :--- | :--- |
| **RMBG-2.0** | [RMBG-2.0](https://modelscope.cn/models/AI-ModelScope/RMBG-2.0/tree/master/onnx) | `models/rmbg/model.onnx` |
| **SAM 3** | https://modelscope.cn/models/facebook/sam3 | `models/sam3.pt` (or as configured) |

> **Note**: For SAM 3 (or the specific segmentation checkpoint used), place the `.pt` file in `models/` and update `config.yaml`.

### 5. Install Dependencies

**Backend:**
```bash
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 6. Configuration

1.  **Config File**: Copy the example config.
    ```bash
    cp config/config.yaml.example config/config.yaml
    ```
2.  **Environment Variables**: Create a `.env` file in the root directory.
    ```env
    AZURE_ENDPOINT=your_azure_endpoint
    AZURE_API_KEY=your_azure_key
    # Add other keys as needed
    ```

## Usage

### 1. Web Interface (Recommended)

Start the Backend:
```bash
python server_pa.py
# Server runs at http://localhost:8000
```

Start the Frontend:
```bash
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173
```
Open your browser, upload an image, and view the result in the embedded DrawIO editor.

### 2. Command Line Interface (CLI)

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

## 💬 Join WeChat Group

Welcome to join our WeChat group to discuss and exchange ideas! Scan the QR code below to join:

<p align="center">
  <img src="/static/wechat_pr.png" width="200" alt="WeChat Group QR Code"/>
  <br/>
  <em>Scan to join the Edit Banana community</em>
</p>

> 💡 If the QR code has expired, please submit an [Issue](https://github.com/XiangjianYi/Image2DrawIO/issues) to request an updated one.

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
| Yang Haotian |  |
| An Minghao |  |
| Yu Mingjie |  |

## 📄 License
This project is open-source under the [Apache License 2.0](LICENSE), allowing commercial use and secondary development (with copyright notice retained).

---
## 🌟 Star History

🌟 If this project helps you, please star it to show your support!

![Star History Chart](https://api.star-history.com/svg?repos=bit-datalab/edit-banana&type=date&legend=top-left)(https://www.star-history.com/#bit-datalab/edit-banana&type=date&legend=top-left)
