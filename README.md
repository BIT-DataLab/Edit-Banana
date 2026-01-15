# Image to DrawIO (XML) Converter

This project implements a sophisticated pipeline to convert images (like flowcharts, diagrams, and technical drawings) into editable DrawIO (mxGraph) XML files. It leverages advanced Computer Vision models and Large Language Models to achieve high-fidelity reconstruction.

## Key Features

*   **Advanced Segmentation**: Uses **SAM 3 (Segment Anything Model 3)** for state-of-the-art segmentation of diagram elements (shapes, arrows, icons).
*   **Intelligent Refinement**: Iterative extraction process guided by **Multimodal LLMs (GPT-4o/Gemini/Claude)** to identify missed elements.
*   **High-Quality OCR**:
    *   **Azure Document Intelligence** for precise text localization.
    *   **Mistral Vision/MLLM** for correcting text and converting mathematical formulas to **LaTeX** ($\int f(x) dx$).
    *   **Crop-Guided Strategy**: Extracts text/formula regions and sends high-res crops to LLMs for pixel-perfect recognition.
*   **Smart Background Removal**: Integrated **RMBG-2.0** model to automatically remove backgrounds from icons, pictures, and arrows, ensuring they layer correctly in DrawIO.
*   **Arrow Handling**: Arrows are extracted as transparent images (rather than complex vector paths) to guarantee visual fidelity, handling dashed lines, curves, and complex routing without error.
*   **Vector Shape Recovery**: Standard shapes (rectangles, diamonds, ellipses, etc.) are converted to native DrawIO vector shapes with accurate fill and stroke colors.
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
├── inputs/                 # Input images directory
├── models/                 # Model weights (RMBG, etc.)
├── output/                 # Results directory
├── sam3/                   # SAM3 Model Library
├── scripts/                # Core Processing Scripts
│   ├── sam3_extractor.py   # Segmentation & Image Extraction Logic
│   ├── merge_xml.py        # XML Merging & Orchestration
│   └── run_all.py          # CLI Entry point
├── server.py               # FastAPI Backend Server
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites
*   Python 3.10+
*   Node.js & npm (for frontend)
*   CUDA-capable GPU (Recommended for SAM3/RMBG)

### Setup

1.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Model Setup**:
    Ensure the following models are placed in the `models/` directory:
    *   `models/rmbg/model.onnx` (RMBG-2.0)
    *   SAM3 checkpoints (configured in `config/config.yaml`)

3.  **Environment Configuration**:
    Create `.env` files in `flowchart_text/.env` and root if necessary.
    ```env
    AZURE_ENDPOINT=your_azure_endpoint
    AZURE_API_KEY=your_azure_key
    MISTRAL_API_KEY=your_mistral_key
    ```

## Usage

### 1. Web Interface (Recommended)

Start the Backend:
```bash
python server.py
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
python scripts/run_all.py --image input/test_diagram.png
```
The output XML will be saved in the `output/` directory.

## Configuration `config.yaml`

Customize the pipeline behavior in `config/config.yaml`:
*   **sam3**: Adjust score thresholds, NMS (Non-Maximum Suppression) thresholds, max iteration loops.
*   **paths**: Set input/output directories.
*   **dominant_color**: Fine-tune color extraction sensitivity.

