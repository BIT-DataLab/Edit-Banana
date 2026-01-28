# Image to DrawIO (XML) — Algorithm Pipeline

Convert static diagrams (flowcharts, architecture diagrams, schematics) into **editable DrawIO (mxGraph) XML** using SAM 3 and optional OCR/VLM. This repository contains **only the algorithm pipeline** (no web frontend or API server).

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-2F80ED?style=flat-square&logo=apache&logoColor=white)](LICENSE)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%20Recommended-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-downloads)

## Features

* **SAM 3 segmentation**: Extract shapes, arrows, icons, and background regions via prompt groups (see `prompts/` and `config/config.yaml`).
* **Element processing**: Vector shapes (color extraction), arrows (path or image fallback), icons/pictures (background removal, base64).
* **Text (optional)**: OCR + layout/format; can be disabled with `--no-text`.
* **Pipeline**: Single entry point `main.py` — process one image or a directory.

## Project Structure

```
.
├── config/             # config.yaml.example → config.yaml
├── modules/            # Algorithm modules
│   ├── sam3_info_extractor.py
│   ├── arrow_processor.py
│   ├── basic_shape_processor.py
│   ├── icon_picture_processor.py
│   ├── xml_merger.py
│   ├── text/           # OCR & text layout
│   └── utils/
├── prompts/            # SAM3 prompt lists (arrow, shape, image, background)
├── sam3_service/       # SAM3/RMBG client (inference)
├── main.py             # CLI & pipeline entry
├── requirements.txt
└── README.md
```

## Installation

### 1. Prerequisites

* Python 3.10+
* CUDA-capable GPU (recommended for SAM3)

### 2. Directories

```bash
mkdir -p input output sam3_output
mkdir -p models/rmbg
```

### 3. Model Weights

| Model     | Path (example)        |
|----------|------------------------|
| RMBG-2.0 | `models/rmbg/model.onnx` |
| SAM 3    | Path set in `config.yaml` (`sam3.checkpoint_path`) |

Place weights as required; set `config/config.yaml` accordingly.

### 4. Config

```bash
cp config/config.yaml.example config/config.yaml
# Edit config.yaml: sam3.checkpoint_path, sam3.bpe_path, paths, etc.
```

### 5. Dependencies

```bash
pip install -r requirements.txt
# Install PyTorch (and SAM3 code/weights) separately for your environment.
```

## Usage (CLI)

Single image:

```bash
python main.py -i input/diagram.png
```

Output directory:

```bash
python main.py -i input/diagram.png -o output/custom/
```

Batch (all images in `input/`):

```bash
python main.py
```

Skip text processing:

```bash
python main.py -i input/diagram.png --no-text
```

Quality refinement (evaluate + refine):

```bash
python main.py -i input/diagram.png --refine
```

Output XML is written under the configured output directory (default `output/`).

## Configuration

In `config/config.yaml`:

* **sam3**: `checkpoint_path`, `bpe_path`, score thresholds, min area.
* **prompt_groups**: Per-group thresholds and priority (prompt words are in `prompts/*.py`).
* **paths**: `input_dir`, `output_dir`, etc.

## License

Apache License 2.0.
