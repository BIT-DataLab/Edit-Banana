#!/usr/bin/env python3
"""
OCR & Text Extraction — entry point for text-only pipeline.

Reads an image, runs OCR and formula recognition, writes DrawIO XML for text layers.
Used by the full pipeline; can also be run standalone for text-only output.

Usage:
    python flowchart_text/main.py -i input/diagram.png -o output/
    python flowchart_text/main.py -i input/diagram.png
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Use shared text pipeline
from modules.text.restorer import TextRestorer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OCR & text extraction to DrawIO XML.")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("--formula", choices=["pix2text", "none"], default="none",
                        help="Formula engine (default: none)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    restorer = TextRestorer(formula_engine=args.formula)
    xml_content = restorer.process(args.input)

    out_path = os.path.join(args.output, "text_only.drawio")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"Text XML written: {out_path}")


if __name__ == "__main__":
    main()
