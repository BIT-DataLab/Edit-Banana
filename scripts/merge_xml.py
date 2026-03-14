#!/usr/bin/env python3
"""
XML Merging & Orchestration — script entry for the pipeline.

This script runs the full image-to-DrawIO pipeline (segmentation, text extraction, merge).
For config and paths, see config/config.yaml and README.

Usage:
    python scripts/merge_xml.py -i input/test_diagram.png -o output/
    python scripts/merge_xml.py -i input/test.png
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Delegate to main pipeline so behavior stays in sync
from main import load_config, Pipeline


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run pipeline and merge XML (orchestration entry).")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", default=None, help="Output directory (default: config)")
    parser.add_argument("--no-text", action="store_true", help="Skip OCR/text step")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found {args.input}")
        sys.exit(1)

    config = load_config()
    output_dir = args.output or config.get("paths", {}).get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = Pipeline(config)
    result = pipeline.process_image(
        args.input,
        output_dir=output_dir,
        with_refinement=False,
        with_text=not args.no_text,
    )
    if result:
        print(f"Output: {result}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
