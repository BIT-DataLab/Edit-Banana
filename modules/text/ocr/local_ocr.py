"""
本地 OCR 模块（Tesseract）

功能：使用 Tesseract 做文字与版面识别，无需云服务，安装简单。
安装：pip install pytesseract；系统需安装 tesseract-ocr。
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

from .base import TextBlock, OCRResult


class LocalOCR:
    """
    本地 Tesseract OCR 客户端（默认文字识别引擎）
    """

    def __init__(self, lang: str = "eng+chi_sim"):
        if pytesseract is None:
            raise ImportError("请安装 pytesseract: pip install pytesseract，并安装系统 Tesseract。")
        self.lang = lang

    def analyze_image(self, image_path: str) -> OCRResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")

        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size

        data = pytesseract.image_to_data(img, lang=self.lang, output_type=pytesseract.Output.DICT)
        n = len(data["text"])

        text_blocks = []
        i = 0
        while i < n:
            text = (data["text"][i] or "").strip()
            if not text:
                i += 1
                continue

            left = data["left"][i]
            top = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            conf = float(data["conf"][i]) / 100.0 if data["conf"][i] != -1 else 1.0

            polygon = [
                (float(left), float(top)),
                (float(left + w), float(top)),
                (float(left + w), float(top + h)),
                (float(left), float(top + h)),
            ]
            font_size_px = max(h, 12.0)

            block = TextBlock(
                text=text,
                polygon=polygon,
                confidence=conf,
                font_size_px=font_size_px,
                spans=[],
            )
            text_blocks.append(block)
            i += 1

        return OCRResult(
            image_width=width,
            image_height=height,
            text_blocks=text_blocks,
            styles=[],
        )
