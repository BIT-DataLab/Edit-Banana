"""
文字还原器 - 主接口模块

功能：
    将流程图图片中的文字和公式识别并转换为 draw.io XML 格式。

Pipeline 接口：
    from modules.text import TextRestorer
    
    restorer = TextRestorer()
    xml_string = restorer.process("input.png")  # 返回 XML 字符串
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image

# OCR 模块（相对导入）
from .ocr.local_ocr import LocalOCR
from .coord_processor import CoordProcessor
from .xml_generator import MxGraphXMLGenerator

# 四个处理器（相对导入）
from .processors.font_size import FontSizeProcessor
from .processors.font_family import FontFamilyProcessor
from .processors.style import StyleProcessor
from .processors.formula import FormulaProcessor


class TextRestorer:
    """
    文字还原器：协调 OCR、各处理器和输出模块，完成文字还原。
    默认使用本地 Tesseract OCR + 可选 Pix2Text 公式识别。
    """

    def __init__(self, formula_engine: str = "pix2text"):
        """
        Args:
            formula_engine: 公式识别引擎 ('pix2text', 'none')
        """
        self.formula_engine = formula_engine

        self._layout_ocr = None
        self._pix2text_ocr = None

        self.font_size_processor = FontSizeProcessor()
        self.font_family_processor = FontFamilyProcessor()
        self.style_processor = StyleProcessor()
        self.formula_processor = FormulaProcessor()

        self.timing = {
            "text_ocr": 0.0,
            "pix2text_ocr": 0.0,
            "processing": 0.0,
            "total": 0.0,
        }

    @property
    def layout_ocr(self) -> LocalOCR:
        """延迟初始化本地 OCR"""
        if self._layout_ocr is None:
            self._layout_ocr = LocalOCR()
        return self._layout_ocr

    @property
    def pix2text_ocr(self):
        """延迟初始化 Pix2Text OCR"""
        if self._pix2text_ocr is None:
            from .ocr.pix2text import Pix2TextOCR
            self._pix2text_ocr = Pix2TextOCR()
        return self._pix2text_ocr
    
    def process(self, image_path: str) -> str:
        """
        处理图像，返回 XML 字符串（Pipeline 主接口）
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            draw.io 格式的 XML 字符串
        """
        image_path = Path(image_path)
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # 处理图像
        text_blocks = self.process_image(str(image_path))
        
        # 生成 XML
        generator = MxGraphXMLGenerator(
            diagram_name=image_path.stem,
            page_width=image_width,
            page_height=image_height
        )
        
        text_cells = []
        for block in text_blocks:
            geo = block["geometry"]
            cell = generator.create_text_cell(
                text=block["text"],
                x=geo["x"],
                y=geo["y"],
                width=max(geo["width"], 20),
                height=max(geo["height"], 10),
                font_size=block.get("font_size", 12),
                is_latex=block.get("is_latex", False),
                rotation=geo.get("rotation", 0),
                font_weight=block.get("font_weight"),
                font_style=block.get("font_style"),
                font_color=block.get("font_color"),
                font_family=block.get("font_family")
            )
            text_cells.append(cell)
        
        return generator.generate_xml(text_cells)
    
    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        处理图像，返回文字块列表
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            处理后的文字块列表
        """
        total_start = time.time()
        image_path = Path(image_path)
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # Step 1: OCR 识别
        ocr_result, formula_result = self._run_ocr(str(image_path))

        # Step 2: 公式处理（合并 layout OCR 与 Pix2Text）
        processing_start = time.time()

        if formula_result:
            print("\n🔗 公式处理...")
            merged_blocks = self.formula_processor.merge_ocr_results(ocr_result, formula_result)
            text_blocks = self.formula_processor.to_dict_list(merged_blocks)
        else:
            text_blocks = self._ocr_result_to_dict_list(ocr_result)
        
        print(f"   {len(text_blocks)} 个文字块")
        
        # Step 3: 坐标转换
        print("\n📐 坐标转换...")
        coord_processor = CoordProcessor(
            source_width=image_width,
            source_height=image_height
        )
        
        for block in text_blocks:
            polygon = block.get("polygon", [])
            if polygon:
                geometry = coord_processor.polygon_to_geometry(polygon)
                block["geometry"] = geometry
            else:
                block["geometry"] = {"x": 0, "y": 0, "width": 100, "height": 20, "rotation": 0}
        
        # Step 4: 字号处理
        print("\n🔧 字号处理...")
        text_blocks = self.font_size_processor.process(text_blocks)
        
        # Step 5: 字体处理
        print("\n🎨 字体处理...")
        global_font = self._detect_global_font(ocr_result)
        text_blocks = self.font_family_processor.process(text_blocks, global_font=global_font)

        # Step 6: 样式处理（加粗/颜色）
        print("\n🎨 样式处理...")
        ocr_styles = getattr(ocr_result, "styles", [])
        text_blocks = self.style_processor.process(text_blocks, ocr_styles=ocr_styles)
        
        self.timing["processing"] = time.time() - processing_start
        self.timing["total"] = time.time() - total_start
        
        return text_blocks
    
    def restore(
        self,
        image_path: str,
        output_path: str = None,
        save_metadata: bool = True,
        save_debug_image: bool = True
    ) -> str:
        """
        完整还原流程：处理图像并生成 draw.io 文件
        
        Args:
            image_path: 输入图像路径
            output_path: 输出文件路径
            save_metadata: 是否保存元数据
            save_debug_image: 是否生成调试图
            
        Returns:
            输出文件路径
        """
        image_path = Path(image_path)
        
        # 设置输出路径
        if output_path is None:
            output_path = image_path.with_suffix(".drawio")
        else:
            output_path = Path(output_path)
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        print(f"📄 输入: {image_path}")
        print(f"📝 输出: {output_path}")
        print(f"📐 尺寸: {image_width} x {image_height}")
        
        # 处理图像
        text_blocks = self.process_image(str(image_path))
        
        # 生成 XML
        print("\n📄 生成 XML...")
        xml_start = time.time()
        
        generator = MxGraphXMLGenerator(
            diagram_name=image_path.stem,
            page_width=image_width,
            page_height=image_height
        )
        
        text_cells = []
        for block in text_blocks:
            geo = block["geometry"]
            cell = generator.create_text_cell(
                text=block["text"],
                x=geo["x"],
                y=geo["y"],
                width=max(geo["width"], 20),
                height=max(geo["height"], 10),
                font_size=block.get("font_size", 12),
                is_latex=block.get("is_latex", False),
                rotation=geo.get("rotation", 0),
                font_weight=block.get("font_weight"),
                font_style=block.get("font_style"),
                font_color=block.get("font_color"),
                font_family=block.get("font_family")
            )
            text_cells.append(cell)
        
        generator.save_to_file(text_cells, str(output_path))
        
        xml_time = time.time() - xml_start
        self.timing["total"] += xml_time
        
        # 保存元数据
        if save_metadata:
            self._save_metadata(str(image_path), str(output_path), text_blocks, image_width, image_height)
        
        # 生成调试图
        if save_debug_image:
            debug_path = output_path.parent / "debug.png"
            self._generate_debug_image(str(image_path), str(debug_path))
        
        # 打印统计
        self._print_stats(text_blocks)
        
        return str(output_path)
    
    def _run_ocr(self, image_path: str):
        """运行 OCR（本地 Tesseract + 可选 Pix2Text 公式优化）"""
        print("\n📖 Text OCR (Tesseract)...")
        text_start = time.time()
        ocr_result = self.layout_ocr.analyze_image(image_path)
        self.timing["text_ocr"] = time.time() - text_start
        print(f"   {len(ocr_result.text_blocks)} 个文字块 ({self.timing['text_ocr']:.2f}s)")

        formula_result = None

        if self.formula_engine == "pix2text":
            print("\n🔬 公式优化 (Refinement Mode)...")
            refine_start = time.time()
            fixed_count = 0

            processed_indices = set()
            new_blocks_map = {}
            indices_to_remove = set()

            blocks = ocr_result.text_blocks
            i = 0
            while i < len(blocks):
                if i in processed_indices:
                    i += 1
                    continue
                
                # 当前块
                curr_block = blocks[i]
                curr_poly = curr_block.polygon
                
                # 检查是否值得 Refine (初步过滤)
                if not self._should_refine_block(curr_block.text):
                    i += 1
                    continue
                
                # 尝试向后寻找可以合并的块
                group_indices = [i]
                group_polygon = curr_poly
                
                j = i + 1
                while j < len(blocks):
                    next_block = blocks[j]
                    
                    # 距离检查
                    if self._is_spatially_close(group_polygon, next_block.polygon):
                        if self._should_refine_block(next_block.text): 
                            group_indices.append(j)
                            group_polygon = self._merge_polygons(group_polygon, next_block.polygon)
                            j += 1
                        else:
                            break
                    else:
                        break
                
                # 确定最终的识别区域
                target_polygon = group_polygon
                
                # 调用 Pix2Text
                latex_text = self.pix2text_ocr.recognize_region(image_path, target_polygon)
                
                if latex_text and self.formula_processor.is_valid_formula(latex_text):
                    original_text_combined = " ".join([blocks[k].text for k in group_indices])
                    
                    if self._is_refinement_meaningful(original_text_combined, latex_text):
                        cleaned_latex = self.formula_processor.clean_latex(latex_text)
                        
                        import copy
                        new_block = copy.deepcopy(curr_block)
                        new_block.text = f"${cleaned_latex}$"
                        new_block.is_latex = True
                        new_block.polygon = target_polygon
                        new_block.font_family = "Latin Modern Math"
                        
                        if len(group_indices) > 1:
                            print(f"   Refine [Merge {group_indices}]: '{original_text_combined}' -> '${cleaned_latex}$'")
                            indices_to_remove.update(group_indices)
                            new_blocks_map[i] = new_block
                        else:
                            print(f"   Refine [{i}]: '{curr_block.text}' -> '${cleaned_latex}$'")
                            curr_block.text = f"${cleaned_latex}$"
                            curr_block.is_latex = True
                            curr_block.font_family = "Latin Modern Math"
                            fixed_count += 1
                        
                        processed_indices.update(group_indices)
                        i = j
                        continue
                
                i += 1
            
            if indices_to_remove:
                final_blocks = []
                for idx, block in enumerate(blocks):
                    if idx in new_blocks_map:
                        final_blocks.append(new_blocks_map[idx])
                        fixed_count += 1
                    elif idx not in indices_to_remove:
                        final_blocks.append(block)
                ocr_result.text_blocks = final_blocks

            self.timing["pix2text_ocr"] = time.time() - refine_start
            print(f"   优化了 {fixed_count} 个公式块 ({self.timing['pix2text_ocr']:.2f}s)")

            formula_result = None

        else:
            print("\n⏭️  跳过公式识别")

        return ocr_result, formula_result

    def _should_refine_block(self, text: str) -> bool:
        """判断是否需要尝试 Refine"""
        if not text: return False
        
        if '?' in text or '？' in text or '(?)' in text:
            return True
        
        words = text.split()
        if len(words) > 8: return False
        
        import re
        if re.match(r'^[a-zA-Z\s\-,.:!\\\'"]+$', text):
            if len(text) < 4: 
                return True
            return False 
            
        return True

    def _is_refinement_meaningful(self, original: str, new_latex: str) -> bool:
        """判断 Refine 结果是否有实质性改变"""
        import re
        
        core_latex = re.sub(r'\\(mathbf|mathrm|textit|text|boldsymbol|mathcal|mathscr)\{([^\}]+)\}', r'\2', new_latex)
        core_latex = re.sub(r'\s|~', '', core_latex)
        core_latex = core_latex.replace('$', '')
        
        core_original = re.sub(r'\s', '', original)
        
        if core_latex == core_original:
            return False
            
        return True

    def _is_spatially_close(self, poly1, poly2) -> bool:
        """判断两个多边形是否在空间上接近"""
        def get_bbox(p):
            xs, ys = [pt[0] for pt in p], [pt[1] for pt in p]
            return min(xs), min(ys), max(xs), max(ys)
        
        x1_min, y1_min, x1_max, y1_max = get_bbox(poly1)
        x2_min, y2_min, x2_max, y2_max = get_bbox(poly2)
        
        h1, h2 = y1_max - y1_min, y2_max - y2_min
        ref_h = max(h1, h2)
        
        y_overlap = min(y1_max, y2_max) - max(y1_min, y2_min)
        is_y_aligned = y_overlap > -ref_h * 0.5 
        
        if is_y_aligned:
            x_dist = max(0, x2_min - x1_max) if x1_min < x2_min else max(0, x1_min - x2_max)
            if x_dist < ref_h * 1.2:
                h_ratio = min(h1, h2) / max(h1, h2)
                if h_ratio > 0.6:
                    return True

        x_overlap = min(x1_max, x2_max) - max(x1_min, x2_min)
        wmin = min(x1_max - x1_min, x2_max - x2_min)
        
        if x_overlap > wmin * 0.2: 
            y_dist = max(0, y2_min - y1_max) if y1_min < y2_min else max(0, y1_min - y2_max)
            if y_dist < ref_h * 0.5:
                return True
                
        return False

    def _merge_polygons(self, poly1, poly2):
        """合并两个多边形"""
        xs = [p[0] for p in poly1] + [p[0] for p in poly2]
        ys = [p[1] for p in poly1] + [p[1] for p in poly2]
        min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    
    def _ocr_result_to_dict_list(self, ocr_result) -> List[Dict[str, Any]]:
        """将 OCR 结果转换为字典列表"""
        return [
            {
                "text": block.text,
                "polygon": block.polygon,
                "confidence": getattr(block, "confidence", 1.0),
                "font_size_px": block.font_size_px,
                "is_latex": getattr(block, "is_latex", False),
                "font_family": getattr(block, "font_family", getattr(block, "font_name", None)),
                "font_weight": getattr(block, "font_weight", None),
                "font_style": getattr(block, "font_style", None),
                "font_color": getattr(block, "font_color", None),
                "is_bold": getattr(block, "is_bold", False),
                "is_italic": getattr(block, "is_italic", False),
                "spans": getattr(block, "spans", []),
            }
            for block in ocr_result.text_blocks
        ]

    def _detect_global_font(self, ocr_result) -> str:
        """检测全局主字体"""
        if not ocr_result.text_blocks:
            return "Arial"

        def get_area(block):
            polygon = block.polygon
            if not polygon or len(polygon) < 4:
                return 0
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        best_block = max(ocr_result.text_blocks, key=get_area)
        font = getattr(best_block, "font_name", None)
        
        if font:
            print(f"   ✨ 识别到主字体: {font}")
            return font
        
        return "Arial"
    
    def _save_metadata(self, image_path: str, output_path: str, text_blocks: List[Dict], 
                       image_width: int, image_height: int):
        """保存元数据"""
        import json
        from datetime import datetime
        
        metadata_path = Path(output_path).parent / "metadata.json"
        
        font_stats = {}
        for block in text_blocks:
            font = block.get("font_family", "unknown")
            font_stats[font] = font_stats.get(font, 0) + 1
        
        metadata = {
            "version": "3.0",
            "generated_at": datetime.now().isoformat(),
            "input": {"path": image_path, "width": image_width, "height": image_height},
            "output": {"drawio_path": output_path},
            "mode": f"local+{self.formula_engine}",
            "timing": self.timing,
            "statistics": {
                "total_cells": len(text_blocks),
                "text_cells": sum(1 for b in text_blocks if not b.get("is_latex")),
                "formula_cells": sum(1 for b in text_blocks if b.get("is_latex")),
                "fonts": font_stats
            },
            "text_blocks": [
                {
                    "id": i + 1,
                    "text": block["text"][:100],
                    "position": block["geometry"],
                    "style": {
                        "font_size": block.get("font_size"),
                        "font_family": block.get("font_family"),
                        "font_weight": block.get("font_weight"),
                        "font_color": block.get("font_color"),
                        "is_formula": block.get("is_latex", False)
                    }
                }
                for i, block in enumerate(text_blocks)
            ]
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"   元数据已保存: {metadata_path}")
    
    def _generate_debug_image(self, image_path: str, output_path: str):
        """生成调试图"""
        try:
            # 简单实现：复制原图作为调试图
            from PIL import Image
            img = Image.open(image_path)
            img.save(output_path)
        except Exception as e:
            print(f"   ⚠️ 调试图生成失败: {e}")
    
    def _print_stats(self, text_blocks: List[Dict]):
        """打印统计信息"""
        print(f"\n⏱️  耗时:")
        print(f"   Text OCR:  {self.timing['text_ocr']:.2f}s")
        print(f"   Pix2Text:  {self.timing['pix2text_ocr']:.2f}s")
        print(f"   处理:      {self.timing['processing']:.2f}s")
        print(f"   总计:      {self.timing['total']:.2f}s")
        
        print(f"\n✅ 完成！共 {len(text_blocks)} 个文本单元格")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python restorer.py <image_path> [output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    restorer = TextRestorer()
    restorer.restore(image_path, output_path)
