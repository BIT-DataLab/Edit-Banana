#!/usr/bin/env python3
"""
Edit Banana — CLI entry. Image/PDF to editable DrawIO XML.

Pipeline: Input -> Segmentation (SAM3) -> Text Extraction (OCR) -> XML/PPTX generation.
See README for setup (config, models, env).

Usage:
    python main.py -i input/test.png
    python main.py
    python main.py -i input/test.png -o output/custom/
    python main.py -i input/test.png --refine
    python main.py -i input/test.png --no-text
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules import (
    # 核心处理器
    Sam3InfoExtractor,
    IconPictureProcessor,
    BasicShapeProcessor,
    ArrowProcessor,
    XMLMerger,
    MetricEvaluator,
    RefinementProcessor,
    
    # 文字处理（已整合到 modules/text/）
    TextRestorer,
    
    # 上下文和数据类型
    ProcessingContext,
    ProcessingResult,
    ElementInfo,
    LayerLevel,
    get_layer_level,
)

# 导入分组枚举，方便按需提取
from modules.sam3_info_extractor import PromptGroup

# 超分模型（可选依赖）
from modules.icon_picture_processor import UpscaleModel, SPANDREL_AVAILABLE

# 文字处理模块可用性标记（依赖 ocr/coord_processor 等，缺失时为 False）
TEXT_MODULE_AVAILABLE = TextRestorer is not None

# 条件超分阈值配置
UPSCALE_MIN_DIMENSION = 800  # 原图短边小于此值时触发超分


# ======================== config ========================
def load_config() -> dict:
    """Load config/config.yaml."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"警告：配置文件不存在 {config_path}，使用默认配置")
        return {
            'paths': {
                'input_dir': './input',
                'output_dir': './output',
            }
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ======================== pipeline ========================
class Pipeline:
    """Runs segmentation, text extraction, and XML merge (see README pipeline)."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self._text_restorer = None
        self._sam3_extractor = None
        self._icon_processor = None
        self._shape_processor = None
        self._arrow_processor = None
        self._xml_merger = None
        self._metric_evaluator = None
        self._refinement_processor = None
        self._upscale_model = None
        
        # 超分配置
        self._upscale_min_dimension = self.config.get('upscale', {}).get('min_dimension', UPSCALE_MIN_DIMENSION)
        self._upscale_enabled = self.config.get('upscale', {}).get('enabled', True)
    
    @property
    def text_restorer(self):
        """OCR/text step; None if deps missing."""
        if self._text_restorer is None and TextRestorer is not None:
            self._text_restorer = TextRestorer(formula_engine='none')
        return self._text_restorer
    
    @property
    def sam3_extractor(self) -> Sam3InfoExtractor:
        if self._sam3_extractor is None:
            self._sam3_extractor = Sam3InfoExtractor()
        return self._sam3_extractor
    
    @property
    def icon_processor(self) -> IconPictureProcessor:
        if self._icon_processor is None:
            self._icon_processor = IconPictureProcessor()
        return self._icon_processor
    
    @property
    def shape_processor(self) -> BasicShapeProcessor:
        if self._shape_processor is None:
            self._shape_processor = BasicShapeProcessor()
        return self._shape_processor
    
    @property
    def arrow_processor(self) -> ArrowProcessor:
        if self._arrow_processor is None:
            self._arrow_processor = ArrowProcessor()
        return self._arrow_processor
    
    @property
    def xml_merger(self) -> XMLMerger:
        if self._xml_merger is None:
            self._xml_merger = XMLMerger()
        return self._xml_merger
    
    @property
    def metric_evaluator(self) -> MetricEvaluator:
        if self._metric_evaluator is None:
            self._metric_evaluator = MetricEvaluator()
        return self._metric_evaluator
    
    @property
    def refinement_processor(self) -> RefinementProcessor:
        if self._refinement_processor is None:
            self._refinement_processor = RefinementProcessor()
        return self._refinement_processor
    
    @property
    def upscale_model(self) -> UpscaleModel:
        """Optional upscale (lazy)."""
        if self._upscale_model is None:
            self._upscale_model = UpscaleModel(model_path=None)  # 使用默认路径
        return self._upscale_model
    
    def _preprocess_image(self, image_path: str, output_dir: str) -> tuple:
        """Optional upscale when image is small. Returns (path, was_upscaled, scale)."""
        from PIL import Image

        if not self._upscale_enabled:
            return image_path, False, 1.0
        
        # 检查依赖是否可用
        if not SPANDREL_AVAILABLE:
            print("   [预处理] 超分依赖未安装，跳过")
            return image_path, False, 1.0
        
        # 读取原图尺寸
        with Image.open(image_path) as img:
            width, height = img.size
            min_dim = min(width, height)
        
        # 判断是否需要超分
        if min_dim >= self._upscale_min_dimension:
            print(f"   [预处理] 原图尺寸 {width}x{height}，无需超分")
            return image_path, False, 1.0
        
        print(f"   [预处理] 原图尺寸 {width}x{height} < {self._upscale_min_dimension}，启动超分...")
        
        # 加载超分模型
        try:
            self.upscale_model.load()
            
            if self.upscale_model._model is None:
                print("   [预处理] 超分模型不可用，跳过")
                return image_path, False, 1.0
            
            # 执行超分
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                upscaled = self.upscale_model.upscale(img_rgb)
            
            # 保存超分后的图片
            upscaled_path = os.path.join(output_dir, "upscaled_input.png")
            upscaled.save(upscaled_path)
            
            new_width, new_height = upscaled.size
            scale_factor = new_width / width
            
            print(f"   [预处理] 超分完成: {width}x{height} → {new_width}x{new_height} ({scale_factor:.1f}x)")
            print(f"   [预处理] 保存至: {upscaled_path}")
            
            return upscaled_path, True, scale_factor
            
        except Exception as e:
            print(f"   [预处理] 超分失败: {e}，使用原图继续")
            return image_path, False, 1.0
    
    def process_image(self,
                      image_path: str,
                      output_dir: str = None,
                      with_refinement: bool = False,
                      with_text: bool = True,
                      groups: List[PromptGroup] = None) -> Optional[str]:
        """Run pipeline on one image. Returns output XML path or None."""
        print(f"\n{'='*60}")
        print(f"开始处理: {image_path}")
        print(f"{'='*60}")
        
        # 准备输出目录
        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('output_dir', './output')
        
        img_stem = Path(image_path).stem
        img_output_dir = os.path.join(output_dir, img_stem)
        os.makedirs(img_output_dir, exist_ok=True)
        
        print("\n[0] Preprocess...")
        processed_image_path, was_upscaled, scale_factor = self._preprocess_image(image_path, img_output_dir)

        context = ProcessingContext(
            image_path=processed_image_path,
            output_dir=img_output_dir
        )
        
        # 记录超分信息到上下文
        context.intermediate_results['original_image_path'] = image_path
        context.intermediate_results['was_upscaled'] = was_upscaled
        context.intermediate_results['upscale_factor'] = scale_factor

        try:
            if with_text and self.text_restorer is not None:
                print("\n[1] Text extraction (OCR)...")
                try:
                    text_xml_content = self.text_restorer.process(image_path)
                    text_output_path = os.path.join(img_output_dir, "text_only.drawio")
                    with open(text_output_path, 'w', encoding='utf-8') as f:
                        f.write(text_xml_content)
                    context.intermediate_results['text_xml'] = text_xml_content
                    print(f"   Saved: {text_output_path}")
                except Exception as e:
                    print(f"   Text step failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print("   Continuing without text...")
            elif with_text:
                print("\n[1] Text extraction (skipped - deps)")
            else:
                print("\n[1] Text extraction (skipped)")

            print("\n[2] Segmentation (SAM3)...")
            
            if groups:
                # 指定组提取
                all_elements = []
                for group in groups:
                    result = self.sam3_extractor.extract_by_group(context, group)
                    all_elements.extend(result.elements)
                for i, elem in enumerate(all_elements):
                    elem.id = i
                context.elements = all_elements
                context.canvas_width = result.canvas_width
                context.canvas_height = result.canvas_height
            else:
                # 全部组提取
                result = self.sam3_extractor.process(context)
                if not result.success:
                    raise Exception(f"SAM3提取失败: {result.error_message}")
                context.elements = result.elements
                context.canvas_width = result.canvas_width
                context.canvas_height = result.canvas_height
            
            print(f"   Elements: {len(context.elements)}")
            vis_path = os.path.join(img_output_dir, "sam3_extraction.png")
            self.sam3_extractor.save_visualization(context, vis_path)
            meta_path = os.path.join(img_output_dir, "sam3_metadata.json")
            self.sam3_extractor.save_metadata(context, meta_path)

            print("\n[3] Shape/icon processing...")
            result = self.icon_processor.process(context)
            print(f"   Icons: {result.metadata.get('processed_count', 0)}")
            result = self.shape_processor.process(context)
            print(f"   Shapes: {result.metadata.get('processed_count', 0)}")

            print("\n[4] Arrows...")
            result = self.arrow_processor.process(context)
            print(f"   Arrows: {result.metadata.get('arrows_processed', 0)}")

            print("\n[5] XML fragments...")
            self._generate_xml_fragments(context)
            xml_count = len([e for e in context.elements if e.has_xml()])
            print(f"   Fragments: {xml_count}")

            if with_refinement:
                print("\n[6] Metric evaluation...")
                eval_result = self.metric_evaluator.process(context)
                
                overall_score = eval_result.metadata.get('overall_score', 0)
                bad_regions = eval_result.metadata.get('bad_regions', [])
                needs_refinement = eval_result.metadata.get('needs_refinement', False)
                bad_region_ratio = eval_result.metadata.get('bad_region_ratio', 0)
                pixel_coverage = eval_result.metadata.get('pixel_coverage', 0)
                print(f"   Score: {overall_score:.1f}/100, bad regions: {len(bad_regions)} ({bad_region_ratio:.1f}%)")
                print(f"   Coverage: {pixel_coverage:.1f}%, needs_refine: {needs_refinement}")

                REFINEMENT_THRESHOLD = 90.0
                should_refine = overall_score < REFINEMENT_THRESHOLD and bad_regions

                if should_refine:
                    print("\n[7] Refinement...")
                    context.intermediate_results['bad_regions'] = bad_regions
                    refine_result = self.refinement_processor.process(context)
                    new_count = refine_result.metadata.get('new_elements_count', 0)
                    print(f"   新增 {new_count} 个元素")
                    
                    if new_count > 0:
                        refine_vis_path = os.path.join(img_output_dir, "refinement_result.png")
                        new_elements = context.elements[-new_count:] if new_count > 0 else []
                        self.refinement_processor.save_visualization(context, new_elements, refine_vis_path)
                        print(f"   Saved: {refine_vis_path}")
                elif not bad_regions:
                    print("\n[7] Refinement skipped (no bad regions)")
                else:
                    print("\n[7] Refinement skipped (score ok)")

            print("\n[8] Merge XML...")
            merge_result = self.xml_merger.process(context)
            
            if not merge_result.success:
                raise Exception(f"XML合并失败: {merge_result.error_message}")
            
            output_path = merge_result.metadata.get('output_path')
            print(f"   Output: {output_path}")
            print(f"\n{'='*60}\nDone.\n{'='*60}")
            
            return output_path
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_xml_fragments(self, context: ProcessingContext):
        """Generate XML for elements that do not have one yet."""
        for elem in context.elements:
            if elem.has_xml():
                continue
            
            # 根据元素类型生成XML
            elem_type = elem.element_type.lower()
            
            if elem_type in {'icon', 'picture', 'logo', 'chart', 'function_graph'}:
                # 图片类：使用base64图片
                if elem.base64:
                    style = f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png,{elem.base64}"
                else:
                    style = "rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;"
                elem.layer_level = LayerLevel.IMAGE.value
                
            elif elem_type in {'arrow', 'line', 'connector'}:
                # 箭头类
                style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;"
                elem.layer_level = LayerLevel.ARROW.value
                
            elif elem_type in {'section_panel', 'title_bar'}:
                # 背景/容器类
                fill = elem.fill_color or "#ffffff"
                stroke = elem.stroke_color or "#000000"
                style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};dashed=1;"
                elem.layer_level = LayerLevel.BACKGROUND.value
                
            else:
                # 基本图形
                fill = elem.fill_color or "#ffffff"
                stroke = elem.stroke_color or "#000000"
                
                if elem_type == 'rounded rectangle':
                    style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                elif elem_type == 'diamond':
                    style = f"rhombus;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                elif elem_type in {'ellipse', 'circle'}:
                    style = f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                elif elem_type == 'cloud':
                    style = f"ellipse;shape=cloud;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                else:
                    style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                
                elem.layer_level = LayerLevel.BASIC_SHAPE.value
            
            # 生成mxCell XML
            elem.xml_fragment = f'''<mxCell id="{elem.id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{elem.bbox.x1}" y="{elem.bbox.y1}" width="{elem.bbox.width}" height="{elem.bbox.height}" as="geometry"/>
</mxCell>'''


# ======================== CLI ========================
def main():
    parser = argparse.ArgumentParser(
        description="Edit Banana — image to DrawIO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i input/test.png
  python main.py
  python main.py -i test.png --refine
  python main.py -i test.png --groups image arrow
        """
    )
    
    parser.add_argument("-i", "--input", type=str, 
                        help="输入图片路径（不指定则处理input/目录下所有图片）")
    parser.add_argument("-o", "--output", type=str, 
                        help="输出目录（默认：./output）")
    parser.add_argument("--refine", action="store_true",
                        help="启用质量评估和二次处理")
    parser.add_argument("--no-text", action="store_true",
                        help="跳过文字处理（不调用 OCR）")
    parser.add_argument("--groups", nargs='+', 
                        choices=['image', 'arrow', 'shape', 'background'],
                        help="指定要处理的提示词组（默认全部）")
    parser.add_argument("--show-prompts", action="store_true",
                        help="显示当前词库配置")
    
    args = parser.parse_args()
    
    # 显示词库配置
    if args.show_prompts:
        extractor = Sam3InfoExtractor()
        extractor.print_prompt_groups()
        return
    
    # 加载配置
    config = load_config()
    
    # 创建流水线
    pipeline = Pipeline(config)
    
    # 解析分组参数
    groups = None
    if args.groups:
        group_map = {
            'image': PromptGroup.IMAGE,
            'arrow': PromptGroup.ARROW,
            'shape': PromptGroup.BASIC_SHAPE,
            'background': PromptGroup.BACKGROUND,
        }
        groups = [group_map[g] for g in args.groups]
    
    # 确定输出目录
    output_dir = args.output or config.get('paths', {}).get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集待处理图片
    image_paths = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if args.input:
        # 指定单张图片
        if not os.path.exists(args.input):
            print(f"❌ 错误：文件不存在 {args.input}")
            sys.exit(1)
        image_paths.append(args.input)
    else:
        # 批量处理input/目录
        input_dir = config.get('paths', {}).get('input_dir', './input')
        
        if not os.path.exists(input_dir):
            print(f"❌ 错误：输入目录不存在 {input_dir}")
            print(f"   请创建目录并放入图片，或使用 -i 参数指定图片路径")
            sys.exit(1)
        
        for file in os.listdir(input_dir):
            ext = Path(file).suffix.lower()
            if ext in supported_formats:
                image_paths.append(os.path.join(input_dir, file))
        
        if not image_paths:
            print(f"❌ 错误：{input_dir} 目录下没有找到支持的图片文件")
            print(f"   支持的格式: {', '.join(supported_formats)}")
            sys.exit(1)
    
    # 处理图片
    print(f"\n即将处理 {len(image_paths)} 张图片...")
    
    success_count = 0
    for img_path in image_paths:
        result = pipeline.process_image(
            img_path,
            output_dir=output_dir,
            with_refinement=args.refine,
            with_text=not args.no_text,
            groups=groups
        )
        if result:
            success_count += 1
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"处理完成: {success_count}/{len(image_paths)} 张图片成功")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
