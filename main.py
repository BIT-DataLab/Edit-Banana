#!/usr/bin/env python3
"""
图片转可编辑DrawIO格式 - 主入口

功能：
    - 将流程图/架构图等图片转换为可编辑的DrawIO XML格式
    - 支持单张图片处理或批量处理
    - 模块化设计，支持多人协作开发

使用方法：
    # 处理单张图片
    python main.py -i input/test.png
    
    # 批量处理 input/ 目录下所有图片
    python main.py
    
    # 指定输出目录
    python main.py -i input/test.png -o output/custom/
    
    # 启用质量评估和二次处理
    python main.py -i input/test.png --refine

    # 跳过文字处理
    python main.py -i input/test.png --no-text

模块调用流程：
    1. 【文字处理】TextRestorer → OCR识别文字和公式，输出文字XML
    2. SAM3提取信息 → 输出结构化ElementInfo列表（分组词库提取）
    3. Icon/Picture处理 → 去背景，转base64，输出XML
    4. 基本图形处理 → 取色，输出XML
    5. 箭头处理 → 分析方向，输出XML
    6. XML合并 → 按层级合并所有XML（包含文字XML）
    7. (可选) 质量评估 → 识别问题区域
    8. (可选) 二次处理 → 处理问题区域
    9. 最终合并 → 输出完整DrawIO XML

作者：图形组
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


# ======================== 配置加载 ========================
def load_config() -> dict:
    """加载配置文件"""
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


# ======================== 核心处理流程 ========================
class Pipeline:
    """
    图片处理流水线
    
    将多个处理模块串联起来，形成完整的处理流程。
    各模块相互独立，可以单独测试和修改。
    """
    
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        
        # 初始化各处理模块（懒加载，按需创建）
        self._text_restorer = None  # 文字处理（第一步）
        self._sam3_extractor = None
        self._icon_processor = None
        self._shape_processor = None
        self._arrow_processor = None
        self._xml_merger = None
        self._metric_evaluator = None
        self._refinement_processor = None
        self._upscale_model = None  # 条件超分模型
        
        # 超分配置
        self._upscale_min_dimension = self.config.get('upscale', {}).get('min_dimension', UPSCALE_MIN_DIMENSION)
        self._upscale_enabled = self.config.get('upscale', {}).get('enabled', True)
    
    @property
    def text_restorer(self):
        """文字处理模块（已整合到 modules/text/）；依赖缺失时为 None"""
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
        """条件超分模型（懒加载）"""
        if self._upscale_model is None:
            self._upscale_model = UpscaleModel(model_path=None)  # 使用默认路径
        return self._upscale_model
    
    def _preprocess_image(self, image_path: str, output_dir: str) -> tuple:
        """
        图像预处理：条件超分
        
        如果原图分辨率低于阈值，先进行超分处理
        
        Args:
            image_path: 原始图片路径
            output_dir: 输出目录
            
        Returns:
            (processed_image_path, was_upscaled, scale_factor)
        """
        from PIL import Image
        
        # 检查是否启用超分
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
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录（可选）
            with_refinement: 是否进行质量评估和二次处理
            with_text: 是否进行文字处理（默认True）
            groups: 要处理的提示词组（默认全部），可选：
                    PromptGroup.IMAGE, PromptGroup.ARROW, 
                    PromptGroup.BASIC_SHAPE, PromptGroup.BACKGROUND
            
        Returns:
            输出XML文件路径，失败返回None
        """
        print(f"\n{'='*60}")
        print(f"开始处理: {image_path}")
        print(f"{'='*60}")
        
        # 准备输出目录
        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('output_dir', './output')
        
        img_stem = Path(image_path).stem
        img_output_dir = os.path.join(output_dir, img_stem)
        os.makedirs(img_output_dir, exist_ok=True)
        
        # ===== 步骤0: 条件超分预处理 =====
        print("\n[步骤0] 图像预处理（条件超分）...")
        processed_image_path, was_upscaled, scale_factor = self._preprocess_image(image_path, img_output_dir)
        
        # 创建处理上下文（使用预处理后的图片路径）
        context = ProcessingContext(
            image_path=processed_image_path,
            output_dir=img_output_dir
        )
        
        # 记录超分信息到上下文
        context.intermediate_results['original_image_path'] = image_path
        context.intermediate_results['was_upscaled'] = was_upscaled
        context.intermediate_results['upscale_factor'] = scale_factor
        
        try:
            # ===== 步骤1: 文字处理（放在最前面）=====
            text_xml_content = None
            if with_text and self.text_restorer is not None:
                print("\n[步骤1] 文字处理（OCR识别 + 公式）...")
                try:
                    # 调用 modules/text 的 TextRestorer.process() 方法
                    # 返回文字部分的 XML 字符串
                    text_xml_content = self.text_restorer.process(image_path)
                    
                    # 保存文字处理结果
                    text_output_path = os.path.join(img_output_dir, "text_only.drawio")
                    with open(text_output_path, 'w', encoding='utf-8') as f:
                        f.write(text_xml_content)
                    
                    # 存储到上下文中，供后续 XML 合并使用
                    context.intermediate_results['text_xml'] = text_xml_content
                    
                    print(f"   文字处理完成，已保存: {text_output_path}")
                except Exception as e:
                    print(f"   文字处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"   继续处理其他元素...")
            elif with_text:
                print("\n[步骤1] 文字处理（跳过 - 依赖不可用）")
            else:
                print("\n[步骤1] 文字处理（跳过 - 用户指定）")
            
            # ===== 步骤2: SAM3提取信息（分组词库） =====
            print("\n[步骤2] SAM3提取元素（分组词库模式）...")
            
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
            
            print(f"   提取到 {len(context.elements)} 个元素")
            
            # 保存提取结果可视化
            vis_path = os.path.join(img_output_dir, "sam3_extraction.png")
            self.sam3_extractor.save_visualization(context, vis_path)
            
            # 保存元数据
            meta_path = os.path.join(img_output_dir, "sam3_metadata.json")
            self.sam3_extractor.save_metadata(context, meta_path)
            
            # ===== 步骤3: Icon/Picture处理 =====
            print("\n[步骤3] 处理Icon/Picture...")
            result = self.icon_processor.process(context)
            print(f"   处理完成: {result.metadata.get('processed_count', 0)} 个元素")
            
            # ===== 步骤4: 基本图形处理 =====
            print("\n[步骤4] 处理基本图形...")
            result = self.shape_processor.process(context)
            print(f"   处理完成: {result.metadata.get('processed_count', 0)} 个元素")
            
            # ===== 步骤5: 箭头处理 =====
            print("\n[步骤5] 处理箭头...")
            result = self.arrow_processor.process(context)
            print(f"   处理完成: {result.metadata.get('arrows_processed', 0)} 个箭头")
            
            # ===== 步骤6: 为每个元素生成XML（如果还没有） =====
            print("\n[步骤6] 生成XML片段...")
            self._generate_xml_fragments(context)
            xml_count = len([e for e in context.elements if e.has_xml()])
            print(f"   生成 {xml_count} 个XML片段")
            
            # ===== 步骤7: 质量评估（可选） =====
            if with_refinement:
                print("\n[步骤7] 质量评估...")
                eval_result = self.metric_evaluator.process(context)
                
                overall_score = eval_result.metadata.get('overall_score', 0)
                bad_regions = eval_result.metadata.get('bad_regions', [])
                needs_refinement = eval_result.metadata.get('needs_refinement', False)
                
                bad_region_ratio = eval_result.metadata.get('bad_region_ratio', 0)
                pixel_coverage = eval_result.metadata.get('pixel_coverage', 0)
                print(f"   评估分数: {overall_score:.1f}/100 (100 - 问题区域面积比例)")
                print(f"   问题区域: {len(bad_regions)} 个, 总面积: {bad_region_ratio:.1f}%")
                print(f"   像素覆盖率: {pixel_coverage:.1f}% (辅助指标)")
                print(f"   需要refinement: {needs_refinement}")
                print(f"   可视化已保存: {img_output_dir}/metric_uncovered.png")
                print(f"   评估结果已保存: {img_output_dir}/metric_evaluation.json")
                
                # ===== 步骤8: 二次处理（只有评分 < 90 才启动） =====
                REFINEMENT_THRESHOLD = 90.0
                should_refine = overall_score < REFINEMENT_THRESHOLD and bad_regions
                
                if should_refine:
                    print("\n[步骤8] 二次处理（Fallback补救）...")
                    context.intermediate_results['bad_regions'] = bad_regions
                    refine_result = self.refinement_processor.process(context)
                    new_count = refine_result.metadata.get('new_elements_count', 0)
                    print(f"   新增 {new_count} 个元素")
                    
                    if new_count > 0:
                        # 保存 refinement 可视化
                        refine_vis_path = os.path.join(img_output_dir, "refinement_result.png")
                        # 获取新增的元素（最后 new_count 个）
                        new_elements = context.elements[-new_count:] if new_count > 0 else []
                        self.refinement_processor.save_visualization(context, new_elements, refine_vis_path)
                        print(f"   可视化已保存: {refine_vis_path}")
                elif not bad_regions:
                    print("\n[步骤8] 跳过二次处理（无问题区域）")
                else:
                    print("\n[步骤8] 跳过二次处理（覆盖率已达标）")
            
            # ===== 步骤9: XML合并 =====
            print("\n[步骤9] 合并XML...")
            merge_result = self.xml_merger.process(context)
            
            if not merge_result.success:
                raise Exception(f"XML合并失败: {merge_result.error_message}")
            
            output_path = merge_result.metadata.get('output_path')
            print(f"   输出文件: {output_path}")
            
            print(f"\n{'='*60}")
            print(f"✅ 处理完成!")
            print(f"{'='*60}")
            
            return output_path
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_xml_fragments(self, context: ProcessingContext):
        """
        为还没有XML的元素生成XML片段
        
        这是一个临时实现，各子模块应该在自己的process()中完成XML生成。
        """
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


# ======================== 命令行接口 ========================
def main():
    parser = argparse.ArgumentParser(
        description="图片转可编辑DrawIO格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py -i input/test.png           # 处理单张图片
  python main.py                              # 批量处理input/目录
  python main.py -i test.png --refine        # 启用质量评估和二次处理
  python main.py -i test.png --groups image arrow  # 只提取图片和箭头
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
