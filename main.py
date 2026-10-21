#!/usr/bin/env python3
"""
Edit Banana — CLI entry: image to editable DrawIO XML.

Pipeline: input image -> preprocess -> text OCR -> SAM3 segmentation -> shape/icon processing -> XML merge -> output .drawio.
Requires: config/config.yaml (sam3.checkpoint_path, sam3.bpe_path), SAM3 library and weights, Tesseract or PaddleOCR.
See README and docs/SETUP_SAM3.md.

Usage:
    python main.py -i input/test.png
    python main.py
    python main.py -i input/test.png -o output/custom/
    python main.py -i input/test.png --refine
    python main.py -i input/test.png --no-text
    python main.py -i input/test.png --vector-level=all
    python main.py -i input/test.png --no-vectors
"""

import os
import sys
import argparse
import warnings
import yaml
from pathlib import Path
from typing import Optional, List

# Skip PaddleX model host connectivity check to avoid startup delay
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Suppress requests urllib3/chardet version warning
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

# Project root on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules import (
    # Core processors
    Sam3InfoExtractor,
    IconPictureProcessor,
    BasicShapeProcessor,
    XMLMerger,
    MetricEvaluator,
    RefinementProcessor,

    # Stage 8: Vector export
    VectorExporter,

    # Text (modules/text/)
    TextRestorer,

    # Context and data types
    ProcessingContext,
    ProcessingResult,
    ElementInfo,
    LayerLevel,
    get_layer_level,
)

# Prompt groups enum
from modules.sam3_info_extractor import PromptGroup

# Text module available (depends on ocr/coord_processor etc.)
TEXT_MODULE_AVAILABLE = TextRestorer is not None


# ======================== config ========================
def load_config() -> dict:
    """Load config/config.yaml."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Warning: config file not found at {config_path}, using defaults")
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
        self._xml_merger = None
        self._metric_evaluator = None
        self._refinement_processor = None
        self._vector_exporter = None
    
    @property
    def text_restorer(self):
        """OCR/text step; None if deps missing."""
        if self._text_restorer is None and TextRestorer is not None:
            ocr_engine = (self.config.get("ocr") or {}).get("engine", "tesseract")
            self._text_restorer = TextRestorer(
                formula_engine="none",
                ocr_engine=ocr_engine,
            )
        return self._text_restorer
    
    @property
    def sam3_extractor(self) -> Sam3InfoExtractor:
        if self._sam3_extractor is None:
            self._sam3_extractor = Sam3InfoExtractor()
        return self._sam3_extractor
    
    @property
    def icon_processor(self) -> IconPictureProcessor:
        if self._icon_processor is None:
            rmbg_cfg = self.config.get("rmbg") or {}
            rmbg_path = rmbg_cfg.get("model_path")
            self._icon_processor = IconPictureProcessor(rmbg_model_path=rmbg_path)
        return self._icon_processor
    
    @property
    def shape_processor(self) -> BasicShapeProcessor:
        if self._shape_processor is None:
            self._shape_processor = BasicShapeProcessor()
        return self._shape_processor
    
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
    def vector_exporter(self) -> VectorExporter:
        if self._vector_exporter is None:
            self._vector_exporter = VectorExporter()
        return self._vector_exporter
    
    def process_image(self,
                      image_path: str,
                      output_dir: str = None,
                      with_refinement: bool = False,
                      with_text: bool = True,
                      groups: List[PromptGroup] = None,
                      vector_level: str = "granular",
                      no_vectors: bool = False) -> Optional[str]:
        """Run pipeline on one image. Returns output XML path or None."""
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")
        
        # Output directory
        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('output_dir', './output')
        
        img_stem = Path(image_path).stem
        img_output_dir = os.path.join(output_dir, img_stem)
        os.makedirs(img_output_dir, exist_ok=True)
        
        print("\n[0] Preprocess...")
        context = ProcessingContext(
            image_path=image_path,
            output_dir=img_output_dir
        )
        context.intermediate_results['original_image_path'] = image_path
        context.intermediate_results['was_upscaled'] = False
        context.intermediate_results['upscale_factor'] = 1.0

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
                    print("   Continuing without text...")
            elif with_text:
                print("\n[1] Text extraction (skipped - deps)")
            else:
                print("\n[1] Text extraction (skipped)")

            print("\n[2] Segmentation (SAM3)...")

            if groups:
                # Extract by group
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
                # Full extraction
                result = self.sam3_extractor.process(context)
                if not result.success:
                    raise Exception(f"SAM3 extraction failed: {result.error_message}")
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

            print("\n[4] XML fragments...")
            self._generate_xml_fragments(context)
            xml_count = len([e for e in context.elements if e.has_xml()])
            print(f"   Fragments: {xml_count}")

            if with_refinement:
                print("\n[5] Metric evaluation...")
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
                    print("\n[6] Refinement...")
                    context.intermediate_results['bad_regions'] = bad_regions
                    refine_result = self.refinement_processor.process(context)
                    new_count = refine_result.metadata.get('new_elements_count', 0)
                    print(f"   Added {new_count} elements")
                    
                    if new_count > 0:
                        refine_vis_path = os.path.join(img_output_dir, "refinement_result.png")
                        new_elements = context.elements[-new_count:] if new_count > 0 else []
                        self.refinement_processor.save_visualization(context, new_elements, refine_vis_path)
                        print(f"   Saved: {refine_vis_path}")
                elif not bad_regions:
                    print("\n[6] Refinement skipped (no bad regions)")
                else:
                    print("\n[6] Refinement skipped (score ok)")

            print("\n[7] Merge XML...")
            merge_result = self.xml_merger.process(context)
            
            if not merge_result.success:
                raise Exception(f"XML merge failed: {merge_result.error_message}")
            
            output_path = merge_result.metadata.get('output_path')
            print(f"   Output: {output_path}")

            # ============ Stage 8: Vector Export ============
            if not no_vectors:
                print(f"\n[8] Vector export (level={vector_level})...")
                context.intermediate_results['vector_level'] = vector_level
                try:
                    vec_result = self.vector_exporter.process(context)
                    if vec_result.success:
                        vec_count = vec_result.metadata.get('exported_count', 0)
                        vec_dir = vec_result.metadata.get('vector_dir', '')
                        print(f"   Exported {vec_count} elements -> {vec_dir}")
                    else:
                        print(f"   Vector export failed: {vec_result.error_message}")
                except Exception as e:
                    print(f"   Vector export failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("\n[8] Vector export (skipped)")

            print(f"\n{'='*60}\nDone.\n{'='*60}")

            return output_path
            
        except Exception as e:
            print(f"\nFailed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_xml_fragments(self, context: ProcessingContext):
        """Generate XML for elements that do not have one yet. Arrows are treated as icon (image crop)."""
        for elem in context.elements:
            if elem.has_xml():
                continue
            
            elem_type = elem.element_type.lower()
            
            if elem_type in {'icon', 'picture', 'logo', 'chart', 'function_graph', 'arrow', 'line', 'connector'}:
                # Image/arrow: use base64 image
                if elem.base64:
                    style = f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image=data:image/png,{elem.base64}"
                else:
                    style = "rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;"
                elem.layer_level = LayerLevel.IMAGE.value
                
            elif elem_type in {'section_panel', 'title_bar'}:
                # Background/container
                fill = elem.fill_color or "#ffffff"
                stroke = elem.stroke_color or "#000000"
                style = f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};dashed=1;"
                elem.layer_level = LayerLevel.BACKGROUND.value
                
            else:
                # Basic shape
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
            
            # Build mxCell XML
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
  python main.py -i test.png --vector-level=all
  python main.py -i test.png --no-vectors
        """
    )
    
    parser.add_argument("-i", "--input", type=str, 
                        help="Input image path (omit to process all images in input/)")
    parser.add_argument("-o", "--output", type=str, 
                        help="Output directory (default: ./output)")
    parser.add_argument("--refine", action="store_true",
                        help="Enable quality evaluation and refinement")
    parser.add_argument("--no-text", action="store_true",
                        help="Skip text step (no OCR)")
    parser.add_argument("--groups", nargs='+', 
                        choices=['image', 'arrow', 'shape', 'background'],
                        help="Prompt groups to process (default: all)")
    parser.add_argument("--show-prompts", action="store_true",
                        help="Show prompt config")

    # Stage 8: Vector export options
    parser.add_argument("--vector-level", type=str, default="granular",
                        choices=['granular', 'section', 'component', 'all'],
                        help="Vector export granularity (default: granular)")
    parser.add_argument("--no-vectors", action="store_true",
                        help="Skip vector export (Stage 8)")
    
    args = parser.parse_args()
    
    # Show prompt config
    if args.show_prompts:
        extractor = Sam3InfoExtractor()
        extractor.print_prompt_groups()
        return
    
    # Load config
    config = load_config()
    
    # Create pipeline
    pipeline = Pipeline(config)
    
    # Parse group args
    groups = None
    if args.groups:
        group_map = {
            'image': PromptGroup.IMAGE,
            'arrow': PromptGroup.ARROW,
            'shape': PromptGroup.BASIC_SHAPE,
            'background': PromptGroup.BACKGROUND,
        }
        groups = [group_map[g] for g in args.groups]
    
    # Output dir
    output_dir = args.output or config.get('paths', {}).get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect images
    image_paths = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if args.input:
        # Single image
        if not os.path.exists(args.input):
            print(f"Error: file not found {args.input}")
            sys.exit(1)
        image_paths.append(args.input)
    else:
        # Batch from input/
        input_dir = config.get('paths', {}).get('input_dir', './input')
        
        if not os.path.exists(input_dir):
            print(f"Error: input directory does not exist: {input_dir}")
            print(f"   Create it and add images, or use -i to specify an image path")
            sys.exit(1)
        
        for file in os.listdir(input_dir):
            ext = Path(file).suffix.lower()
            if ext in supported_formats:
                image_paths.append(os.path.join(input_dir, file))
        
        if not image_paths:
            print(f"Error: no supported image files in {input_dir}")
            print(f"   Supported formats: {', '.join(supported_formats)}")
            sys.exit(1)
    
    # Process
    print(f"\nProcessing {len(image_paths)} image(s)...")
    
    success_count = 0
    for img_path in image_paths:
        result = pipeline.process_image(
            img_path,
            output_dir=output_dir,
            with_refinement=args.refine,
            with_text=not args.no_text,
            groups=groups,
            vector_level=args.vector_level,
            no_vectors=args.no_vectors,
        )
        if result:
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {success_count}/{len(image_paths)} succeeded")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
