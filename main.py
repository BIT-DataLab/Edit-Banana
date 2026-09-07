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
"""

import os
import sys
import argparse
import warnings
import yaml
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import asdict, dataclass, field
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

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

    # Text (modules/text/)
    TextRestorer,

    # Context and data types
    ProcessingContext,
    ProcessingResult,
    ElementInfo,
    LayerLevel,
    get_layer_level,
)

# Import exceptions for structured error handling
from modules.exceptions import (
    EditBananaException,
    ErrorSeverity,
    SegmentationError,
)

# Import retry decorator for resilient operations
from modules.core import retry_with_defaults

# Prompt groups enum
from modules.sam3_info_extractor import PromptGroup

# Text module available (depends on ocr/coord_processor etc.)
TEXT_MODULE_AVAILABLE = TextRestorer is not None


# ======================== Checkpoint Manager ========================
class CheckpointManager:
    """Manages pipeline checkpoint saving and recovery."""

    CHECKPOINT_FILENAME = "pipeline_checkpoint.json"

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(output_dir, self.CHECKPOINT_FILENAME)

    def save_checkpoint(self, stage: int, context: 'ProcessingContext',
                       stage_results: Optional[Dict[str, Any]] = None) -> str:
        """Save checkpoint after completing a stage.

        Args:
            stage: Completed stage number (0-7)
            context: Current processing context
            stage_results: Optional stage-specific results

        Returns:
            Path to saved checkpoint file
        """
        checkpoint_data = {
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'completed_stage': stage,
            'image_path': context.image_path,
            'output_dir': context.output_dir,
            'canvas_width': context.canvas_width,
            'canvas_height': context.canvas_height,
            'elements': [elem.to_dict() for elem in context.elements],
            'xml_fragments': [frag.to_dict() for frag in context.xml_fragments],
            'intermediate_results': context.intermediate_results,
            'stage_results': stage_results or {}
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Checkpoint saved: stage {stage} -> {self.checkpoint_path}")
        return self.checkpoint_path

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint if it exists.

        Returns:
            Checkpoint data dict or None if not found
        """
        if not os.path.exists(self.checkpoint_path):
            return None

        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded: stage {data.get('completed_stage', 'unknown')}")
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def restore_context(self, checkpoint_data: Dict[str, Any]) -> 'ProcessingContext':
        """Restore ProcessingContext from checkpoint data.

        Args:
            checkpoint_data: Loaded checkpoint data

        Returns:
            Restored ProcessingContext
        """
        from modules.data_types import ElementInfo, XMLFragment

        context = ProcessingContext(
            image_path=checkpoint_data['image_path'],
            output_dir=checkpoint_data['output_dir'],
            canvas_width=checkpoint_data.get('canvas_width', 0),
            canvas_height=checkpoint_data.get('canvas_height', 0),
        )

        # Restore elements
        if 'elements' in checkpoint_data:
            context.elements = [
                ElementInfo.from_dict(elem_data)
                for elem_data in checkpoint_data['elements']
            ]

        # Restore XML fragments
        if 'xml_fragments' in checkpoint_data:
            context.xml_fragments = [
                XMLFragment.from_dict(frag_data)
                for frag_data in checkpoint_data['xml_fragments']
            ]

        # Restore intermediate results
        context.intermediate_results = checkpoint_data.get('intermediate_results', {})

        return context

    def clear_checkpoint(self):
        """Remove checkpoint file after successful completion."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            logger.debug(f"Checkpoint cleared: {self.checkpoint_path}")

    def get_last_completed_stage(self) -> int:
        """Get the last completed stage from checkpoint.

        Returns:
            Stage number (0-7) or -1 if no checkpoint
        """
        data = self.load_checkpoint()
        if data is None:
            return -1
        return data.get('completed_stage', -1)


# ======================== Pipeline Result ========================
@dataclass
class PipelineResult:
    """Result of pipeline processing with partial data on failure."""
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    can_retry: bool = False
    last_completed_stage: int = -1
    partial_elements: List[Dict[str, Any]] = field(default_factory=list)
    partial_xml_fragments: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'success': self.success,
            'output_path': self.output_path,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'can_retry': self.can_retry,
            'last_completed_stage': self.last_completed_stage,
            'partial_elements_count': len(self.partial_elements),
            'partial_xml_fragments_count': len(self.partial_xml_fragments),
            'checkpoint_path': self.checkpoint_path,
        }


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
    
    @property
    def text_restorer(self):
        """OCR/text step; None if deps missing."""
        if self._text_restorer is None and TextRestorer is not None:
            ocr_engine = (self.config.get("ocr") or {}).get("engine", "tesseract")
            from modules.text.text_restorer import TextRestorer as TrClass
            self._text_restorer = TrClass(
                formula_engine="none",
                ocr_engine=ocr_engine,
            )
            # Wrap process method with retry
            self._text_restorer.process = retry_with_defaults(
                max_retries=3, base_delay=1.0
            )(self._text_restorer.process)
        return self._text_restorer
    
    @property
    def sam3_extractor(self) -> Sam3InfoExtractor:
        if self._sam3_extractor is None:
            from modules.sam3_info_extractor import Sam3InfoExtractor as Sam3Class
            self._sam3_extractor = Sam3Class()
            # Wrap key methods with retry for resilience
            self._sam3_extractor.extract_by_group = retry_with_defaults(
                max_retries=3, base_delay=2.0
            )(self._sam3_extractor.extract_by_group)
            self._sam3_extractor.process = retry_with_defaults(
                max_retries=3, base_delay=2.0
            )(self._sam3_extractor.process)
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
    
    def process_image(self,
                      image_path: str,
                      output_dir: str = None,
                      with_refinement: bool = False,
                      with_text: bool = True,
                      groups: List[PromptGroup] = None,
                      resume_from_checkpoint: bool = False) -> PipelineResult:
        """Run pipeline on one image. Returns PipelineResult with partial data on failure.

        Args:
            image_path: Input image path
            output_dir: Output directory (default: ./output)
            with_refinement: Enable quality evaluation and refinement
            with_text: Enable text extraction (OCR)
            groups: Prompt groups to process (default: all)
            resume_from_checkpoint: Resume from last checkpoint if available

        Returns:
            PipelineResult with success status, output path (if successful),
            error details (if failed), and partial results for recovery.
        """
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")

        # Output directory
        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('output_dir', './output')

        img_stem = Path(image_path).stem
        img_output_dir = os.path.join(output_dir, img_stem)
        os.makedirs(img_output_dir, exist_ok=True)

        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(img_output_dir)

        # Check for existing checkpoint
        start_stage = 0
        context = None

        if resume_from_checkpoint:
            checkpoint_data = checkpoint_mgr.load_checkpoint()
            if checkpoint_data:
                last_completed = checkpoint_data.get('completed_stage', -1)
                if last_completed >= 0:
                    print(f"\n[Resume] Found checkpoint at stage {last_completed}, resuming...")
                    try:
                        context = checkpoint_mgr.restore_context(checkpoint_data)
                        start_stage = last_completed + 1
                        print(f"[Resume] Restored {len(context.elements)} elements, {len(context.xml_fragments)} fragments")
                    except Exception as e:
                        print(f"[Resume] Failed to restore checkpoint: {e}")
                        print("[Resume] Starting from beginning...")
                        context = None

        # Initialize context if not restored
        if context is None:
            print("\n[0] Preprocess...")
            context = ProcessingContext(
                image_path=image_path,
                output_dir=img_output_dir
            )
            context.intermediate_results['original_image_path'] = image_path
            context.intermediate_results['was_upscaled'] = False
            context.intermediate_results['upscale_factor'] = 1.0
            # Save initial checkpoint
            checkpoint_mgr.save_checkpoint(0, context)

        try:
            # Stage 1: Text extraction (OCR)
            if start_stage <= 1:
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
                checkpoint_mgr.save_checkpoint(1, context)

            # Stage 2: Segmentation (SAM3)
            if start_stage <= 2:
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
                        raise SegmentationError(
                            message=f"SAM3 extraction failed: {result.error_message}",
                            context={'stage': 2, 'elements_found': len(context.elements)}
                        )
                    context.elements = result.elements
                    context.canvas_width = result.canvas_width
                    context.canvas_height = result.canvas_height

                print(f"   Elements: {len(context.elements)}")
                vis_path = os.path.join(img_output_dir, "sam3_extraction.png")
                self.sam3_extractor.save_visualization(context, vis_path)
                meta_path = os.path.join(img_output_dir, "sam3_metadata.json")
                self.sam3_extractor.save_metadata(context, meta_path)
                checkpoint_mgr.save_checkpoint(2, context)

            # Stage 3: Shape/icon processing
            if start_stage <= 3:
                print("\n[3] Shape/icon processing...")
                result = self.icon_processor.process(context)
                print(f"   Icons: {result.metadata.get('processed_count', 0)}")
                result = self.shape_processor.process(context)
                print(f"   Shapes: {result.metadata.get('processed_count', 0)}")
                checkpoint_mgr.save_checkpoint(3, context)

            # Stage 4: XML fragments
            if start_stage <= 4:
                print("\n[4] XML fragments...")
                self._generate_xml_fragments(context)
                xml_count = len([e for e in context.elements if e.has_xml()])
                print(f"   Fragments: {xml_count}")
                checkpoint_mgr.save_checkpoint(4, context)

            # Stage 5-6: Metric evaluation and refinement (optional)
            if with_refinement:
                if start_stage <= 5:
                    print("\n[5] Metric evaluation...")
                    eval_result = self.metric_evaluator.process(context)

                    overall_score = eval_result.metadata.get('overall_score', 0)
                    bad_regions = eval_result.metadata.get('bad_regions', [])
                    needs_refinement = eval_result.metadata.get('needs_refinement', False)
                    bad_region_ratio = eval_result.metadata.get('bad_region_ratio', 0)
                    pixel_coverage = eval_result.metadata.get('pixel_coverage', 0)
                    print(f"   Score: {overall_score:.1f}/100, bad regions: {len(bad_regions)} ({bad_region_ratio:.1f}%)")
                    print(f"   Coverage: {pixel_coverage:.1f}%, needs_refine: {needs_refinement}")

                    # Store for stage 6
                    context.intermediate_results['eval_result'] = {
                        'overall_score': overall_score,
                        'bad_regions': bad_regions,
                        'needs_refinement': needs_refinement,
                        'bad_region_ratio': bad_region_ratio,
                        'pixel_coverage': pixel_coverage,
                    }
                    checkpoint_mgr.save_checkpoint(5, context)

                if start_stage <= 6:
                    # Reconstruct eval data from checkpoint if needed
                    eval_data = context.intermediate_results.get('eval_result', {})
                    overall_score = eval_data.get('overall_score', 0)
                    bad_regions = eval_data.get('bad_regions', [])

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
                    checkpoint_mgr.save_checkpoint(6, context)

            # Stage 7: Merge XML
            if start_stage <= 7:
                print("\n[7] Merge XML...")
                merge_result = self.xml_merger.process(context)

                if not merge_result.success:
                    raise Exception(f"XML merge failed: {merge_result.error_message}")

                output_path = merge_result.metadata.get('output_path')
                print(f"   Output: {output_path}")

                # Clear checkpoint on success
                checkpoint_mgr.clear_checkpoint()
                print(f"\n{'='*60}\nDone.\n{'='*60}")

                return PipelineResult(
                    success=True,
                    output_path=output_path,
                    last_completed_stage=7
                )

        except Exception as e:
            error_msg = str(e)
            error_code = "PROCESSING_ERROR"
            can_retry = True
            last_stage = start_stage - 1

            # Extract error details from EditBananaException
            if isinstance(e, EditBananaException):
                error_code = e.error_code
                can_retry = e.retry_allowed
                logger.warning(f"Pipeline failed at stage {start_stage}: [{error_code}] {error_msg}")
            else:
                logger.warning(f"Pipeline failed at stage {start_stage}: {error_msg}")

            import traceback
            traceback.print_exc()

            # Save failure state for potential resume
            checkpoint_mgr.save_checkpoint(last_stage, context, {
                'error': error_msg,
                'failed_stage': start_stage,
                'error_code': error_code,
            })

            # Build partial results from context
            partial_elements = [elem.to_dict() for elem in context.elements] if context else []
            partial_fragments = [frag.to_dict() for frag in context.xml_fragments] if context else []

            print(f"\n[Checkpoint] Saved state at stage {last_stage}. Resume with --resume")

            return PipelineResult(
                success=False,
                error_message=error_msg,
                error_code=error_code,
                can_retry=can_retry,
                last_completed_stage=last_stage,
                partial_elements=partial_elements,
                partial_xml_fragments=partial_fragments,
                checkpoint_path=checkpoint_mgr.checkpoint_path
            )
    
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
            groups=groups
        )
        if result.success:
            success_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {success_count}/{len(image_paths)} succeeded")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
