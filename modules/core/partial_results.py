"""
Partial Results Handler

Manages saving and loading of intermediate processing state when errors occur.
This allows users to recover partial results even when the full pipeline fails.
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from ..exceptions import EditBananaException, ProcessingPartialResultError
from ..data_types import ProcessingResult, ElementInfo, BoundingBox
from ..base import ProcessingContext


@dataclass
class PartialResultState:
    """
    Serializable state of a partially completed processing job.

    Captures everything needed to resume or extract partial results.
    """
    # Original input
    image_path: str
    original_image_path: Optional[str] = None

    # Processing state
    elements: List[Dict[str, Any]] = field(default_factory=list)
    canvas_width: int = 0
    canvas_height: int = 0

    # XML fragments that were successfully generated
    xml_fragments: List[str] = field(default_factory=list)

    # Intermediate results from each stage
    intermediate_results: Dict[str, Any] = field(default_factory=dict)

    # Error information
    failed_stage: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_stages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'image_path': self.image_path,
            'original_image_path': self.original_image_path,
            'elements': self.elements,
            'canvas_width': self.canvas_width,
            'canvas_height': self.canvas_height,
            'xml_fragments': self.xml_fragments,
            'intermediate_results': self._serialize_intermediate_results(),
            'failed_stage': self.failed_stage,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'timestamp': self.timestamp,
            'completed_stages': self.completed_stages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PartialResultState':
        """Create from dictionary."""
        return cls(
            image_path=data['image_path'],
            original_image_path=data.get('original_image_path'),
            elements=data.get('elements', []),
            canvas_width=data.get('canvas_width', 0),
            canvas_height=data.get('canvas_height', 0),
            xml_fragments=data.get('xml_fragments', []),
            intermediate_results=data.get('intermediate_results', {}),
            failed_stage=data.get('failed_stage'),
            error_message=data.get('error_message'),
            error_code=data.get('error_code'),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            completed_stages=data.get('completed_stages', []),
        )

    def _serialize_intermediate_results(self) -> Dict[str, Any]:
        """Serialize intermediate results, handling non-JSON types."""
        serialized = {}
        for key, value in self.intermediate_results.items():
            if key == 'text_xml':
                # Store text XML as string
                serialized[key] = value if isinstance(value, str) else None
            elif key in ('was_upscaled', 'upscale_factor', 'original_image_path'):
                serialized[key] = value
            elif isinstance(value, (str, int, float, bool, list, dict)):
                serialized[key] = value
            else:
                # Skip non-serializable objects
                serialized[key] = f"<{type(value).__name__}>"
        return serialized


class PartialResultsHandler:
    """
    Handles saving and loading of partial processing results.

    When a pipeline stage fails, this captures the state so that:
    1. Users can access partial XML output
    2. Processing can potentially be resumed
    3. Debugging information is preserved
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, 'partial_state.json')

    def save_from_context(
        self,
        context: ProcessingContext,
        failed_stage: str,
        error: Exception,
        completed_stages: List[str]
    ) -> str:
        """
        Save partial state from a ProcessingContext.

        Args:
            context: The processing context at time of failure
            failed_stage: Name of the stage that failed
            error: The exception that caused failure
            completed_stages: List of stages that completed successfully

        Returns:
            Path to the saved state file
        """
        # Convert elements to serializable format
        elements_data = []
        xml_fragments = []

        for elem in context.elements:
            elem_dict = elem.to_dict()
            elements_data.append(elem_dict)

            if elem.has_xml():
                xml_fragments.append(elem.xml_fragment)

        # Build state
        state = PartialResultState(
            image_path=context.image_path,
            original_image_path=context.intermediate_results.get('original_image_path'),
            elements=elements_data,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            xml_fragments=xml_fragments,
            intermediate_results=dict(context.intermediate_results),
            failed_stage=failed_stage,
            error_message=str(error),
            error_code=getattr(error, 'error_code', 'UNKNOWN_ERROR'),
            completed_stages=completed_stages,
        )

        # Save JSON version (human-readable)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

        return self.state_file

    def load_state(self) -> Optional[PartialResultState]:
        """Load the saved partial state."""
        if not os.path.exists(self.state_file):
            return None

        with open(self.state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return PartialResultState.from_dict(data)

    def load_context(self) -> Optional[ProcessingContext]:
        """Load context from saved state."""
        state = self.load_state()
        if not state:
            return None

        # Reconstruct ProcessingContext
        elements = []
        for elem_data in state.elements:
            try:
                elem = ElementInfo.from_dict(elem_data)
                elements.append(elem)
            except Exception:
                # Skip corrupted elements
                continue

        context = ProcessingContext(
            image_path=state.image_path,
            canvas_width=state.canvas_width,
            canvas_height=state.canvas_height,
            output_dir=self.output_dir,
        )
        context.elements = elements
        context.intermediate_results = state.intermediate_results

        # Restore XML fragments to elements
        xml_fragments = state.xml_fragments
        for i, elem in enumerate(elements):
            if i < len(xml_fragments):
                elem.xml_fragment = xml_fragments[i]

        return context

    def generate_partial_xml(self) -> Optional[str]:
        """
        Generate XML from partial results.

        Returns complete DrawIO XML with whatever elements were processed.
        """
        state = self.load_state()
        if not state:
            return None

        # Build minimal DrawIO XML
        xml_parts = [
            '<mxfile host="EditBanana" version="partial">',
            '  <diagram name="Partial Result">',
            '    <mxGraphModel dx="1422" dy="762" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169">',
            '      <root>',
            '        <mxCell id="0" />',
            '        <mxCell id="1" parent="0" />',
        ]

        # Add elements that have XML
        for i, fragment in enumerate(state.xml_fragments):
            # Replace IDs to ensure sequential ordering
            xml_parts.append(f"        {fragment}")

        xml_parts.extend([
            '      </root>',
            '    </mxGraphModel>',
            '  </diagram>',
            '</mxfile>',
        ])

        return '\n'.join(xml_parts)

    def save_partial_xml(self) -> Optional[str]:
        """Save partial XML to file. Returns path or None."""
        xml_content = self.generate_partial_xml()
        if not xml_content:
            return None

        output_path = os.path.join(self.output_dir, 'partial_result.drawio')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the partial results."""
        state = self.load_state()
        if not state:
            return {'error': 'No partial state found'}

        return {
            'image_path': state.image_path,
            'completed_stages': state.completed_stages,
            'failed_stage': state.failed_stage,
            'error_message': state.error_message,
            'error_code': state.error_code,
            'elements_processed': len(state.elements),
            'xml_fragments_generated': len(state.xml_fragments),
            'timestamp': state.timestamp,
        }


def save_partial_results(
    context: ProcessingContext,
    output_dir: str,
    failed_stage: str,
    error: Exception,
    completed_stages: List[str]
) -> Dict[str, str]:
    """
    Convenience function to save partial results.

    Returns dict with paths to saved files.
    """
    handler = PartialResultsHandler(output_dir)

    # Save state
    state_path = handler.save_from_context(
        context=context,
        failed_stage=failed_stage,
        error=error,
        completed_stages=completed_stages
    )

    # Save partial XML
    xml_path = handler.save_partial_xml()

    return {
        'state_file': state_path,
        'partial_xml': xml_path,
    }


def load_partial_results(output_dir: str) -> Optional[ProcessingContext]:
    """Convenience function to load partial results."""
    handler = PartialResultsHandler(output_dir)
    return handler.load_context()
