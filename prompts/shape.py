# Basic shape prompts for SAM3
#
# Design principles (from SAM3 best practices):
#   - Short, specific noun phrases
#   - Cover scientific figure conventions (trapezoids for encoders, cubes for volumes)
#   - Avoid redundancy; the dedup step handles overlaps across prompts
#
# Downstream support: basic_shape_processor handles rectangle, rounded_rectangle,
# diamond, ellipse, circle, cylinder, cloud, actor, hexagon, triangle,
# parallelogram, trapezoid, square (see modules/basic_shape_processor.py).
SHAPE_PROMPT = [
    # Core rectangles (most common in scientific figures)
    "rectangle",
    "rounded rectangle",
    "square",

    # Conic / pill shapes
    "ellipse",
    "circle",

    # Rhombic
    "diamond",

    # Encoder/decoder shapes (very common in ML/DL figures — were missed in v1)
    "trapezoid",
    "parallelogram",

    # Polygons
    "triangle",
    "hexagon",

    # 3D / volumetric (common in medical imaging & ML figures — were missed in v1)
    "3D cube",
    "isometric box",
    "cylinder",

    # Small colored elements (for legends — were missed in v1 due to size)
    "color swatch",
    "small colored square",

    # Stacked / layered (for image stacks, CT volumes — were missed in v1)
    "stack of rectangles",
    "layered boxes",
]
