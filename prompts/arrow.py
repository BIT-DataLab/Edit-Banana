# Arrow/connector prompts for SAM3
#
# Design principles (from SAM3 best practices):
#   - Short, specific noun phrases
#   - Distinguish arrow types by visual feature (straight / curved / thick / dashed)
#   - Include common diagram connector variants
#
# Note: SAM3 treats each prompt independently; similar prompts may produce
# overlapping detections that get merged in the dedup step.
ARROW_PROMPT = [
    # Core arrow types
    "arrow",
    "straight arrow",
    "thin arrow",

    # Thick / block arrows (e.g. "weight transfer" orange arrows in ML figures)
    "thick arrow",
    "block arrow",

    # Curved arrows (e.g. "iterative refinement" loops — were missed in v1)
    "curved arrow",
    "looping arrow",

    # Multi-endpoint arrows
    "bidirectional arrow",
    "double-headed arrow",

    # Line-type connectors
    "line",
    "connector",
    "connecting line",

    # Dashed / dotted (for skip connections, data flow — common in neural net figures)
    "dashed line",
    "dotted line",
    "skip connection",

    # L-shaped / elbow connectors (common in flowcharts)
    "L-shaped connector",
    "elbow arrow",
]
