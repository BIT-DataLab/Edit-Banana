# Background/container prompts for SAM3
#
# Design principles (from SAM3 best practices):
#   - Short, specific noun phrases
#   - Target large container regions (panels, sub-figures, legend groups)
#   - These get priority=1 (lowest) so they don't overshadow inner elements
#
# These are used for:
#   1. Detecting sub-figure panels (a), (b), (c), (d), (e) in scientific figures
#   2. Providing the BACKGROUND layer in SVG/DrawIO output
BACKGROUND_PROMPT = [
    # Core panel / container types
    "panel",
    "sub-figure panel",
    "container",
    "background region",

    # Filled / shaded regions
    "filled region",
    "shaded region",
    "colored background",

    # Bordered / grouped sections
    "dashed border rectangle",
    "bordered section",
    "grouped section",

    # Legend-specific (common in scientific figures with color-coded legends)
    "color legend",
    "legend box",
    "legend panel",

    # Title bars / headers
    "title bar",
    "header strip",
]
