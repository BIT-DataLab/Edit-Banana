# Image/icon prompts for SAM3
#
# Design principles (from SAM3 best practices):
#   - Short, specific noun phrases
#   - Include scientific/medical figure icons specifically (person, scan, render)
#   - Cover the full spectrum: icons, photos, renders, charts, illustrations
#
# The IconPictureProcessor crops these regions and (optionally) removes
# backgrounds with RMBG-2.0. Outputs are embedded as base64 in vectors.
IMAGE_PROMPT = [
    # Generic image categories
    "icon",
    "picture",
    "photograph",
    "logo",
    "illustration",

    # Data visualizations
    "chart",
    "graph",
    "plot",
    "diagram",

    # Scientific figure icons (were missed in v1 — CCT-FM panel had many)
    "person icon",
    "human silhouette",
    "group of people",
    "crowd icon",

    # Medical imaging (for figures like CT pipelines, Bern CCTA, etc.)
    "medical scan",
    "CT scan image",
    "MRI image",
    "ultrasound image",
    "anatomy rendering",
    "3D heart model",
    "3D rendering",

    # Medical icons
    "stethoscope icon",
    "heart icon",

    # Stacked / layered images (for dataset thumbnails)
    "stack of images",
    "image stack",
    "thumbnail strip",

    # Texture / pattern (for masking blocks, checkerboards)
    "checkerboard pattern",
    "grid pattern",

    # Computer/UI icons (for HITL / expert correction figures)
    "computer monitor icon",
    "screen icon",
]
