"""
颜色工具：RGB <-> 十六进制互转
"""

from typing import Tuple, Union


def rgb_to_hex(r: Union[int, Tuple[int, int, int]], g: int = None, b: int = None) -> str:
    """RGB -> #RRGGBB"""
    if g is not None and b is not None:
        r, g, b = int(r), int(g), int(b)
    else:
        r, g, b = int(r[0]), int(r[1]), int(r[2])
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b)),
    )


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """#RRGGBB -> (r, g, b)"""
    h = hex_str.lstrip("#")
    if len(h) == 6:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    if len(h) == 3:
        return (int(h[0] * 2, 16), int(h[1] * 2, 16), int(h[2] * 2, 16))
    return (0, 0, 0)
