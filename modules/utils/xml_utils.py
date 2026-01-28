"""
XML 工具：DrawIO mxCell / mxGeometry 构建、解析、格式化
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional, Any, Dict


def create_mxcell(
    cell_id: str,
    parent: str = "1",
    value: str = "",
    style: str = "",
    vertex: bool = True,
    edge: bool = False,
    **attrs: Any,
) -> ET.Element:
    """创建 mxCell 元素"""
    d = {"id": str(cell_id), "parent": str(parent), "value": value or "", "style": style or ""}
    if vertex:
        d["vertex"] = "1"
    if edge:
        d["edge"] = "1"
    d.update({k.replace("_", ""): str(v) for k, v in attrs.items()})
    return ET.Element("mxCell", d)


def create_geometry(
    x: float = 0,
    y: float = 0,
    width: float = 0,
    height: float = 0,
    relative: bool = False,
    as_type: str = "geometry",
) -> ET.Element:
    """创建 mxGeometry 元素"""
    g = ET.Element("mxGeometry", {"as": as_type, "x": str(x), "y": str(y), "width": str(width), "height": str(height)})
    if relative:
        g.set("relative", "1")
    return g


def prettify_xml(elem: ET.Element) -> str:
    """格式化 XML 输出（移除声明、过滤空行）"""
    rough = ET.tostring(elem, encoding="utf-8").decode("utf-8")
    try:
        parsed = minidom.parseString(rough)
        lines = parsed.toprettyxml(indent="  ").split("\n")
        return "\n".join(
            line for line in lines
            if line.strip() and not line.strip().startswith("<?xml")
        )
    except Exception:
        return rough


def parse_drawio_xml(path_or_string: str) -> ET.Element:
    """解析 DrawIO XML 文件或字符串，返回根元素"""
    if path_or_string.strip().startswith("<"):
        return ET.fromstring(path_or_string)
    tree = ET.parse(path_or_string)
    return tree.getroot()
