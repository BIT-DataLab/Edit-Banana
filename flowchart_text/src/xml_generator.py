# """
# XML 合成引擎模块
# 生成 draw.io (mxGraph) 格式的 XML 文件
# """

# import xml.etree.ElementTree as ET
# from xml.dom import minidom
# from dataclasses import dataclass
# from typing import Optional
# from pathlib import Path
# import html

# import sys
# sys.path.append(str(Path(__file__).parent.parent))


# @dataclass
# class TextCellData:
#     """文本单元格数据"""
#     cell_id: int
#     text: str
#     x: float
#     y: float
#     width: float
#     height: float
#     font_size: float
#     is_latex: bool = False
#     rotation: float = 0.0


# class MxGraphXMLGenerator:
#     """
#     mxGraph XML 生成器
    
#     生成符合 draw.io 标准的 XML 文件
    
#     设计要求的样式:
#     - whiteSpace=nowrap; 不换行，保持单行显示
#     - autosize=1;        允许 draw.io 根据字号微调框高
#     - resizable=0;       锁定宽度，保持物理对齐
#     - html=1;            允许富文本和公式渲染
#     """
    
#     def __init__(self, diagram_name: str = "Page-1", 
#                  page_width: int = 1169, page_height: int = 827):
#         """
#         初始化 XML 生成器
        
#         Args:
#             diagram_name: 图表名称
#             page_width: 页面宽度（像素）
#             page_height: 页面高度（像素）
#         """
#         self.diagram_name = diagram_name
#         self.page_width = page_width
#         self.page_height = page_height
#         self.next_id = 2  # draw.io 保留 id 0 和 1
        
#     def _get_next_id(self) -> int:
#         """获取下一个可用的 cell ID"""
#         current_id = self.next_id
#         self.next_id += 1
#         return current_id
    
#     def _build_style_string(self, cell_data: TextCellData) -> str:
#         """
#         构建单元格样式字符串
        
#         Args:
#             cell_data: 文本单元格数据
            
#         Returns:
#             str: 样式字符串
#         """
#         styles = [
#             "text",  # 标记为文本类型
#             "html=1",  # 允许 HTML/富文本
#             "whiteSpace=nowrap",  # 不换行，保持单行显示
#             "autosize=1",  # 自动调整大小
#             "resizable=0",  # 锁定宽度
#             f"fontSize={int(cell_data.font_size)}",  # 字号
#             "align=center",  # 水平居中对齐
#             "verticalAlign=middle",  # 垂直居中
#             "overflow=visible",  # 允许溢出显示
#         ]
        
#         # 如果有旋转
#         if cell_data.rotation != 0:
#             styles.append(f"rotation={cell_data.rotation}")
        
#         return ";".join(styles) + ";"
    
#     def _escape_text(self, text: str, is_latex: bool = False) -> str:
#         """
#         转义文本内容以适应 draw.io HTML 格式
        
#         Args:
#             text: 原始文本
#             is_latex: 是否为 LaTeX 公式
            
#         Returns:
#             str: 转义后的文本
#         """
#         # HTML 转义
#         escaped = html.escape(text)
        
#         # 如果是 LaTeX，包装在 draw.io 的公式标签中
#         if is_latex:
#             # draw.io 使用特殊语法渲染 LaTeX
#             # 格式: `latex formula`
#             # 移除已有的 $ 符号
#             latex_content = escaped.replace("$", "").strip()
#             # 对于 draw.io，使用 MathJax 格式
#             escaped = f"\\({latex_content}\\)"
        
#         return escaped
    
#     def generate_xml(self, cells: list[TextCellData]) -> str:
#         """
#         生成完整的 mxGraph XML
        
#         Args:
#             cells: 文本单元格列表
            
#         Returns:
#             str: XML 字符串
#         """
#         # 创建根元素 mxfile
#         mxfile = ET.Element("mxfile")
#         mxfile.set("host", "app.diagrams.net")
#         mxfile.set("modified", "2024-01-01T00:00:00.000Z")
#         mxfile.set("agent", "OCR Vector Restore")
#         mxfile.set("version", "1.0.0")
#         mxfile.set("type", "device")
        
#         # 创建 diagram 元素
#         diagram = ET.SubElement(mxfile, "diagram")
#         diagram.set("name", self.diagram_name)
#         diagram.set("id", "diagram-1")
        
#         # 创建 mxGraphModel
#         graph_model = ET.SubElement(diagram, "mxGraphModel")
#         graph_model.set("dx", "0")
#         graph_model.set("dy", "0")
#         graph_model.set("grid", "1")
#         graph_model.set("gridSize", "10")
#         graph_model.set("guides", "1")
#         graph_model.set("tooltips", "1")
#         graph_model.set("connect", "1")
#         graph_model.set("arrows", "1")
#         graph_model.set("fold", "1")
#         graph_model.set("page", "1")
#         graph_model.set("pageScale", "1")
#         graph_model.set("pageWidth", str(self.page_width))
#         graph_model.set("pageHeight", str(self.page_height))
#         graph_model.set("math", "1")  # 启用数学公式支持
        
#         # 创建 root 元素
#         root = ET.SubElement(graph_model, "root")
        
#         # 添加必需的父级 cell（id=0 和 id=1）
#         cell_0 = ET.SubElement(root, "mxCell")
#         cell_0.set("id", "0")
        
#         cell_1 = ET.SubElement(root, "mxCell")
#         cell_1.set("id", "1")
#         cell_1.set("parent", "0")
        
#         # 添加文本单元格
#         for cell_data in cells:
#             self._add_text_cell(root, cell_data)
        
#         # 生成格式化的 XML 字符串
#         xml_string = ET.tostring(mxfile, encoding="unicode")
        
#         # 使用 minidom 格式化
#         dom = minidom.parseString(xml_string)
#         pretty_xml = dom.toprettyxml(indent="  ")
        
#         # 移除 minidom 添加的 XML 声明（draw.io 不需要）
#         lines = pretty_xml.split("\n")
#         if lines[0].startswith("<?xml"):
#             lines = lines[1:]
        
#         return "\n".join(lines)
    
#     def _add_text_cell(self, root: ET.Element, cell_data: TextCellData) -> None:
#         """
#         添加文本单元格到 root
        
#         Args:
#             root: root 元素
#             cell_data: 文本单元格数据
#         """
#         cell = ET.SubElement(root, "mxCell")
#         cell.set("id", str(cell_data.cell_id))
#         cell.set("value", self._escape_text(cell_data.text, cell_data.is_latex))
#         cell.set("style", self._build_style_string(cell_data))
#         cell.set("vertex", "1")
#         cell.set("parent", "1")
        
#         # 添加 geometry
#         geometry = ET.SubElement(cell, "mxGeometry")
#         geometry.set("x", str(round(cell_data.x, 2)))
#         geometry.set("y", str(round(cell_data.y, 2)))
#         geometry.set("width", str(round(cell_data.width, 2)))
#         geometry.set("height", str(round(cell_data.height, 2)))
#         geometry.set("as", "geometry")
    
#     def create_text_cell(
#         self,
#         text: str,
#         x: float,
#         y: float,
#         width: float,
#         height: float,
#         font_size: float,
#         is_latex: bool = False,
#         rotation: float = 0.0
#     ) -> TextCellData:
#         """
#         创建文本单元格数据
        
#         Args:
#             text: 文本内容
#             x: X 坐标
#             y: Y 坐标
#             width: 宽度
#             height: 高度
#             font_size: 字号
#             is_latex: 是否为 LaTeX
#             rotation: 旋转角度
            
#         Returns:
#             TextCellData: 单元格数据
#         """
#         return TextCellData(
#             cell_id=self._get_next_id(),
#             text=text,
#             x=x,
#             y=y,
#             width=width,
#             height=height,
#             font_size=font_size,
#             is_latex=is_latex,
#             rotation=rotation
#         )
    
#     def save_to_file(self, cells: list[TextCellData], output_path: str) -> None:
#         """
#         将 XML 保存到文件
        
#         Args:
#             cells: 文本单元格列表
#             output_path: 输出文件路径
#         """
#         xml_content = self.generate_xml(cells)
        
#         output_path = Path(output_path)
        
#         # 确保扩展名正确
#         if output_path.suffix.lower() != ".drawio":
#             output_path = output_path.with_suffix(".drawio")
        
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(xml_content)
        
#         print(f"已保存到: {output_path}")


# def create_generator(diagram_name: str = "Page-1") -> MxGraphXMLGenerator:
#     """
#     便捷函数：创建 XML 生成器
    
#     Args:
#         diagram_name: 图表名称
        
#     Returns:
#         MxGraphXMLGenerator: 生成器实例
#     """
#     return MxGraphXMLGenerator(diagram_name)


# if __name__ == "__main__":
#     # 测试代码
#     generator = MxGraphXMLGenerator("测试图表")
    
#     # 创建测试文本单元格
#     cells = [
#         generator.create_text_cell(
#             text="Hello World",
#             x=100, y=100,
#             width=120, height=30,
#             font_size=14
#         ),
#         generator.create_text_cell(
#             text="$x^2 + y^2 = r^2$",
#             x=100, y=150,
#             width=150, height=30,
#             font_size=12,
#             is_latex=True
#         ),
#         generator.create_text_cell(
#             text="CowID_1_Img_1",
#             x=100, y=200,
#             width=100, height=20,
#             font_size=10
#         ),
#     ]
    
#     # 生成 XML
#     xml = generator.generate_xml(cells)
#     print("生成的 XML:")
#     print(xml[:1000])  # 只显示前 1000 字符
    
#     # 保存到文件
#     # generator.save_to_file(cells, "test_output.drawio")

"""
XML 合成引擎模块
生成 draw.io (mxGraph) 格式的 XML 文件
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import html

import sys
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class TextCellData:
    """文本单元格数据"""
    cell_id: int
    text: str
    x: float
    y: float
    width: float
    height: float
    font_size: float
    is_latex: bool = False
    rotation: float = 0.0


class MxGraphXMLGenerator:
    """
    mxGraph XML 生成器
    
    生成符合 draw.io 标准的 XML 文件
    
    设计要求的样式:
    - whiteSpace=nowrap; 不换行，保持单行显示
    - autosize=1;        允许 draw.io 根据字号微调框高
    - resizable=0;       锁定宽度，保持物理对齐
    - html=1;            允许富文本和公式渲染
    """
    
    def __init__(self, diagram_name: str = "Page-1", 
                 page_width: int = 1169, page_height: int = 827):
        """
        初始化 XML 生成器
        
        Args:
            diagram_name: 图表名称
            page_width: 页面宽度（像素）
            page_height: 页面高度（像素）
        """
        self.diagram_name = diagram_name
        self.page_width = page_width
        self.page_height = page_height
        # 核心修改：绑定dx/dy与页面尺寸，对齐图形脚本画布属性
        self.dx = page_width
        self.dy = page_height
        self.next_id = 2  # draw.io 保留 id 0 和 1
        
    def _get_next_id(self) -> int:
        """获取下一个可用的 cell ID"""
        current_id = self.next_id
        self.next_id += 1
        return current_id
    
    def _build_style_string(self, cell_data: TextCellData) -> str:
        """
        构建单元格样式字符串
        
        Args:
            cell_data: 文本单元格数据
            
        Returns:
            str: 样式字符串
        """
        styles = [
            "text",  # 标记为文本类型
            "html=1",  # 允许 HTML/富文本
            "whiteSpace=nowrap",  # 不换行，保持单行显示
            "autosize=1",  # 自动调整大小
            "resizable=0",  # 锁定宽度
            f"fontSize={int(cell_data.font_size)}",  # 字号
            "align=center",  # 水平居中对齐
            "verticalAlign=middle",  # 垂直居中
            "overflow=visible",  # 允许溢出显示
        ]
        
        # 如果有旋转
        if cell_data.rotation != 0:
            styles.append(f"rotation={cell_data.rotation}")
        
        return ";".join(styles) + ";"
    
    def _escape_text(self, text: str, is_latex: bool = False) -> str:
        """
        转义文本内容以适应 draw.io HTML 格式
        
        Args:
            text: 原始文本
            is_latex: 是否为 LaTeX 公式
            
        Returns:
            str: 转义后的文本
        """
        # HTML 转义
        escaped = html.escape(text)
        
        # 如果是 LaTeX，包装在 draw.io 的公式标签中
        if is_latex:
            # draw.io 使用特殊语法渲染 LaTeX
            # 格式: `latex formula`
            # 移除已有的 $ 符号
            latex_content = escaped.replace("$", "").strip()
            # 对于 draw.io，使用 MathJax 格式
            # 使用 $$ 强制显示模式，通常比 \( \) 更可靠
            escaped = f"$${latex_content}$$"
        
        return escaped
    
    def generate_xml(self, cells: list[TextCellData]) -> str:
        """
        生成完整的 mxGraph XML
        
        Args:
            cells: 文本单元格列表
            
        Returns:
            str: XML 字符串
        """
        # 创建根元素 mxfile
        mxfile = ET.Element("mxfile")
        mxfile.set("host", "app.diagrams.net")
        mxfile.set("modified", "2024-01-01T00:00:00.000Z")
        mxfile.set("agent", "OCR Vector Restore")
        mxfile.set("version", "1.0.0")
        mxfile.set("type", "device")
        
        # 创建 diagram 元素
        diagram = ET.SubElement(mxfile, "diagram")
        diagram.set("name", self.diagram_name)
        diagram.set("id", "diagram-1")
        
        # 创建 mxGraphModel
        graph_model = ET.SubElement(diagram, "mxGraphModel")
        # 核心修改：替换硬编码的"0"，使用与页面尺寸一致的dx/dy
        graph_model.set("dx", str(self.dx))
        graph_model.set("dy", str(self.dy))
        graph_model.set("grid", "1")
        graph_model.set("gridSize", "10")
        graph_model.set("guides", "1")
        graph_model.set("tooltips", "1")
        graph_model.set("connect", "1")
        graph_model.set("arrows", "1")
        graph_model.set("fold", "1")
        graph_model.set("page", "1")
        graph_model.set("pageScale", "1")
        graph_model.set("pageWidth", str(self.page_width))
        graph_model.set("pageHeight", str(self.page_height))
        graph_model.set("math", "1")  # 启用数学公式支持
        # 补充兼容属性，对齐图形脚本，提升DrawIO打开兼容性
        graph_model.set("shadow", "0")
        graph_model.set("background", "#ffffff")
        
        # 创建 root 元素
        root = ET.SubElement(graph_model, "root")
        
        # 添加必需的父级 cell（id=0 和 id=1）
        cell_0 = ET.SubElement(root, "mxCell")
        cell_0.set("id", "0")
        
        cell_1 = ET.SubElement(root, "mxCell")
        cell_1.set("id", "1")
        cell_1.set("parent", "0")
        
        # 添加文本单元格
        for cell_data in cells:
            self._add_text_cell(root, cell_data)
        
        # 生成格式化的 XML 字符串
        xml_string = ET.tostring(mxfile, encoding="unicode")
        
        # 使用 minidom 格式化
        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # 移除 minidom 添加的 XML 声明（draw.io 不需要）
        lines = pretty_xml.split("\n")
        if lines[0].startswith("<?xml"):
            lines = lines[1:]
        
        return "\n".join(lines)
    
    def _add_text_cell(self, root: ET.Element, cell_data: TextCellData) -> None:
        """
        添加文本单元格到 root
        
        Args:
            root: root 元素
            cell_data: 文本单元格数据
        """
        cell = ET.SubElement(root, "mxCell")
        cell.set("id", str(cell_data.cell_id))
        cell.set("value", self._escape_text(cell_data.text, cell_data.is_latex))
        cell.set("style", self._build_style_string(cell_data))
        cell.set("vertex", "1")
        cell.set("parent", "1")
        
        # 添加 geometry
        geometry = ET.SubElement(cell, "mxGeometry")
        geometry.set("x", str(round(cell_data.x, 2)))
        geometry.set("y", str(round(cell_data.y, 2)))
        geometry.set("width", str(round(cell_data.width, 2)))
        geometry.set("height", str(round(cell_data.height, 2)))
        geometry.set("as", "geometry")
    
    def create_text_cell(
        self,
        text: str,
        x: float,
        y: float,
        width: float,
        height: float,
        font_size: float,
        is_latex: bool = False,
        rotation: float = 0.0
    ) -> TextCellData:
        """
        创建文本单元格数据
        
        Args:
            text: 文本内容
            x: X 坐标
            y: Y 坐标
            width: 宽度
            height: 高度
            font_size: 字号
            is_latex: 是否为 LaTeX
            rotation: 旋转角度
            
        Returns:
            TextCellData: 单元格数据
        """
        return TextCellData(
            cell_id=self._get_next_id(),
            text=text,
            x=x,
            y=y,
            width=width,
            height=height,
            font_size=font_size,
            is_latex=is_latex,
            rotation=rotation
        )
    
    def save_to_file(self, cells: list[TextCellData], output_path: str) -> None:
        """
        将 XML 保存到文件
        
        Args:
            cells: 文本单元格列表
            output_path: 输出文件路径
        """
        xml_content = self.generate_xml(cells)
        
        output_path = Path(output_path)
        
        # 核心修改：统一文件扩展名为 .drawio.xml，与图形脚本对齐
        if not output_path.name.lower().endswith(".drawio.xml"):
            output_path = output_path.with_suffix("").with_suffix(".drawio.xml")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        
        print(f"已保存到: {output_path}")


def create_generator(diagram_name: str = "Page-1") -> MxGraphXMLGenerator:
    """
    便捷函数：创建 XML 生成器
    
    Args:
        diagram_name: 图表名称
        
    Returns:
        MxGraphXMLGenerator: 生成器实例
    """
    return MxGraphXMLGenerator(diagram_name)


if __name__ == "__main__":
    # 测试代码
    generator = MxGraphXMLGenerator("测试图表")
    
    # 创建测试文本单元格
    cells = [
        generator.create_text_cell(
            text="Hello World",
            x=100, y=100,
            width=120, height=30,
            font_size=14
        ),
        generator.create_text_cell(
            text="$x^2 + y^2 = r^2$",
            x=100, y=150,
            width=150, height=30,
            font_size=12,
            is_latex=True
        ),
        generator.create_text_cell(
            text="CowID_1_Img_1",
            x=100, y=200,
            width=100, height=20,
            font_size=10
        ),
    ]
    
    # 生成 XML
    xml = generator.generate_xml(cells)
    print("生成的 XML:")
    print(xml[:1000])  # 只显示前 1000 字符
    
    # 保存到文件
    # generator.save_to_file(cells, "test_output.drawio.xml")