"""
XML 生成器模块

功能：
    生成 draw.io (mxGraph) 格式的 XML 文件。
    将处理后的文字块转换为可在 draw.io 中编辑的矢量文本。

输出格式：
    draw.io 使用 mxGraph 格式，结构如下：
    
    <mxfile>
      <diagram>
        <mxGraphModel>
          <root>
            <mxCell id="0"/>                    <!-- 根节点 -->
            <mxCell id="1" parent="0"/>         <!-- 图层 -->
            <mxCell id="2" value="文字" ...>   <!-- 文本单元格 -->
              <mxGeometry x="100" y="200" width="80" height="20"/>
            </mxCell>
          </root>
        </mxGraphModel>
      </diagram>
    </mxfile>

样式支持：
    - fontSize: 字号（pt）
    - fontStyle: 1=粗体, 2=斜体, 3=粗斜体
    - fontColor: 颜色（如 #1d1d1d）
    - fontFamily: 字体名称
    - rotation: 旋转角度

公式支持：
    使用 MathJax 格式: \\(LaTeX公式\\)
    需要在 mxGraphModel 中设置 math="1"
"""

import html
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class TextCellData:
    """
    文本单元格数据
    
    存储单个文本框的所有信息。
    """
    cell_id: int                           # 单元格 ID（唯一）
    text: str                              # 文字内容
    x: float                               # 左上角 X 坐标
    y: float                               # 左上角 Y 坐标
    width: float                           # 宽度
    height: float                          # 高度
    font_size: float                       # 字号（pt）
    is_latex: bool = False                 # 是否为 LaTeX 公式
    rotation: float = 0.0                  # 旋转角度
    font_weight: Optional[str] = None      # 字重: normal, bold
    font_style: Optional[str] = None       # 样式: normal, italic
    font_color: Optional[str] = None       # 颜色（十六进制）
    font_family: Optional[str] = None      # 字体名称


class MxGraphXMLGenerator:
    """
    mxGraph XML 生成器
    
    使用示例：
        generator = MxGraphXMLGenerator()
        cell = generator.create_text_cell("Hello", 100, 200, 80, 20, 12)
        generator.save_to_file([cell], "output.drawio")
    """
    
    def __init__(self, diagram_name: str = "Page-1", 
                 page_width: int = 1169, page_height: int = 827):
        """
        初始化生成器
        
        Args:
            diagram_name: 图表名称
            page_width: 页面宽度
            page_height: 页面高度
        """
        self.diagram_name = diagram_name
        self.page_width = page_width
        self.page_height = page_height
        self.next_id = 2  # ID 从 2 开始（0 和 1 被根节点和图层占用）
        
    def _get_next_id(self) -> int:
        """获取下一个可用 ID"""
        current_id = self.next_id
        self.next_id += 1
        return current_id
    
    def _build_style_string(self, cell_data: TextCellData) -> str:
        """
        构建 mxCell 样式字符串
        
        draw.io 使用分号分隔的键值对表示样式：
        style="text;fontSize=12;fontStyle=1;fontColor=#000000;"
        """
        styles = [
            "text", "html=1", "whiteSpace=nowrap", "autosize=1", "resizable=0",
            f"fontSize={int(cell_data.font_size)}", "align=center",
            "verticalAlign=middle", "overflow=visible",
        ]
        
        # 字体样式：1=粗体, 2=斜体, 3=粗斜体
        font_style_value = 0
        if cell_data.font_weight == 'bold':
            font_style_value += 1
        if cell_data.font_style == 'italic':
            font_style_value += 2
        if font_style_value > 0:
            styles.append(f"fontStyle={font_style_value}")
        
        # 字体颜色
        if cell_data.font_color:
            styles.append(f"fontColor={cell_data.font_color}")
        
        # 字体名称
        if cell_data.font_family:
            first_font = cell_data.font_family.split(",")[0].strip()
            styles.append(f"fontFamily={first_font}")
        
        # 旋转角度
        if cell_data.rotation != 0:
            styles.append(f"rotation={cell_data.rotation}")
        
        return ";".join(styles) + ";"
    
    def _escape_text(self, text: str, is_latex: bool = False) -> str:
        """
        转义文本内容
        
        - HTML 特殊字符转义
        - LaTeX 公式转换为 MathJax 格式 \\(公式\\)
        """
        escaped = html.escape(text)
        
        if is_latex:
            # 移除 $ 符号，转换为 MathJax 格式
            latex_content = escaped.replace("$", "").strip()
            escaped = f"\\({latex_content}\\)"
        
        return escaped
    
    def generate_xml(self, cells: list[TextCellData]) -> str:
        """
        生成完整的 draw.io XML
        
        Args:
            cells: 文本单元格列表
            
        Returns:
            XML 字符串
        """
        # 创建根元素
        mxfile = ET.Element("mxfile")
        mxfile.set("host", "app.diagrams.net")
        mxfile.set("modified", "2024-01-01T00:00:00.000Z")
        mxfile.set("agent", "OCR Vector Restore")
        mxfile.set("version", "1.0.0")
        mxfile.set("type", "device")
        
        # 创建 diagram
        diagram = ET.SubElement(mxfile, "diagram")
        diagram.set("name", self.diagram_name)
        diagram.set("id", "diagram-1")
        
        # 创建 mxGraphModel
        graph_model = ET.SubElement(diagram, "mxGraphModel")
        graph_model.set("dx", "0")
        graph_model.set("dy", "0")
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
        graph_model.set("math", "1")  # 启用数学公式渲染
        
        # 创建 root
        root = ET.SubElement(graph_model, "root")
        
        # 添加必需的基础单元格
        cell_0 = ET.SubElement(root, "mxCell")
        cell_0.set("id", "0")
        
        cell_1 = ET.SubElement(root, "mxCell")
        cell_1.set("id", "1")
        cell_1.set("parent", "0")
        
        # 添加所有文本单元格
        for cell_data in cells:
            self._add_text_cell(root, cell_data)
        
        # 格式化 XML
        xml_string = ET.tostring(mxfile, encoding="unicode")
        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # 移除 XML 声明
        lines = pretty_xml.split("\n")
        if lines[0].startswith("<?xml"):
            lines = lines[1:]
        
        return "\n".join(lines)
    
    def _add_text_cell(self, root: ET.Element, cell_data: TextCellData) -> None:
        """添加文本单元格到 XML"""
        cell = ET.SubElement(root, "mxCell")
        cell.set("id", str(cell_data.cell_id))
        cell.set("value", self._escape_text(cell_data.text, cell_data.is_latex))
        cell.set("style", self._build_style_string(cell_data))
        cell.set("vertex", "1")
        cell.set("parent", "1")
        
        # 添加几何信息
        geometry = ET.SubElement(cell, "mxGeometry")
        geometry.set("x", str(round(cell_data.x, 2)))
        geometry.set("y", str(round(cell_data.y, 2)))
        geometry.set("width", str(round(cell_data.width, 2)))
        geometry.set("height", str(round(cell_data.height, 2)))
        geometry.set("as", "geometry")
    
    def create_text_cell(
        self, text: str, x: float, y: float, width: float, height: float,
        font_size: float, is_latex: bool = False, rotation: float = 0.0,
        font_weight: Optional[str] = None, font_style: Optional[str] = None,
        font_color: Optional[str] = None, font_family: Optional[str] = None,
        is_bold: bool = False, is_italic: bool = False
    ) -> TextCellData:
        """
        创建文本单元格
        
        Args:
            text: 文字内容
            x, y: 左上角坐标
            width, height: 尺寸
            font_size: 字号（pt）
            is_latex: 是否为公式
            rotation: 旋转角度
            font_weight: 字重 ("normal", "bold")
            font_style: 样式 ("normal", "italic")
            font_color: 颜色 (十六进制，如 "#1d1d1d")
            font_family: 字体名称 (如 "Arial", "Times New Roman")
            is_bold: 是否加粗（与 font_weight="bold" 等效）
            is_italic: 是否斜体（与 font_style="italic" 等效）
            
        Returns:
            TextCellData 对象
        """
        # 兼容 is_bold/is_italic 参数
        if is_bold and font_weight is None:
            font_weight = "bold"
        if is_italic and font_style is None:
            font_style = "italic"
        
        # 默认字体
        if font_family is None:
            font_family = "Arial"
        
        return TextCellData(
            cell_id=self._get_next_id(),
            text=text, x=x, y=y, width=width, height=height,
            font_size=font_size, is_latex=is_latex, rotation=rotation,
            font_weight=font_weight, font_style=font_style,
            font_color=font_color, font_family=font_family
        )
    
    def save_to_file(self, cells: list[TextCellData], output_path: str) -> None:
        """
        保存到文件
        
        Args:
            cells: 文本单元格列表
            output_path: 输出文件路径（自动添加 .drawio 后缀）
        """
        xml_content = self.generate_xml(cells)
        output_path = Path(output_path)
        
        if output_path.suffix.lower() != ".drawio":
            output_path = output_path.with_suffix(".drawio")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 删除旧文件
        if output_path.exists():
            output_path.unlink()
            print(f"已删除旧文件: {output_path}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        
        print(f"已保存到: {output_path}")
