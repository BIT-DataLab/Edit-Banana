"""
统一的数据结构定义

这个文件定义了各模块之间传递数据的标准格式，
不同的开发者在实现自己的模块时，都应该使用这些数据结构。

重构说明：
    - 每个子模块负责生成自己的XML代码
    - XMLMerger只负责按层级合并，不负责生成样式
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ElementType(Enum):
    """元素类型枚举"""
    # 基本图形类（可矢量化）
    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    DIAMOND = "diamond"
    ELLIPSE = "ellipse"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"
    PARALLELOGRAM = "parallelogram"
    CYLINDER = "cylinder"
    CLOUD = "cloud"
    ACTOR = "actor"
    TITLE_BAR = "title_bar"
    SECTION_PANEL = "section_panel"
    
    # 非基本图形类（需转base64）
    ICON = "icon"
    PICTURE = "picture"
    LOGO = "logo"
    CHART = "chart"
    FUNCTION_GRAPH = "function_graph"
    
    # 箭头/连接线
    ARROW = "arrow"
    LINE = "line"
    CONNECTOR = "connector"
    
    # 其他
    TEXT = "text"
    UNKNOWN = "unknown"


class LayerLevel(Enum):
    """
    层级定义（从底到顶）
    
    数值越小越在底层，越大越在顶层
    """
    BACKGROUND = 0      # 背景/大面积容器（section_panel, title_bar）
    BASIC_SHAPE = 1     # 基本图形（rectangle, ellipse等）
    IMAGE = 2           # 图片类（icon, picture）
    ARROW = 3           # 箭头/连接线
    TEXT = 4            # 文字（最上层）
    OTHER = 5           # 其他（默认最上层）


@dataclass
class BoundingBox:
    """边界框"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, coords: List[int]) -> 'BoundingBox':
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])


@dataclass
class ElementInfo:
    """
    元素信息结构 - 所有模块之间传递元素的标准格式
    
    SAM3提取后会填充: id, element_type, bbox, polygon, score, mask
    各子模块处理后填充: xml_fragment, layer_level
    """
    # === 基础信息（SAM3提取） ===
    id: int                                      # 元素唯一ID
    element_type: str                            # 元素类型（字符串，对应ElementType）
    bbox: BoundingBox                            # 边界框
    score: float = 0.0                           # 置信度分数
    polygon: List[List[int]] = field(default_factory=list)  # 轮廓多边形
    mask: Optional[Any] = None                   # 二值掩码（numpy array，可选）
    
    # === XML输出（各子模块生成） ===
    xml_fragment: Optional[str] = None           # ⭐ 该元素的完整mxCell XML字符串
    layer_level: int = LayerLevel.OTHER.value    # 层级（用于合并时排序）
    
    # === 以下字段保留，供子模块内部使用 ===
    base64: Optional[str] = None                 # Base64编码的图像数据（供Icon模块内部用）
    fill_color: Optional[str] = None             # 填充颜色（供BasicShape模块内部用）
    stroke_color: Optional[str] = None           # 描边颜色（供BasicShape模块内部用）
    stroke_width: int = 1                        # 描边宽度
    arrow_start: Optional[tuple] = None          # 箭头起点（供Arrow模块内部用）
    arrow_end: Optional[tuple] = None            # 箭头终点
    vector_points: Optional[List[List[int]]] = None  # 矢量箭头路径点 [[x,y], [x,y], ...]
    arrow_style: Optional[str] = None            # 箭头样式（classic, open等）
    
    # === 元数据 ===
    source_prompt: Optional[str] = None          # 触发此元素识别的prompt
    processing_notes: List[str] = field(default_factory=list)  # 处理过程中的备注
    
    def has_xml(self) -> bool:
        """检查是否已生成XML"""
        return self.xml_fragment is not None and len(self.xml_fragment.strip()) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于JSON序列化，不包含xml_fragment）"""
        return {
            'id': self.id,
            'element_type': self.element_type,
            'bbox': self.bbox.to_list(),
            'score': self.score,
            'polygon': self.polygon,
            'layer_level': self.layer_level,
            'has_xml': self.has_xml(),
            'source_prompt': self.source_prompt,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ElementInfo':
        """从字典创建"""
        bbox = BoundingBox.from_list(data['bbox'])
        return cls(
            id=data['id'],
            element_type=data['element_type'],
            bbox=bbox,
            score=data.get('score', 0.0),
            polygon=data.get('polygon', []),
            layer_level=data.get('layer_level', LayerLevel.OTHER.value),
            source_prompt=data.get('source_prompt'),
        )


@dataclass
class XMLFragment:
    """
    XML片段结构 - 用于XMLMerger合并
    
    每个子模块处理完成后，应该生成XMLFragment对象
    """
    element_id: int                    # 关联的元素ID
    xml_content: str                   # 完整的mxCell XML字符串
    layer_level: int                   # 层级（数值越小越底层）
    bbox: Optional[BoundingBox] = None # 位置信息（用于同层级内按位置/面积排序）
    element_type: str = "unknown"      # 元素类型（用于调试）
    
    @property
    def area(self) -> int:
        """获取面积（用于排序）"""
        if self.bbox:
            return self.bbox.area
        return 0


@dataclass 
class ProcessingResult:
    """
    处理结果结构 - 每个模块处理完成后返回的标准格式
    """
    success: bool                                         # 处理是否成功
    elements: List[ElementInfo] = field(default_factory=list)  # 处理后的元素列表
    xml_fragments: List[XMLFragment] = field(default_factory=list)  # ⭐ XML片段列表
    canvas_width: int = 0                                 # 画布宽度
    canvas_height: int = 0                                # 画布高度
    error_message: Optional[str] = None                   # 错误信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def get_elements_by_type(self, element_type: str) -> List[ElementInfo]:
        """按类型筛选元素"""
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_all_xml_fragments(self) -> List[XMLFragment]:
        """
        获取所有XML片段（优先从xml_fragments，否则从elements提取）
        """
        if self.xml_fragments:
            return self.xml_fragments
        
        # 从elements中提取有xml_fragment的
        fragments = []
        for elem in self.elements:
            if elem.has_xml():
                fragments.append(XMLFragment(
                    element_id=elem.id,
                    xml_content=elem.xml_fragment,
                    layer_level=elem.layer_level,
                    bbox=elem.bbox,
                    element_type=elem.element_type
                ))
        return fragments
    
    def add_element(self, element: ElementInfo):
        """添加元素"""
        self.elements.append(element)
        
    def add_xml_fragment(self, fragment: XMLFragment):
        """添加XML片段"""
        self.xml_fragments.append(fragment)


@dataclass
class ProcessingConfig:
    """处理配置"""
    # SAM3配置
    score_threshold: float = 0.5
    min_area: int = 100
    epsilon_factor: float = 0.02
    
    # 输出配置
    output_dir: str = "./output"
    save_intermediate: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ProcessingConfig':
        """从YAML文件加载配置"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return cls(
            score_threshold=config.get('sam3', {}).get('score_threshold', 0.5),
            min_area=config.get('sam3', {}).get('min_area', 100),
            epsilon_factor=config.get('sam3', {}).get('epsilon_factor', 0.02),
            output_dir=config.get('paths', {}).get('output_dir', './output'),
        )


# ======================== 辅助函数 ========================
def get_layer_level(element_type: str) -> int:
    """
    根据元素类型获取默认层级
    
    供各子模块使用，确保层级分配一致
    """
    element_type = element_type.lower()
    
    # 背景/容器类（最底层）
    if element_type in {'section_panel', 'title_bar'}:
        return LayerLevel.BACKGROUND.value
    
    # 箭头/连接线
    if element_type in {'arrow', 'line', 'connector'}:
        return LayerLevel.ARROW.value
    
    # 文字
    if element_type == 'text':
        return LayerLevel.TEXT.value
    
    # 图片类
    if element_type in {'icon', 'picture', 'image', 'logo', 'chart', 'function_graph'}:
        return LayerLevel.IMAGE.value
    
    # 基本图形
    if element_type in {
        'rectangle', 'rounded_rectangle', 'rounded rectangle',
        'diamond', 'ellipse', 'circle', 'cylinder', 'cloud',
        'hexagon', 'triangle', 'parallelogram', 'actor'
    }:
        return LayerLevel.BASIC_SHAPE.value
    
    # 其他
    return LayerLevel.OTHER.value
