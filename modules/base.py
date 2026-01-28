"""
模块基类定义

所有处理模块都应该继承 BaseProcessor，实现统一的接口。
这样可以保证不同开发者的模块能够无缝协作。

重构说明：
    - 每个子模块负责生成自己的XML代码
    - 子模块处理完成后，设置 element.xml_fragment 和 element.layer_level
    - XMLMerger只负责收集和合并，不负责生成样式
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field
import os

# 避免循环导入
if TYPE_CHECKING:
    from .data_types import ElementInfo, ProcessingResult, ProcessingConfig, XMLFragment


@dataclass
class ProcessingContext:
    """
    处理上下文 - 在整个处理流程中共享的数据
    
    每个模块可以从context中读取需要的数据，也可以往里写入处理结果。
    这样不同模块之间可以通过context传递数据。
    """
    # === 输入信息 ===
    image_path: str                              # 原始图片路径
    canvas_width: int = 0                        # 画布宽度
    canvas_height: int = 0                       # 画布高度
    
    # === 元素数据（各模块共享） ===
    elements: List['ElementInfo'] = field(default_factory=list)
    
    # === XML片段（各子模块生成，XMLMerger合并） ===
    xml_fragments: List['XMLFragment'] = field(default_factory=list)
    
    # === 配置 ===
    config: Optional['ProcessingConfig'] = None
    output_dir: str = "./output"
    
    # === 模型实例（可共享，避免重复加载） ===
    shared_models: Dict[str, Any] = field(default_factory=dict)
    
    # === 中间结果（用于debug或二次处理） ===
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    def get_elements_by_type(self, element_type: str) -> List['ElementInfo']:
        """按类型获取元素"""
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_elements_without_xml(self) -> List['ElementInfo']:
        """获取还没有生成XML的元素"""
        return [e for e in self.elements if not e.has_xml()]
    
    def get_elements_with_xml(self) -> List['ElementInfo']:
        """获取已经生成XML的元素"""
        return [e for e in self.elements if e.has_xml()]
    
    def add_xml_fragment(self, fragment: 'XMLFragment'):
        """添加XML片段"""
        self.xml_fragments.append(fragment)


class BaseProcessor(ABC):
    """
    处理模块基类
    
    所有处理模块都应该继承此类并实现 process() 方法。
    
    子模块职责：
        1. 处理自己负责的元素类型
        2. 生成对应的mxCell XML代码
        3. 设置 element.xml_fragment 和 element.layer_level
    
    使用示例:
        class MyProcessor(BaseProcessor):
            def process(self, context: ProcessingContext) -> ProcessingResult:
                for elem in context.elements:
                    if self._should_process(elem):
                        xml = self._generate_xml(elem)
                        elem.xml_fragment = xml
                        elem.layer_level = LayerLevel.BASIC_SHAPE.value
                return ProcessingResult(success=True, elements=context.elements)
    """
    
    def __init__(self, config: Optional['ProcessingConfig'] = None):
        """
        初始化处理器
        
        Args:
            config: 处理配置，如果为None则使用默认配置
        """
        from .data_types import ProcessingConfig
        self.config = config or ProcessingConfig()
        self._model = None  # 延迟加载的模型实例
    
    @property
    def name(self) -> str:
        """模块名称（用于日志和标识）"""
        return self.__class__.__name__
    
    @abstractmethod
    def process(self, context: ProcessingContext) -> 'ProcessingResult':
        """
        处理入口 - 子类必须实现
        
        Args:
            context: 处理上下文，包含输入数据和共享状态
            
        Returns:
            ProcessingResult: 处理结果
            
        子类实现要点：
            1. 遍历 context.elements，找到需要处理的元素
            2. 生成 mxCell XML字符串
            3. 设置 element.xml_fragment = xml_string
            4. 设置 element.layer_level = 合适的层级
            5. 返回 ProcessingResult
        """
        pass
    
    def load_model(self):
        """
        加载模型 - 子类可以重写以实现模型加载
        
        模型加载应该是懒加载的，只在第一次需要时加载。
        调用模型时应该就是简单的一行代码。
        """
        pass
    
    def unload_model(self):
        """卸载模型（释放GPU内存）"""
        self._model = None
    
    def _ensure_output_dir(self, output_dir: str):
        """确保输出目录存在"""
        os.makedirs(output_dir, exist_ok=True)
    
    def _log(self, message: str):
        """简单的日志输出"""
        print(f"[{self.name}] {message}")
    
    # ======================== XML生成辅助方法 ========================
    
    def _create_mxcell_xml(self, 
                           cell_id: int,
                           style: str,
                           x: int, 
                           y: int, 
                           width: int, 
                           height: int,
                           value: str = "",
                           parent: str = "1") -> str:
        """
        生成mxCell XML字符串 - 供子模块使用
        
        Args:
            cell_id: 临时ID（合并时会被重新分配）
            style: DrawIO样式字符串
            x, y: 位置
            width, height: 尺寸
            value: 单元格值（通常为空或文本内容）
            parent: 父元素ID
            
        Returns:
            mxCell XML字符串
        """
        # 转义特殊字符
        value = value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        style = style.replace('"', '&quot;')
        
        xml = f'''<mxCell id="{cell_id}" parent="{parent}" vertex="1" value="{value}" style="{style}">
  <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/>
</mxCell>'''
        
        return xml


class ModelWrapper:
    """
    模型封装基类
    
    为了让"调用模型时应该就是一句话"，我们将模型封装成简单的接口。
    
    使用示例:
        model = SAM3Model()
        result = model.predict(image, prompts)  # 一行调用
    """
    
    def __init__(self):
        self._model = None
        self._is_loaded = False
    
    @abstractmethod
    def load(self):
        """加载模型"""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        """模型推理"""
        pass
    
    def unload(self):
        """卸载模型"""
        self._model = None
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
