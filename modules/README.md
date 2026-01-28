# 图片转可编辑形式 - 模块化处理框架

## 📁 项目结构

```
new_pipeline_editbanana/
├── main.py                 # 🚀 主入口脚本
├── requirements.txt        # 📦 依赖列表
├── config/
│   └── config.yaml         # ⚙️ 配置文件
├── input/                  # 📥 输入图片目录
├── output/                 # 📤 输出结果目录
├── models/
│   └── rmbg/model.onnx     # 🤖 RMBG模型
├── modules/                # 📚 处理模块
│   ├── __init__.py
│   ├── base.py             # 基类定义
│   ├── data_types.py       # 数据结构
│   ├── sam3_info_extractor.py    # SAM3提取
│   ├── icon_picture_processor.py # Icon/Picture处理
│   ├── basic_shape_processor.py  # 基本图形处理
│   ├── arrow_processor.py        # 箭头处理
│   ├── xml_merger.py             # XML合并
│   ├── metric_evaluator.py       # 质量评估
│   ├── refinement_processor.py   # 二次处理
│   ├── text/                     # 文字处理模块
│   │   ├── restorer.py           # TextRestorer 主入口
│   │   ├── ocr/                  # OCR模块
│   │   └── processors/           # 字体/样式处理器
│   └── utils/
│       ├── color_utils.py   # 颜色工具
│       ├── image_utils.py   # 图像工具
│       ├── xml_utils.py     # XML工具
│       └── drawio_library.py # DrawIO样式库
└── sam3/                   # SAM3模型代码
```

## 🔄 处理流程

```
                        输入图片
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  任务1: SAM3提取信息                                  │
    │  模块: Sam3InfoExtractor                             │
    │  输出: List[ElementInfo] (bbox, mask, type, score)   │
    └──────────────────────────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 任务2: Icon  │   │ 任务3: 图形  │   │ 任务5: 箭头  │
│ Picture处理  │   │ 基本图形处理  │   │   箭头处理   │
│              │   │              │   │              │
│ - 去背景     │   │ - 提取颜色   │   │ - 方向检测   │
│ - 转base64   │   │ - 生成XML    │   │ - 生成XML    │
│ - 生成XML    │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  任务6: XMLMerger                                     │
    │  - 收集所有 xml_fragment                             │
    │  - 按 layer_level 排序                               │
    │  - 生成完整DrawIO XML                                │
    └──────────────────────────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  任务7: MetricEvaluator (可选)                        │
    │  - 评估转换质量                                       │
    │  - 识别问题区域                                       │
    └──────────────────────────────────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  任务8: RefinementProcessor (可选)                    │
    │  - 处理问题区域                                       │
    │  - 补充遗漏元素                                       │
    └──────────────────────────────────────────────────────┘
                           │
                           ▼
                      输出 DrawIO XML
```

## 👥 多人协作分工

| 任务 | 模块 | 状态 | 输入 | 输出 |
|------|------|------|------|------|
| 1 | `Sam3InfoExtractor` | ✅ 已实现 | image_path | List[ElementInfo] |
| 2 | `IconPictureProcessor` | ✅ 已实现 | ElementInfo + image | xml_fragment |
| 3 | `BasicShapeProcessor` | ✅ 已实现 | ElementInfo + image | xml_fragment |
| 4 | `ArrowProcessor` | ✅ 已实现 | ElementInfo | xml_fragment |
| 5 | `XMLMerger` | ✅ 已实现 | List[XMLFragment] | .drawio.xml |
| 6 | `MetricEvaluator` | ✅ 已实现 | ElementInfo + image | bad_regions |
| 7 | `RefinementProcessor` | ✅ 已实现 | bad_regions | xml_fragment |
| 8 | `TextRestorer` | ✅ 已实现 | image_path | xml_fragment |

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 处理图片

```bash
# 处理单张图片
python main.py -i input/flowchart.png

# 批量处理 input/ 目录
python main.py

# 使用VLM迭代模式（更完整但更慢）
python main.py -i input/flowchart.png --iterative

# 启用质量评估和二次处理
python main.py -i input/flowchart.png --refine
```

### 代码调用

```python
from modules import (
    Sam3InfoExtractor,
    IconPictureProcessor,
    BasicShapeProcessor,
    ArrowProcessor,
    XMLMerger,
    ProcessingContext,
)

# 创建上下文
context = ProcessingContext(image_path="test.png")

# 步骤1: SAM3提取
extractor = Sam3InfoExtractor()
result = extractor.process(context)
context.elements = result.elements
context.canvas_width = result.canvas_width
context.canvas_height = result.canvas_height

# 步骤2: Icon/Picture处理
icon_processor = IconPictureProcessor()
icon_processor.process(context)

# 步骤3: 基本图形处理
shape_processor = BasicShapeProcessor()
shape_processor.process(context)

# 步骤4: 箭头处理
arrow_processor = ArrowProcessor()
arrow_processor.process(context)

# 步骤5: XML合并
merger = XMLMerger()
result = merger.process(context)
print(f"输出: {result.metadata['output_path']}")
```

## 📝 开发指南

### 新模块开发模板

每个处理模块都应该继承 `BaseProcessor` 并实现 `process()` 方法：

```python
from modules.base import BaseProcessor, ProcessingContext
from modules.data_types import ProcessingResult, get_layer_level

class MyProcessor(BaseProcessor):
    """我的处理模块"""
    
    # 我负责处理的元素类型
    MY_TYPES = {"my_type1", "my_type2"}
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        self._log("开始处理")
        
        for elem in context.elements:
            # 1. 判断是否是我负责的类型
            if elem.element_type not in self.MY_TYPES:
                continue
            
            # 2. 如果已经有XML，跳过
            if elem.has_xml():
                continue
            
            # 3. 处理元素（你的核心逻辑）
            # ...
            
            # 4. 生成XML字符串
            style = "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;"
            elem.xml_fragment = f'''<mxCell id="{elem.id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{elem.bbox.x1}" y="{elem.bbox.y1}" width="{elem.bbox.width}" height="{elem.bbox.height}" as="geometry"/>
</mxCell>'''
            
            # 5. 设置层级
            elem.layer_level = get_layer_level(elem.element_type)
            elem.processing_notes.append(f"{self.name}处理完成")
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height
        )
```

### 层级系统

DrawIO的Z轴规则：先写入的在底层，后写入的在顶层

```
层级定义（从底到顶）：
┌─────────────────────────────────────────────────────────┐
│  LayerLevel.OTHER (5)       - 其他（默认）               │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.TEXT (4)        - 文字                      │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.ARROW (3)       - 箭头/连接线               │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.IMAGE (2)       - 图片类（icon, picture）   │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.BASIC_SHAPE (1) - 基本图形                  │
├─────────────────────────────────────────────────────────┤
│  LayerLevel.BACKGROUND (0)  - 背景/容器（最底层）       │
└─────────────────────────────────────────────────────────┘
```

使用 `get_layer_level()` 函数获取默认层级：

```python
from modules.data_types import get_layer_level, LayerLevel

level = get_layer_level("rectangle")  # → 1 (BASIC_SHAPE)
level = get_layer_level("icon")       # → 2 (IMAGE)
level = get_layer_level("arrow")      # → 3 (ARROW)
level = get_layer_level("text")       # → 4 (TEXT)
```

## 📞 文字处理小组接口

文字处理是另一个小组负责，以下是接口说明：

### 输入

```python
context.image_path      # 原始图片路径
context.canvas_width    # 画布宽度
context.canvas_height   # 画布高度
```

### 输出要求

```python
from modules import XMLFragment, LayerLevel, BoundingBox

# OCR识别后，为每个文字区域创建XMLFragment
fragments = []
for text_region in ocr_results:
    xml = f'''<mxCell id="0" parent="1" vertex="1" value="{text_region.text}" 
            style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize=12;fontColor=#000000;">
      <mxGeometry x="{text_region.x}" y="{text_region.y}" width="{text_region.width}" height="{text_region.height}" as="geometry"/>
    </mxCell>'''
    
    fragments.append(XMLFragment(
        element_id=len(fragments),
        xml_content=xml,
        layer_level=LayerLevel.TEXT.value,  # ⭐ 必须设为TEXT (4)
        bbox=BoundingBox(x1, y1, x2, y2),
        element_type="text"
    ))
```
### 注意事项

1. **layer_level 必须设为 TEXT (4)**：确保文字在最上层
2. **转义特殊字符**：`<` → `&lt;`, `>` → `&gt;`, `&` → `&amp;`
3. **ID可以随意**：XMLMerger会重新分配ID

## ⚠️ 注意事项

1. **子模块必须设置 `xml_fragment`**：这是XMLMerger合并的依据
2. **子模块必须设置 `layer_level`**：决定元素的Z轴位置
3. **修改自己的模块时不要改动其他模块**
4. **保持接口稳定**：修改输入输出格式前先通知其他人

