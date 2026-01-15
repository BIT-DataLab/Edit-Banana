文件 2: 设计文档 `design.md`
该文档详细说明了 Pipeline 的逻辑架构、选型理由及核心算法逻辑。

OCR 矢量还原系统详细设计 (Design)

1. 系统架构图

Image Input -> OCR 识别层 -> 坐标与字号处理器 -> XML 合成引擎 -> Draw.io File

2. 详细设计模块

2.1 OCR 识别层 (多模型互验证)

• 主模型: Azure Document Intelligence (`ocrHighResolution`)。  
  • 职责: 获取每个文字块的 8 点多边形坐标 (Polygon) 及初始样式猜测。  

• 校对模型: Mistral OCR。  
  • 职责: 对 Azure 返回的 text 字段进行核对。若检测到 `$` 符号或数学结构，以 Mistral 的 LaTeX 输出为准。  

2.2 坐标处理器 (基线锚定法)

• 归一化: 将原图宽/高映射为 1000 × 1000 的相对坐标系，消除分辨率干扰。  
• 对齐基准:  
  • 提取 Polygon 的底部两点 (x3, y3) 和 (x4, y4)。  
  • 计算基线 Y_baseline = (y3 + y4)/2。  
  • 在 XML 生成时，将文本容器的垂直位置锚定在基线。  

2.3 字号处理器 (Cap-Height 算法)

• 锚点识别: 扫描文本内容，筛选出所有大写字母 (A-Z) 和数字 (0-9)。  
• 高度计算: 计算这些锚点字符在原图中的垂直像素高度均值 H^{cap}_{px}。  
• 转换公式:  
  • 估算磅值 Estimated_pt = (H^{cap}_{px} ÷ 0.7) × (Canvas_Scale)。  
  • 这里的 0.7 是基于大多数主流字体大写高度约占总字号 70% 的排版学验证。  

2.4 XML 合成引擎 (Python 路径)

• 技术选型: 使用 Python `xml.etree.ElementTree` 库。  
• Cell 样式定义: 为每个 `<mxCell>` 注入以下硬性样式:  
  • `whiteSpace=wrap;`: 强制换行控制。  
  • `autosize=1;`: 核心，允许 draw.io 根据字号微调框高。  
  • `resizable=0;`: 锁定宽度，保持物理对齐。  
  • `html=1;`: 允许富文本和公式渲染。  

3. 数据中间格式 (Schema)json

{"text_blocks":,... [x8,y8]]}}}
