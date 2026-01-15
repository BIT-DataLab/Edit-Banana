文件 3: 任务文档 `task.md`
该文档是将设计转化为工程实现的具体执行步骤，你可以将其分配给开发人员。

工程实现任务清单 (Task)

第一阶段： API 集成 (预计 2 天)

• [ ] 编写 Python 脚本调用 Azure AI `analyze_document` API，开启 `ocr.highResolution` 和 `styleFont` 功能。  
• [ ] 编写 Python 脚本调用 Mistral OCR API 获取 Markdown/LaTeX 结果。  
• [ ] 实现简单的内容对齐算法：通过 IoU（交并比）匹配两个模型返回的相同位置文本。  

第二阶段： 核心算法开发 (预计 3 天)

• [ ] 任务 2.1: 开发 `Polygon-to-Baseline` 转换函数。  
• [ ] 任务 2.2: 开发 `Cap-Height` 提取器。逻辑:  
  • 遍历 JSON 中的词级别 (words) 识别结果。  
  • 正则匹配含有大写字母/数字的词。  
  • 记录这些词的检测框高度，取均值。  

• [ ] 任务 2.3: 实现坐标归一化工具类，支持自适应原图宽高比。  

第三阶段： XML 合成器开发 (预计 2 天)

• [ ] 创建基础 mxGraph XML 模板（包含 `mxfile`, `diagram`, `mxGraphModel`, `root` 节点）。  
• [ ] 实现 `JSON-to-mxCell` 逻辑，重点映射 `style` 字符串中的 `fontSize` 和 `geometry` 坐标。  
• [ ] 确保每个文字块都有唯一的 `id` (0, 1 已被 draw.io 占用，从 2 开始)。  

第四阶段： 测试与验收 (预计 1 天)

• [ ] 使用“牛脸识别”原图进行端到端测试。  
• [ ] 验证：导入 draw.io 后，文字是否刚好落在原图位置？字号是否均匀？  
• [ ] 极限测试: 针对插图中 `CowID_1_Img_1` 等微小标注进行位置准确度校准。
