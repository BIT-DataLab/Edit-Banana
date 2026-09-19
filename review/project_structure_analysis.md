# Edit-Banana 项目整体代码结构分析

> 分析基准：仓库 `main` 分支当前代码（2026-09-19 检查）。
> 本报告只分析代码与仓库结构，不修改业务代码。

## 1. 项目主要用途

代码可以确认：Edit-Banana 的核心用途是把一张静态图片中的流程图/架构图/技术图等视觉内容，重构成可编辑的 DrawIO XML。

核心输入输出关系：

```text
输入图片
  ↓
文本 OCR
  ↓
SAM3 图像分割/元素检测
  ↓
图标/图片处理 + 基本图形处理
  ↓
为元素生成 DrawIO mxCell XML
  ↓
XMLMerger 按图层/面积排序并重新组织
  ↓
可编辑 DrawIO XML
```

依据：`main.py` 顶部注释直接定义 pipeline；`server_pa.py` 的 `/convert` 最终调用同一个 `Pipeline.process_image()`；`README_CN.md` 也明确说明目标是图片到可编辑 DrawIO。

因此，项目不是单纯的 OCR 或单纯的 SAM3 demo，而是一个“图片 → 可编辑 DrawIO 图形”的端到端重构流水线。

## 2. 主要目录和模块职责

### `main.py`
**定位：主业务编排层 / CLI 主入口**

核心类：`Pipeline`

核心函数：
- `load_config()`
- `Pipeline.process_image()`
- `Pipeline._generate_xml_fragments()`
- `main()`

职责：加载配置、创建 `ProcessingContext`、调用 OCR、SAM3、图标/图片处理、基本图形处理、XML fragment 生成，并可选执行质量评估与 refinement，最后调用 `XMLMerger` 输出 DrawIO XML。

### `modules/`
**定位：核心业务模块**

#### `modules/base.py`
基础抽象层。

核心类：`ProcessingContext`、`BaseProcessor`、`ModelWrapper`。

作用：统一 processor 接口，并在模块之间共享图像路径、元素列表、画布尺寸和中间结果。

#### `modules/data_types.py`
核心数据模型。

核心类：`BoundingBox`、`ElementInfo`、`XMLFragment`、`ProcessingResult`、`ProcessingConfig`、`LayerLevel`。

其中最重要的是 `ElementInfo`：它是整个 pipeline 中各模块之间传递的核心业务对象。

```text
SAM3
  ↓
ElementInfo
  ↓
IconProcessor / ShapeProcessor
  ↓
ElementInfo 被不断补充
  ↓
xml_fragment + layer_level
  ↓
XMLMerger
```

#### `modules/sam3_info_extractor.py`
**核心视觉理解/元素发现模块。**

核心类：`Sam3InfoExtractor`、`SAM3Model`、`PromptGroup`、`PromptGroupConfig`、`ConfigLoader`。

核心函数：`process()`、`extract_by_group()`、`extract_with_custom_prompts()`、`_deduplicate_cross_groups()`、`_filter_contained_elements()`。

业务职责：根据 prompt group 调用 SAM3，提取 background / shape / image / arrow，做阈值过滤、去重并转换成 `ElementInfo`。

当前默认处理顺序由代码明确给出：`BACKGROUND → BASIC_SHAPE → IMAGE → ARROW`。

#### `modules/icon_picture_processor.py`
**图像/图标重构模块。**

核心类：`RMBGModel`、`IconPictureProcessor`。

核心函数：`process()`、`_process_element()`、`_generate_xml()`。

逻辑：对 icon / logo / arrow 等元素裁剪，部分类型使用 RMBG 去背景，之后编码为 Base64 并生成 DrawIO image 类型 `mxCell`。

当前代码明确把箭头纳入 icon/picture crop 路径，而不是主流程中的原生 DrawIO connector。

#### `modules/basic_shape_processor.py`
**基本图形重构模块。**

核心类：`BasicShapeProcessor`。

核心函数：`process()`、`_process_element()`、`_generate_xml()`、`_run_cv_detection()`、`_create_element_from_cv()`。

职责：处理 rectangle / ellipse / diamond / triangle / hexagon 等；使用 SAM3 mask 或 bbox 提取颜色；推断 stroke width / 几何参数；生成 DrawIO shape XML；使用 OpenCV 补充检测遗漏矩形。

#### `modules/text/`
**独立的文本重构子管线。**

核心类：`TextRestorer`。

其处理步骤为：

```text
OCR
 ↓
Formula refinement
 ↓
coordinate transform
 ↓
font size
 ↓
font family
 ↓
style
 ↓
DrawIO text XML
```

`modules/text/ocr/`：`LocalOCR`、`PaddleOCRAdapter`、`Pix2TextOCR` 等 OCR/公式引擎。

`modules/text/coord_processor.py`：`CoordProcessor`，负责 polygon → DrawIO geometry。

`modules/text/processors/`：字号、字体、样式和公式处理。

`modules/text/xml_generator.py`：`MxGraphXMLGenerator`，把文本 block 转成 DrawIO XML。

#### `modules/xml_merger.py`
**最终输出核心模块。**

核心类：`XMLMerger`。

核心函数：`process()`、`_collect_fragments()`、`_sort_fragments()`、`_build_xml_structure()`、`_parse_and_update_cell()`、`merge_with_text_xml()`。

职责：收集各模块 XML fragment、按 `layer_level` 排序、同层级按面积排序、重新分配 DrawIO cell ID，并构造完整 XML。

代码定义的层级顺序：

```text
0 BACKGROUND
1 BASIC_SHAPE
2 IMAGE
3 ARROW
4 TEXT
5 OTHER
```

#### `modules/metric_evaluator.py`
**可选质量评估模块。**

核心类：`MetricEvaluator`。

核心函数：`process()`、`_detect_bad_regions()`、`_detect_complex_image_regions()`、`_detect_fine_channel()`、`_detect_coarse_channel()`。

作用：计算内容覆盖情况、检测漏检区域、输出 score，为 refinement 提供 bad regions。

#### `modules/refinement_processor.py`
**Fallback 补救模块。**

核心类：`RefinementProcessor`。

当前代码采用保守策略：找到 bad region → 从原图裁剪 → 转 Base64 picture → 作为新的 image 元素加入 `context.elements`。

### `prompts/`
**模型提示词配置。**

- `arrow.py`：arrow / line / connector
- `background.py`：panel / container / filled region / background
- `image.py`：icon / picture / logo / chart / diagram
- `shape.py`：rectangle / rounded rectangle / diamond / ellipse / circle / triangle / hexagon

这些 prompt 被 `modules/sam3_info_extractor.py` 导入。

### `config/`
**运行配置。**

主要文件：`config/config.yaml.example`。

明确包含 SAM3 checkpoint/BPE、prompt group threshold/min_area/priority、OCR engine、multimodal、RMBG、paths 等。

仓库实际树中没有 `config/config.yaml`，只有 example；README 要求本地复制 example 后使用。

### `sam3_service/`
**独立的 SAM3 HTTP 服务层。**

主要文件：`server.py`、`client.py`、`rmbg_server.py`、`rmbg_client.py`、`run_all_service.py`。

核心类：`Sam3Runtime`、`Sam3ServiceClient`、`Sam3ServicePool`。

代码显示服务端会常驻加载 SAM3，并提供 `/predict`；客户端支持健康检查和预测，多 endpoint 使用轮询。

重要事实：当前 `main.py` 的主 Pipeline 直接使用 `Sam3InfoExtractor → SAM3Model`，没有看到 `Sam3ServiceClient` 被主 Pipeline 直接调用。因此 `sam3_service/` 属于可选服务化基础设施，而不是当前 CLI 主路径的必经模块。线上是否实际使用：**不确定**。

### `server_pa.py`
**FastAPI Web API 入口。**

接口：`GET /health`、`GET /`、`POST /convert`。

`convert()` 最终创建 `Pipeline(config)` 并调用 `process_image()`，因此 CLI 和 Web API 共用同一核心业务 Pipeline。

### `flowchart_text/`
**文本模块独立入口。**

`flowchart_text/main.py` 直接创建 `TextRestorer`，可以独立执行 OCR/text → DrawIO XML。

### `scripts/`
部署和辅助工具：`setup_sam3.sh`、`setup_rmbg.py`、`merge_xml.py`。

### `static/`
Demo 图片、GIF、Logo 等资源，不属于核心业务逻辑。

## 3. 程序入口、配置、依赖、测试

### 主程序入口
`main.py` → `main()` → `Pipeline.process_image()`。

### Web API 入口
`server_pa.py` → `main()` → uvicorn；实际转换接口是 `convert()`。

### 独立文本入口
`flowchart_text/main.py` → `main()`。

### 独立 SAM3 服务入口
`sam3_service/server.py` 在 `__main__` 中创建 `Sam3Runtime` 并启动 uvicorn。

### 配置文件
代码运行时读取 `config/config.yaml`；仓库当前实际存在的是 `config/config.yaml.example`。

### 依赖管理
主依赖：`requirements.txt`。

SAM3 服务额外有：`sam3_service/requirements.txt`。

### 测试目录
**代码可以确认：当前仓库没有独立的 `tests/` 目录。**

同时未发现标准 `test_*.py` / `*_test.py` 自动化测试文件。

README 所称“本地运行测试”主要是手动运行 CLI/API，不等于自动化测试体系。

## 4. 核心业务代码所在位置

第一核心层：
- `main.py`
- `modules/sam3_info_extractor.py`
- `modules/data_types.py`
- `modules/xml_merger.py`

第二核心层：
- `modules/text/restorer.py`
- `modules/icon_picture_processor.py`
- `modules/basic_shape_processor.py`

第三核心层：
- `modules/metric_evaluator.py`
- `modules/refinement_processor.py`

## 5. 代码分类

| 类型 | 文件/目录 | 职责 |
|---|---|---|
| 核心业务 | `main.py` | Pipeline 总编排 |
| 核心业务 | `modules/sam3_info_extractor.py` | SAM3 元素识别 |
| 核心业务 | `modules/basic_shape_processor.py` | 基本图形重构 |
| 核心业务 | `modules/icon_picture_processor.py` | 图片/图标重构 |
| 核心业务 | `modules/text/` | 文本/公式重构 |
| 核心业务 | `modules/xml_merger.py` | 最终 DrawIO XML 合成 |
| 核心业务 | `modules/metric_evaluator.py` | 质量评估 |
| 核心业务 | `modules/refinement_processor.py` | 漏检补救 |
| 基础设施 | `modules/base.py` | Processor 基础框架 |
| 基础设施 | `modules/data_types.py` | 核心跨模块数据结构 |
| 基础设施 | `sam3_service/` | SAM3 服务化推理 |
| 基础设施 | `server_pa.py` | Web API |
| 工具 | `modules/utils/` | 颜色/图像/XML 等工具 |
| 工具 | `scripts/` | 部署/辅助脚本 |
| 配置 | `config/config.yaml.example` | 运行参数 |
| 配置 | `prompts/` | SAM3 prompts |
| 测试 | 无独立测试目录 | 当前仓库未发现标准自动化测试目录 |

## 6. 简化项目结构树

```text
Edit-Banana/
├── main.py                         # 主 CLI + Pipeline
├── server_pa.py                    # FastAPI Web API
├── requirements.txt                # 主依赖
│
├── config/
│   └── config.yaml.example         # 配置模板
│
├── prompts/
│   ├── arrow.py                    # 箭头 prompts
│   ├── background.py               # 背景 prompts
│   ├── image.py                    # 图片 prompts
│   └── shape.py                    # 图形 prompts
│
├── modules/
│   ├── base.py                     # Processor 基类 / Context
│   ├── data_types.py               # ElementInfo 等核心数据结构
│   ├── sam3_info_extractor.py      # SAM3 元素识别
│   ├── icon_picture_processor.py   # 图标/图片处理
│   ├── basic_shape_processor.py    # 基本图形处理
│   ├── metric_evaluator.py         # 质量评估
│   ├── refinement_processor.py     # 漏检补救
│   ├── xml_merger.py               # XML 最终合并
│   │
│   ├── text/
│   │   ├── restorer.py             # 文本主流程
│   │   ├── coord_processor.py      # 坐标转换
│   │   ├── xml_generator.py        # 文本 XML
│   │   ├── ocr/                    # OCR 引擎
│   │   └── processors/             # 字体/样式/公式处理
│   │
│   └── utils/                      # 通用工具
│
├── sam3_service/
│   ├── server.py                   # SAM3 HTTP 服务
│   └── client.py                   # SAM3 服务客户端
│
├── flowchart_text/
│   └── main.py                     # 文本子管线独立入口
│
├── scripts/                        # 安装/辅助脚本
└── static/                         # Demo/静态资源
```

## 7. 推荐优先阅读的 10 个文件

1. **`main.py`**：理解整个业务流程，重点看 `Pipeline.process_image()`、`_generate_xml_fragments()`、`main()`。
2. **`modules/data_types.py`**：理解 `ElementInfo`、`ProcessingContext`、`ProcessingResult`、`LayerLevel`，也就是模块之间“传什么数据”。
3. **`modules/sam3_info_extractor.py`**：理解“图片 → 元素列表”的核心过程，重点看 `process()`、`SAM3Model.predict()`、`_convert_to_elements()`、`_deduplicate_cross_groups()`。
4. **`modules/xml_merger.py`**：理解“多个模块结果 → 最终 DrawIO 文件”的过程，重点看 `process()`、`_collect_fragments()`、`_sort_fragments()`、`_build_xml_structure()`。
5. **`modules/basic_shape_processor.py`**：理解 AI 识别的形状如何变成 DrawIO 原生 shape，以及 CV fallback。
6. **`modules/icon_picture_processor.py`**：理解图片/图标/箭头为什么最终变成 Base64 image。
7. **`modules/text/restorer.py`**：理解文本重构子 Pipeline，以及 OCR→公式→坐标→字体→样式→XML。
8. **`modules/text/xml_generator.py`**：理解 OCR 数据如何落成 DrawIO text cell。
9. **`modules/metric_evaluator.py`**：理解项目如何判断漏检区域，并生成 refinement 输入。
10. **`modules/refinement_processor.py`**：理解质量评估后的 fallback 修复方式。

## 8. 最重要的主调用链

### 普通模式

```mermaid
flowchart TD
    A[main.py: main()] --> B[Pipeline.process_image()]
    B --> C[TextRestorer.process()]
    B --> D[Sam3InfoExtractor.process()]
    D --> E[SAM3Model.predict()]
    D --> F[ElementInfo list]
    F --> G[IconPictureProcessor.process()]
    F --> H[BasicShapeProcessor.process()]
    G --> I[xml_fragment]
    H --> I
    I --> J[Pipeline._generate_xml_fragments()]
    J --> K[XMLMerger.process()]
    K --> L[final DrawIO XML]
```

### 开启 `--refine`

```mermaid
flowchart TD
    A[Pipeline.process_image()] --> B[基础识别与 XML fragment]
    B --> C[MetricEvaluator.process()]
    C --> D{score < 90 && bad_regions}
    D -- yes --> E[RefinementProcessor.process()]
    E --> F[新增 picture ElementInfo]
    F --> G[XMLMerger.process()]
    D -- no --> G
    G --> H[final DrawIO XML]
```

## 9. 当前项目架构的关键认识

```text
Sam3InfoExtractor
      ↓
  ElementInfo
      ↓
┌───────────────┬────────────────┐
│               │                │
TextRestorer    ShapeProcessor   IconProcessor
│               │                │
Text XML        Shape XML        Image XML
└───────────────┴────────────────┘
                ↓
           XMLMerger
                ↓
          DrawIO XML
```

最重要的三个概念是：
- `ElementInfo`：跨模块共享的核心业务数据。
- `Pipeline`：整个转换流程的控制中心。
- `XMLMerger`：所有重构结果的最终汇聚点。

## 10. 明确的不确定项

1. `sam3_service/` 是否已经被线上生产系统使用：**不确定**。代码提供了完整服务端/客户端实现，但当前主 `main.py` 没有直接接入它。
2. README 中所称完整“多模态 VLM”能力，在当前仓库主 Pipeline 中实际执行到什么程度：**不确定**。当前明确可确认的是主 Pipeline 创建 `TextRestorer(formula_engine="none")`，因此主 CLI 默认文本路径并不执行 Pix2Text 公式识别。
3. 当前主流程中是否存在其他未在仓库树中展示的箭头专用实现：**不确定**。当前仓库实际文件列表中没有独立 `arrow_processor.py`，且主 Pipeline 将 arrow 纳入 `IconPictureProcessor`/image fragment 路径。

## 总结

这个项目最适合按照：

**Pipeline → 数据结构 → SAM3 → 文本/图形/图片三类重构器 → XMLMerger → 可选 Metric/Refinement**

的顺序学习，而不是按目录逐个文件阅读。