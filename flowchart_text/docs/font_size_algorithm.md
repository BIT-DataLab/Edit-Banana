# 字体大小还原算法详解

## 一、整体流程图

```
原图 → Azure OCR → 获取文字多边形 → 计算短边高度 → Cap-Height算法 → draw.io字号(pt)
```

## 二、步骤详解

### 步骤 1：OCR 识别获取多边形

Azure OCR 返回每个文字块的**四边形多边形**坐标：

```
多边形: [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
        p0(左上) → p1(右上) → p2(右下) → p3(左下)
```

**示例**（"Embedding Head" 竖排文字）：
```
多边形: [(1467, 682), (1467, 402), (1508, 402), (1507, 682)]
```

**代码位置**：`src/ocr_azure.py` → `analyze_image()` 方法

---

### 步骤 2：计算多边形短边高度

在 `src/ocr_azure.py` 的 `_estimate_font_size` 方法中：

```python
def _estimate_font_size(self, polygon):
    p0, p1, p2, p3 = polygon[:4]
    
    # 计算顶边长度 (p0 → p1)
    edge1_len = sqrt((p1[0]-p0[0])² + (p1[1]-p0[1])²)
    
    # 计算左边长度 (p0 → p3)  
    edge2_len = sqrt((p3[0]-p0[0])² + (p3[1]-p0[1])²)
    
    # 取短边作为字符高度
    font_height = min(edge1_len, edge2_len)
    
    return font_height
```

**为什么取短边？**
- 对于**横排文字**：短边 = 文字的垂直高度
- 对于**竖排文字**：短边 = 单个字符的宽度（约等于高度）

**示例计算**：
```
p0=(1467, 682), p1=(1467, 402), p3=(1507, 682)

edge1 = sqrt((1467-1467)² + (402-682)²) = sqrt(0 + 78400) = 280
edge2 = sqrt((1507-1467)² + (682-682)²) = sqrt(1600 + 0) = 40

font_height = min(280, 40) = 40 px
```

---

### 步骤 3：Cap-Height 算法计算字号

在 `src/font_calculator.py` 的 `calculate_font_size` 方法中。

#### 3.1 理论基础

**Cap-Height（大写字母高度）** 是排版学中的标准概念：

```
┌─────────────────┐ ← Ascender line (上升线)
│     A   H   T   │ ← Cap height (大写高度) ≈ 0.7 × 字号
├─────────────────┤ ← x-height (小写高度)
│     a   x   o   │
├─────────────────┤ ← Baseline (基线)
│     g   y   p   │ ← Descender (下降部分)
└─────────────────┘
```

**核心关系**：
```
Cap-Height ≈ 0.7 × Font-Size
```

#### 3.2 计算公式

```python
# 配置参数（定义在 config.py）
CAP_HEIGHT_RATIO = 0.7    # 排版学标准：大写字母高度 ≈ 70% 字号
RENDER_RATIO = 1.8        # draw.io特性：N pt 文字渲染为 N×1.8 像素

def calculate_font_size(self, text, polygon_height_px, ...):
    
    # 判断是否包含大写字母或数字
    has_anchors = contains_anchor_chars(text)  # 匹配 [A-Z0-9]
    
    if has_anchors:
        # OCR高度 ≈ cap-height
        # font_size(px) = cap_height / 0.7
        font_size_px = polygon_height_px / CAP_HEIGHT_RATIO
    else:
        # OCR高度 ≈ x-height (小写字母高度)
        # font_size(px) = x_height / 0.5
        font_size_px = polygon_height_px / 0.5
    
    # 像素转pt (考虑draw.io渲染特性)
    font_size_pt = font_size_px / RENDER_RATIO
    
    return font_size_pt
```

#### 3.3 公式推导

对于包含大写字母的文字：

```
font_size_pt = polygon_height_px / CAP_HEIGHT_RATIO / RENDER_RATIO
             = polygon_height_px / 0.7 / 1.8
             = polygon_height_px / 1.26
             ≈ polygon_height_px × 0.79
```

**示例计算**：
```
OCR返回高度 = 40 px

font_size_px = 40 / 0.7 = 57.14 px
font_size_pt = 57.14 / 1.8 = 31.7 pt

最终字号 ≈ 32 pt
```

---

## 三、两个关键参数的含义

### CAP_HEIGHT_RATIO = 0.7

| 属性 | 说明 |
|------|------|
| **定义位置** | `config.py` |
| **来源** | 排版学标准 |
| **含义** | 大写字母高度约占完整字号的 70% |
| **特点** | 与图像无关，这是字体设计的通用规范 |

**为什么需要除以 0.7？**
- OCR 返回的是**可见文字的高度**（约等于 cap-height）
- 但字号包含了上下的空白空间
- 所以：`完整字号 = 可见高度 / 0.7`

---

### RENDER_RATIO = 1.8

| 属性 | 说明 |
|------|------|
| **定义位置** | `config.py` |
| **来源** | draw.io 渲染特性（经验值） |
| **含义** | N pt 的文字在 draw.io 中渲染为约 N×1.8 像素高 |
| **特点** | 与图像无关，这是 draw.io 的固有特性 |

**为什么需要除以 1.8？**
- 我们希望 draw.io 渲染出的文字高度 = OCR 检测到的高度
- 如果 draw.io 渲染 N pt → N×1.8 px
- 那么要渲染出 H px 的文字，需要设置字号为 H/1.8 pt

---

## 四、算法鲁棒性分析

这个算法**不依赖于**：
- ❌ 图像尺寸（2816×1536 或其他任何尺寸）
- ❌ 图像 DPI
- ❌ 假设的纸张大小

这个算法**只依赖于**：
- ✅ OCR 返回的多边形高度（直接测量值）
- ✅ Cap-Height 比例 0.7（排版学常数）
- ✅ draw.io 渲染比例 1.8（软件特性常数）

**换图测试**：无论输入什么尺寸的图像，只要 OCR 能正确检测文字多边形，算法都能输出合理的字号。

---

## 五、完整计算示例

以 "Embedding Head"（竖排文字）为例：

```
1. OCR返回多边形
   [(1467, 682), (1467, 402), (1508, 402), (1507, 682)]

2. 计算短边高度
   顶边: sqrt(0² + 280²) = 280
   左边: sqrt(40² + 0²) = 40
   短边 = min(280, 40) = 40 px

3. 检查锚点字符
   "Embedding Head" 包含 E, H 等大写字母 ✓

4. Cap-Height 算法
   font_size_px = 40 / 0.7 = 57.14
   font_size_pt = 57.14 / 1.8 = 31.7

5. 最终结果
   字号 ≈ 32 pt
```

---

## 六、代码文件索引

| 步骤 | 文件 | 方法/配置 |
|------|------|------|
| 配置参数 | `config.py` | `CAP_HEIGHT_RATIO`, `RENDER_RATIO` |
| OCR识别 | `src/ocr_azure.py` | `analyze_image()` |
| 计算短边高度 | `src/ocr_azure.py` | `_estimate_font_size()` |
| Cap-Height算法 | `src/font_calculator.py` | `calculate_font_size()` |

---

## 七、调参指南

如果字号效果不理想，可以调整 `config.py` 中的参数：

| 现象 | 调整方案 |
|------|---------|
| 字号整体偏大 | 增大 `RENDER_RATIO`（如 2.0） |
| 字号整体偏小 | 减小 `RENDER_RATIO`（如 1.5） |
| 大写字母正常，小写偏大 | 减小 x-height 比例（代码中的 0.5） |

---

## 八、简化公式速查

```
字号(pt) = OCR高度(px) ÷ CAP_HEIGHT_RATIO ÷ RENDER_RATIO
        = OCR高度(px) ÷ 0.7 ÷ 1.8
        = OCR高度(px) ÷ 1.26
        ≈ OCR高度(px) × 0.79
```

---

*文档更新日期：2026-01-09*

