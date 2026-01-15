"""
配置管理模块
从环境变量加载 API 密钥和其他配置
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Azure Document Intelligence 配置
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")

# Mistral AI 配置
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# ========== 字号计算参数 (Cap-Height 算法) ==========
# 大写字母高度占总字号的比例（排版学标准）
# 原理：大写字母（如 A, H, T）的高度约为完整字号的 70%
CAP_HEIGHT_RATIO = 0.7

# draw.io 渲染比例：N pt 的文字在 draw.io 中渲染为 N × RENDER_RATIO 像素
# 经过实测，在 draw.io 默认设置下，这个比例约为 1.8
# 这个值是 draw.io 的渲染特性，与图像尺寸无关
RENDER_RATIO = 1.8

# ========== 内容对齐参数 ==========
# 文本块匹配的交并比阈值
IOU_THRESHOLD = 0.5

# ========== 输出目录配置 ==========
# 输出文件夹路径（相对于项目根目录）
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_BATCH_DIR = OUTPUT_DIR / "batch"       # 批量处理输出
OUTPUT_TEST_DIR = OUTPUT_DIR / "test"         # 测试输出
OUTPUT_GOOD_DIR = OUTPUT_DIR / "good"         # 效果较好的输出


def validate_config():
    """验证必要的配置是否已设置"""
    errors = []
    
    if not AZURE_ENDPOINT:
        errors.append("AZURE_ENDPOINT 未设置")
    if not AZURE_API_KEY:
        errors.append("AZURE_API_KEY 未设置")
    if not MISTRAL_API_KEY:
        errors.append("MISTRAL_API_KEY 未设置")
    
    if errors:
        print("配置错误:")
        for error in errors:
            print(f"  - {error}")
        print("\n请复制 .env.example 为 .env 并填写 API 密钥")
        return False
    
    return True

