import os
import json
import base64
import yaml
from pathlib import Path
from openai import OpenAI
import httpx

# -------------------------- 全局配置与常量 --------------------------
# SAM3初始提示词（用于过滤重复，避免返回无效提示词）
SAM3_INITIAL_PROMPTS = [
    "icon", "picture", "rectangle", "section_panel",
    "text_bubble", "title_bar", "arrow", "rounded rectangle"
]

# -------------------------- 配置加载函数 --------------------------
def load_multimodal_config():
    """加载multimodal配置（带容错处理）"""
    try:
        CONFIG_PATH = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "config.yaml"
        )
        if not Path(CONFIG_PATH).exists():
            raise FileNotFoundError(f"配置文件不存在：{CONFIG_PATH}")
        
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f)
        
        # 校验multimodal节点是否存在
        if "multimodal" not in CONFIG:
            raise KeyError("配置文件中缺少'multimodal'节点")
        
        return CONFIG["multimodal"]
    
    except Exception as e:
        print(f"配置加载失败：{str(e)}")
        return None

# 提前加载配置（全局单例，避免重复读取文件）
MULTIMODAL_CONFIG = load_multimodal_config()

# -------------------------- 工具函数 --------------------------
def image_to_base64(image_path: str) -> tuple[str, str]:
    """
    优化版：将图片转为base64编码，并返回图片格式（适配png/jpg/jpeg）
    :param image_path: 图片路径
    :return: (img_base64_str, img_format_str)
    """
    img_path = Path(image_path)
    
    # 校验图片是否存在
    if not img_path.exists():
        raise FileNotFoundError(f"图片文件不存在：{image_path}")
    
    # 获取并处理图片格式（兼容jpg/jpeg，统一转为小写）
    img_format = img_path.suffix.lstrip(".").lower()
    img_format = "jpeg" if img_format == "jpg" else img_format
    
    # 读取并编码为base64
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    return img_base64, img_format

# -------------------------- 核心函数：获取补充提示词 --------------------------
def get_supplement_prompts(mask_vis_path: str, existing_prompts: list = None) -> list:
    """
    优化版：统一使用OpenAI SDK调用大模型（支持Remote API和Local Ollama）
    :param mask_vis_path: 掩码可视化图路径
    :param existing_prompts: 已识别的提示词列表（告诉模型这些不需要了）
    :return: 补充提示词列表（如["diamond", "ellipse"]）
    """
    # 前置校验：配置加载失败直接返回空列表
    if not MULTIMODAL_CONFIG:
        print("错误：多模态配置加载失败，无法调用API")
        return []
    
    # 前置校验：图片路径有效性
    if not mask_vis_path or not Path(mask_vis_path).exists():
        print(f"错误：掩码可视化图路径无效或文件不存在：{mask_vis_path}")
        return []
    
    # 构造已有元素描述
    existing_str = ""
    if existing_prompts:
        unique_existing = list(set(existing_prompts))
        existing_str = f"\n   (已知已识别的元素：{', '.join(unique_existing)}，请忽略这些类别)"

    # 1. 准备图片数据 (无论Local还是Remote都尽量走OpenAI Vision格式)
    try:
        img_base64, img_format = image_to_base64(mask_vis_path)
    except Exception as e:
        print(f"图片处理失败: {e}")
        return []

    # 2. 构建提示词
    prompt = f"""
請严格按照以下要求分析这张掩码可视化图：
1.  图中是一个流程图/架构图的元素分割掩码，彩色区域是已识别的元素，白色区域是未识别的非空白区域。
2.  请找出白色未识别区域对应的标准流程图形状名称。
3.  仅提供能让SAM3模型识别的英文提示词，优先使用以下标准DrawIO名称：
    - diamond (菱形/判断框)
    - cylinder (圆柱/数据库)
    - cloud (云)
    - actor (小人/角色)
    - ellipse (椭圆/圆形)
    - hexagon (六边形)
    - triangle (三角形)
    - parallelogram (平行四边形)
4.  也可以返回其他通用简单的英文名词（如 keyboard, monitor, server 等）。{existing_str}
5.  若所有非空白区域都已被识别，直接返回空JSON数组。
6.  输出要求：仅返回纯JSON数组字符串，无任何Markdown标记。
7.  示例输出：["diamond", "cloud", "cylinder"]
    """.strip()

    try:
        # 3. 确定配置 (Local vs Remote)
        mode = MULTIMODAL_CONFIG.get("mode", "api")
        
        if mode == "local":
            print(f"Using LOCAL Ollama: {MULTIMODAL_CONFIG.get('local_model')}")
            api_key = MULTIMODAL_CONFIG.get("local_api_key", "ollama")
            base_url = MULTIMODAL_CONFIG.get("local_base_url", "http://localhost:11434/v1")
            model_name = MULTIMODAL_CONFIG.get("local_model")
        else:
            print(f"Using REMOTE API: {MULTIMODAL_CONFIG.get('model')}")
            api_key = MULTIMODAL_CONFIG['api_key']
            base_url = MULTIMODAL_CONFIG['base_url']
            model_name = MULTIMODAL_CONFIG["model"]

        # 4. 初始化客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=1,
            http_client=httpx.Client(verify=False)
        )
        
        # 5. 调用API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_format};base64,{img_base64}",
                            }
                        }
                    ]
                }
            ],
            max_tokens=MULTIMODAL_CONFIG["max_tokens"],
            temperature=0.1  # 降低随机性
        )
        
        # check response
        if not response.choices:
            print("错误：API返回无有效choices内容")
            return []
            
        content = response.choices[0].message.content.strip()

        # ----------------- 通用处理逻辑 -----------------
        print(f"多模态模型返回内容：{content}")
        
        # 尝试解析JSON
        try:
            # 清理Markdown代码块标记
            cleaned_content = content.replace("```json", "").replace("```", "").strip()
            # 找到第一个[和最后一个]
            start_idx = cleaned_content.find('[')
            end_idx = cleaned_content.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = cleaned_content[start_idx:end_idx+1]
                prompts = json.loads(json_str)
                if isinstance(prompts, list):
                    # 过滤空字符串
                    return [p for p in prompts if isinstance(p, str) and p.strip()]
            
            print(f"警告：无法解析JSON数组，原始内容：{content}")
            return []
            
        except json.JSONDecodeError:
            print(f"JSON解析异常，原始内容：{content}")
            return []

    except Exception as e:
        print(f"调用多模态大模型失败：{str(e)}")
        return []

# -------------------------- 独立测试入口 --------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="调用Qwen3多模态API获取SAM3补充提示词")
    parser.add_argument("--mask", "-m", required=True, help="掩码可视化图路径（png/jpg/jpeg）")
    args = parser.parse_args()
    
    final_prompts = get_supplement_prompts(args.mask)
    print(f"\n测试完成 | 最终返回补充提示词列表：{final_prompts}")
