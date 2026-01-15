import os
import sys
import yaml
import traceback  # 新增：打印详细堆栈
from pathlib import Path
from sam3_extractor import main as sam3_main
from merge_xml import run_text_extraction, merge_xml

# 加载配置
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)
INPUT_DIR = CONFIG["paths"]["input_dir"]
FINAL_DIR = CONFIG["paths"]["final_dir"]
TEMP_DIR = CONFIG["paths"]["temp_dir"]

def process_single_image(image_path: str):
    """处理单张图片的完整流程"""
    print(f"\n========== 开始处理：{image_path} ==========")
    img_stem = Path(image_path).stem

    try:
        # 步骤1：SAM3迭代提取元素，生成初始XML
        print("步骤1：SAM3迭代提取元素...")
        sam3_xml_path = sam3_main(image_path)
        # 校验SAM3 XML是否存在且非空
        if not os.path.exists(sam3_xml_path) or os.path.getsize(sam3_xml_path) == 0:
            raise Exception(f"SAM3生成的XML为空或不存在：{sam3_xml_path}")

        # 步骤2：文字识别，生成文字XML
        print("步骤2：文字识别...")
        text_xml_path = run_text_extraction(image_path)
        # 校验文字XML
        if not os.path.exists(text_xml_path) or os.path.getsize(text_xml_path) == 0:
            raise Exception(f"文字识别生成的XML为空或不存在：{text_xml_path}")

        # 步骤3：合并XML
        print("步骤3：合并XML...")
        merged_xml_path = os.path.join(TEMP_DIR, f"{img_stem}_merged_temp.drawio.xml")
        merge_xml(sam3_xml_path, text_xml_path, image_path, merged_xml_path)

    except Exception as e:
        # 打印详细堆栈，定位具体错误行
        print(f"\n处理{image_path}失败：{e}")
        traceback.print_exc()  # 打印完整异常堆栈
        return None

def main():
    """遍历输入文件夹，处理所有图片"""
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"错误：输入目录不存在 → {INPUT_DIR}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 支持的图片格式
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # 遍历图片
    image_paths = []
    for file in os.listdir(INPUT_DIR):
        ext = Path(file).suffix.lower()
        if ext in SUPPORTED_FORMATS:
            image_paths.append(os.path.join(INPUT_DIR, file))

    if not image_paths:
        print(f"错误：输入目录{INPUT_DIR}中未找到支持的图片文件")
        sys.exit(1)

    # 处理每张图片
    for img_path in image_paths:
        process_single_image(img_path)

    print(f"\n所有图片处理完成！最终文件保存在：{FINAL_DIR}")

if __name__ == "__main__":
    main()