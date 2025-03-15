"""
图像格式转换工具

此脚本用于将数据集中的图像从一种格式转换为另一种格式，
例如将JPG转换为PNG，或将所有图像统一为同一格式。
"""

import os
import sys
import time
import shutil
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从数据处理模块导入配置
from data_processing.config import DATASET_DIR, DATASET_SUFFIX, SUPPORTED_FORMATS
from data_processing.core.dataset_processor import discover_categories


def convert_image(input_path, output_path, target_format='png', quality=90, resize=None):
    """
    转换单个图像的格式
    
    参数:
        input_path: 输入图像路径
        output_path: 输出图像路径
        target_format: 目标格式 ('jpg', 'png', 'bmp', 'gif')
        quality: JPEG压缩质量 (1-100)
        resize: 调整大小的元组 (宽, 高) 或 None
        
    返回:
        是否成功转换
    """
    try:
        # 打开原始图像
        with Image.open(input_path) as img:
            # 确保图像是RGB模式
            if img.mode != 'RGB' and target_format.lower() in ['jpg', 'jpeg']:
                img = img.convert('RGB')
            
            # 如果需要调整大小
            if resize:
                img = img.resize(resize, Image.LANCZOS)
            
            # 保存为目标格式
            if target_format.lower() in ['jpg', 'jpeg']:
                img.save(output_path, format='JPEG', quality=quality)
            elif target_format.lower() == 'png':
                img.save(output_path, format='PNG')
            elif target_format.lower() == 'bmp':
                img.save(output_path, format='BMP')
            elif target_format.lower() == 'gif':
                img.save(output_path, format='GIF')
            else:
                print(f"Unsupported format: {target_format}")
                return False
            
            return True
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False


def convert_category(category, target_format='png', quality=90, resize=None, 
                    source_formats=None, output_dir=None, keep_originals=True):
    """
    转换特定类别的所有图像
    
    参数:
        category: 类别名称
        target_format: 目标格式 ('jpg', 'png', 'bmp', 'gif')
        quality: JPEG压缩质量 (1-100)
        resize: 调整大小的元组 (宽, 高) 或 None
        source_formats: 要转换的源格式列表，如果为None则转换所有支持的格式
        output_dir: 输出目录，如果为None则使用原始目录
        keep_originals: 是否保留原始文件
        
    返回:
        (总图像数, 成功转换数, 失败数) 元组
    """
    # 构建类别目录路径
    category_dir = os.path.join(DATASET_DIR, f"{category}{DATASET_SUFFIX}")
    
    if not os.path.exists(category_dir):
        print(f"Error: Directory for category '{category}' not found at {category_dir}")
        return 0, 0, 0
    
    # 确定输出目录
    if output_dir:
        target_dir = output_dir
        os.makedirs(target_dir, exist_ok=True)
    else:
        target_dir = category_dir
    
    # 确定源格式
    if source_formats is None:
        source_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    else:
        source_formats = [f".{fmt.lower().lstrip('.')}" for fmt in source_formats]
    
    # 获取目录中的所有图像文件
    image_files = []
    for filename in os.listdir(category_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in source_formats:
            image_files.append(filename)
    
    if not image_files:
        print(f"No matching image files found in {category_dir}")
        return 0, 0, 0
    
    print(f"Found {len(image_files)} images to convert in category '{category}'")
    
    # 初始化计数器
    total_count = len(image_files)
    success_count = 0
    failure_count = 0
    
    # 处理每个文件
    for filename in tqdm(image_files, desc=f"Converting {category}"):
        input_path = os.path.join(category_dir, filename)
        
        # 创建新文件名
        base_name = os.path.splitext(filename)[0]
        new_filename = f"{base_name}.{target_format.lower()}"
        output_path = os.path.join(target_dir, new_filename)
        
        # 如果输入和输出路径相同，使用临时文件
        if input_path == output_path:
            temp_output_path = os.path.join(target_dir, f"temp_{new_filename}")
            success = convert_image(input_path, temp_output_path, target_format, quality, resize)
            
            if success:
                os.remove(input_path)
                os.rename(temp_output_path, output_path)
                success_count += 1
            else:
                failure_count += 1
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
        else:
            success = convert_image(input_path, output_path, target_format, quality, resize)
            
            if success:
                success_count += 1
                # 如果不保留原始文件，则删除
                if not keep_originals and output_dir is not None:
                    os.remove(input_path)
            else:
                failure_count += 1
    
    return total_count, success_count, failure_count


def batch_convert_images(selected_categories=None, target_format='png', quality=90, resize=None,
                        source_formats=None, output_suffix='_converted', keep_originals=True):
    """
    批量转换多个类别的图像
    
    参数:
        selected_categories: 要处理的类别列表，如果为None则处理所有类别
        target_format: 目标格式 ('jpg', 'png', 'bmp', 'gif')
        quality: JPEG压缩质量 (1-100)
        resize: 调整大小的元组 (宽, 高) 或 None
        source_formats: 要转换的源格式列表，如果为None则转换所有支持的格式
        output_suffix: 输出目录后缀，如果为None则使用原始目录
        keep_originals: 是否保留原始文件
    """
    # 记录开始时间
    start_time = time.time()
    
    # 发现可用类别
    categories, _ = discover_categories()
    
    if not categories:
        print("No categories found in the dataset directory.")
        return
    
    print(f"Available categories: {', '.join(categories)}")
    
    # 如果未指定类别，使用所有类别
    if selected_categories is None:
        selected_categories = categories
    else:
        # 验证选定的类别
        valid_categories = [cat for cat in selected_categories if cat in categories]
        if not valid_categories:
            print("None of the selected categories exist in the dataset.")
            return
        selected_categories = valid_categories
    
    print(f"Selected categories: {', '.join(selected_categories)}")
    print(f"Target format: {target_format}")
    
    # 初始化计数器
    total_images = 0
    total_success = 0
    total_failure = 0
    
    # 处理每个类别
    for category in selected_categories:
        # 确定输出目录
        if output_suffix:
            output_dir = os.path.join(DATASET_DIR, f"{category}{output_suffix}")
        else:
            output_dir = None
        
        count, success, failure = convert_category(
            category,
            target_format=target_format,
            quality=quality,
            resize=resize,
            source_formats=source_formats,
            output_dir=output_dir,
            keep_originals=keep_originals
        )
        
        total_images += count
        total_success += success
        total_failure += failure
    
    # 计算总时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 打印摘要
    print("\nConversion Summary:")
    print(f"  Categories processed: {len(selected_categories)}")
    print(f"  Total images: {total_images}")
    print(f"  Successfully converted: {total_success}")
    print(f"  Failed to convert: {total_failure}")
    print(f"  Elapsed time: {elapsed_time:.2f} seconds")


def select_categories_interactive():
    """
    交互式选择要处理的类别
    
    返回:
        选定的类别列表或None表示所有类别
    """
    # 发现可用类别
    categories, _ = discover_categories()
    
    if not categories:
        print("No categories found in the dataset directory.")
        return None
    
    print("\nAvailable categories:")
    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")
    
    print("\nSelect categories to convert (comma-separated numbers, or 'all' for all categories):")
    selection = input("> ").strip().lower()
    
    if selection == 'all' or selection == '':
        print("Processing all categories")
        return None  # None表示所有类别
    
    try:
        # 解析用户输入
        indices = [int(idx.strip()) for idx in selection.split(',') if idx.strip()]
        selected = [categories[idx-1] for idx in indices if 1 <= idx <= len(categories)]
        
        if not selected:
            print("No valid categories selected, processing all categories")
            return None
        
        print(f"Selected categories: {', '.join(selected)}")
        return selected
    except (ValueError, IndexError):
        print("Invalid input, processing all categories")
        return None


def parse_size(size_str):
    """
    解析尺寸字符串
    
    参数:
        size_str: 格式为"宽x高"的字符串，例如"224x224"
        
    返回:
        (宽, 高)元组或None
    """
    if not size_str:
        return None
    
    try:
        width, height = size_str.lower().split('x')
        return (int(width), int(height))
    except:
        print(f"Invalid size format: {size_str}. Expected format: WIDTHxHEIGHT (e.g., 224x224)")
        return None


def main():
    """
    图像转换工具的主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert images to a different format")
    parser.add_argument("--categories", nargs="+", help="Categories to process (default: all)")
    parser.add_argument("--format", choices=['jpg', 'png', 'bmp', 'gif'], default='png',
                        help="Target format (default: png)")
    parser.add_argument("--quality", type=int, default=90,
                        help="JPEG quality (1-100, default: 90)")
    parser.add_argument("--resize", help="Resize images to WIDTHxHEIGHT (e.g., 224x224)")
    parser.add_argument("--source-formats", nargs="+",
                        help="Source formats to convert (default: all)")
    parser.add_argument("--output-suffix", default="_converted",
                        help="Suffix for output directories (default: _converted)")
    parser.add_argument("--in-place", action="store_true",
                        help="Convert images in-place (overwrite originals)")
    parser.add_argument("--keep-originals", action="store_true",
                        help="Keep original files when using --output-suffix")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode for selecting categories")
    
    args = parser.parse_args()
    
    print("Image Conversion Tool")
    print("====================")
    
    # 解析尺寸
    resize = parse_size(args.resize)
    
    # 确定是否保留原始文件
    keep_originals = True
    if args.in_place:
        output_suffix = None
        keep_originals = False
    else:
        output_suffix = args.output_suffix
        keep_originals = args.keep_originals
    
    # 获取要处理的类别
    selected_categories = None
    if args.interactive:
        selected_categories = select_categories_interactive()
    elif args.categories:
        selected_categories = args.categories
    
    # 执行转换
    batch_convert_images(
        selected_categories=selected_categories,
        target_format=args.format,
        quality=args.quality,
        resize=resize,
        source_formats=args.source_formats,
        output_suffix=output_suffix,
        keep_originals=keep_originals
    )


if __name__ == "__main__":
    main() 