"""
图像验证和修复工具

此脚本用于检查数据集中的图像文件，识别损坏或不符合要求的图像，
并提供修复或移除这些图像的选项。
"""

import os
import sys
import cv2
import imghdr
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从数据处理模块导入配置
from data_processing.config import DATASET_DIR, DATASET_SUFFIX, SUPPORTED_FORMATS
from data_processing.core.dataset_processor import discover_categories


def validate_image(image_path, min_size=(32, 32), max_size=(10000, 10000)):
    """
    验证图像文件是否有效且符合要求
    
    参数:
        image_path: 图像文件路径
        min_size: 最小允许的图像尺寸 (宽, 高)
        max_size: 最大允许的图像尺寸 (宽, 高)
        
    返回:
        (是否有效, 问题描述) 元组
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        return False, "文件不存在"
    
    # 检查文件大小是否为0
    if os.path.getsize(image_path) == 0:
        return False, "文件大小为0"
    
    # 检查文件类型
    img_type = imghdr.what(image_path)
    if img_type is None:
        return False, "不是有效的图像文件"
    
    # 检查文件扩展名是否与实际类型匹配
    _, ext = os.path.splitext(image_path)
    ext = ext.lower().lstrip('.')
    if ext not in ['jpg', 'jpeg', 'png', 'bmp', 'gif'] or (ext == 'jpg' and img_type != 'jpeg'):
        return False, f"文件扩展名({ext})与实际类型({img_type})不匹配"
    
    # 尝试使用PIL打开图像
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # 检查图像尺寸
            if width < min_size[0] or height < min_size[1]:
                return False, f"图像尺寸过小: {width}x{height}"
            
            if width > max_size[0] or height > max_size[1]:
                return False, f"图像尺寸过大: {width}x{height}"
            
            # 尝试加载图像数据
            img.load()
    except Exception as e:
        return False, f"无法打开图像: {str(e)}"
    
    # 尝试使用OpenCV读取图像
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "OpenCV无法读取图像"
        
        # 检查图像通道数
        if len(img.shape) < 3:
            return False, "图像不是彩色图像"
    except Exception as e:
        return False, f"OpenCV读取错误: {str(e)}"
    
    return True, "图像有效"


def process_category(category, action='report', broken_dir=None, fix_extensions=False, verbose=False):
    """
    处理特定类别的所有图像
    
    参数:
        category: 类别名称
        action: 对问题图像的处理方式 ('report', 'move', 'delete', 'fix')
        broken_dir: 存放问题图像的目录
        fix_extensions: 是否修复文件扩展名
        verbose: 是否显示详细信息
        
    返回:
        (总图像数, 有效图像数, 问题图像数) 元组
    """
    # 构建类别目录路径
    category_dir = os.path.join(DATASET_DIR, f"{category}{DATASET_SUFFIX}")
    
    if not os.path.exists(category_dir):
        print(f"Error: Directory for category '{category}' not found at {category_dir}")
        return 0, 0, 0
    
    # 如果需要，创建存放问题图像的目录
    if action == 'move' and broken_dir:
        os.makedirs(broken_dir, exist_ok=True)
    
    # 获取目录中的所有图像文件
    image_files = []
    for filename in os.listdir(category_dir):
        if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in {category_dir}")
        return 0, 0, 0
    
    print(f"Found {len(image_files)} images in category '{category}'")
    
    # 初始化计数器
    total_count = len(image_files)
    valid_count = 0
    problem_count = 0
    
    # 处理每个文件
    for filename in tqdm(image_files, desc=f"Validating {category}"):
        image_path = os.path.join(category_dir, filename)
        is_valid, issue = validate_image(image_path)
        
        if is_valid:
            valid_count += 1
        else:
            problem_count += 1
            
            if verbose or action != 'report':
                print(f"  Issue with {filename}: {issue}")
            
            if action == 'move' and broken_dir:
                # 移动到问题图像目录
                dest_path = os.path.join(broken_dir, f"{category}_{filename}")
                shutil.move(image_path, dest_path)
                print(f"  Moved to {dest_path}")
            
            elif action == 'delete':
                # 删除问题图像
                os.remove(image_path)
                print(f"  Deleted {filename}")
            
            elif action == 'fix' and fix_extensions and 'extension' in issue.lower():
                # 尝试修复文件扩展名
                img_type = imghdr.what(image_path)
                if img_type in ['jpeg', 'png', 'bmp', 'gif']:
                    correct_ext = '.jpg' if img_type == 'jpeg' else f".{img_type}"
                    new_filename = os.path.splitext(filename)[0] + correct_ext
                    new_path = os.path.join(category_dir, new_filename)
                    
                    # 检查新文件名是否已存在
                    if os.path.exists(new_path):
                        print(f"  Cannot fix {filename}: {new_filename} already exists")
                    else:
                        os.rename(image_path, new_path)
                        print(f"  Fixed extension: {filename} -> {new_filename}")
    
    return total_count, valid_count, problem_count


def batch_validate_images(selected_categories=None, action='report', broken_dir='./datasets/broken_images', fix_extensions=False, verbose=False):
    """
    批量验证多个类别的图像
    
    参数:
        selected_categories: 要处理的类别列表，如果为None则处理所有类别
        action: 对问题图像的处理方式 ('report', 'move', 'delete', 'fix')
        broken_dir: 存放问题图像的目录
        fix_extensions: 是否修复文件扩展名
        verbose: 是否显示详细信息
    """
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
    
    # 初始化计数器
    total_images = 0
    total_valid = 0
    total_problems = 0
    
    # 处理每个类别
    for category in selected_categories:
        count, valid, problems = process_category(
            category, 
            action=action, 
            broken_dir=broken_dir,
            fix_extensions=fix_extensions,
            verbose=verbose
        )
        total_images += count
        total_valid += valid
        total_problems += problems
    
    # 打印摘要
    print("\nValidation Summary:")
    print(f"  Categories processed: {len(selected_categories)}")
    print(f"  Total images: {total_images}")
    print(f"  Valid images: {total_valid}")
    print(f"  Problem images: {total_problems}")
    
    if action != 'report':
        action_str = {
            'move': 'moved to broken directory',
            'delete': 'deleted',
            'fix': 'attempted to fix'
        }.get(action, 'processed')
        print(f"  Problem images {action_str}: {total_problems}")


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
    
    print("\nSelect categories to validate (comma-separated numbers, or 'all' for all categories):")
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


def main():
    """
    图像验证工具的主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and fix images in the dataset")
    parser.add_argument("--categories", nargs="+", help="Categories to process (default: all)")
    parser.add_argument("--action", choices=['report', 'move', 'delete', 'fix'], default='report',
                        help="Action to take for problem images (default: report)")
    parser.add_argument("--broken-dir", default="./datasets/broken_images",
                        help="Directory to move broken images to (default: ./datasets/broken_images)")
    parser.add_argument("--fix-extensions", action="store_true",
                        help="Attempt to fix incorrect file extensions")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed information about each problem image")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode for selecting categories")
    
    args = parser.parse_args()
    
    print("Image Validation Tool")
    print("====================")
    
    # 获取要处理的类别
    selected_categories = None
    if args.interactive:
        selected_categories = select_categories_interactive()
    elif args.categories:
        selected_categories = args.categories
    
    # 执行验证
    batch_validate_images(
        selected_categories=selected_categories,
        action=args.action,
        broken_dir=args.broken_dir,
        fix_extensions=args.fix_extensions,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main() 