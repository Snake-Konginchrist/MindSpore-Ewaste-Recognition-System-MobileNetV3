"""
批量图像重命名工具

此脚本将数据集中的图像重命名为标准格式："类别_编号(3位数)"
例如：camera_001.jpg, keyboard_023.png 等。
"""

import os
import sys
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从数据处理模块导入配置
from data_processing.config import DATASET_DIR, DATASET_SUFFIX, SUPPORTED_FORMATS
from data_processing.core.dataset_processor import discover_categories


def rename_images_in_category(category, dry_run=False, overwrite=False):
    """
    重命名特定类别文件夹中的所有图像
    
    参数:
        category: 类别名称
        dry_run: 如果为True，只显示将要进行的操作而不实际重命名
        overwrite: 如果为True，覆盖已存在的具有新命名模式的文件
    
    返回:
        (成功计数, 跳过计数, 错误计数)的元组
    """
    # 构建类别目录路径
    category_dir = os.path.join(DATASET_DIR, f"{category}{DATASET_SUFFIX}")
    
    if not os.path.exists(category_dir):
        print(f"Error: Directory for category '{category}' not found at {category_dir}")
        return 0, 0, 0
    
    # 获取目录中的所有图像文件
    image_files = []
    for filename in os.listdir(category_dir):
        if filename.lower().endswith(SUPPORTED_FORMATS):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in {category_dir}")
        return 0, 0, 0
    
    print(f"Found {len(image_files)} images in category '{category}'")
    
    # 排序文件以确保一致的编号
    image_files.sort()
    
    # 初始化计数器
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    # 处理每个文件
    for i, filename in enumerate(tqdm(image_files, desc=f"Renaming {category}")):
        # 获取文件扩展名
        _, ext = os.path.splitext(filename)
        
        # 创建带有3位数编号的新文件名
        new_filename = f"{category}_{(i+1):03d}{ext.lower()}"
        
        # 完整路径
        old_path = os.path.join(category_dir, filename)
        new_path = os.path.join(category_dir, new_filename)
        
        # 检查新文件名是否已存在
        if os.path.exists(new_path) and not overwrite:
            print(f"  Skipping: {new_filename} already exists (use --overwrite to force)")
            skipped_count += 1
            continue
        
        # 执行重命名操作
        try:
            if not dry_run:
                # 先创建临时副本以避免大小写不敏感文件系统的问题
                if filename.lower() == new_filename.lower() and filename != new_filename:
                    temp_path = os.path.join(category_dir, f"temp_{new_filename}")
                    shutil.copy2(old_path, temp_path)
                    os.remove(old_path)
                    os.rename(temp_path, new_path)
                else:
                    os.rename(old_path, new_path)
            
            print(f"  {filename} -> {new_filename}" + (" (dry run)" if dry_run else ""))
            success_count += 1
        except Exception as e:
            print(f"  Error renaming {filename}: {str(e)}")
            error_count += 1
    
    return success_count, skipped_count, error_count


def batch_rename_images(selected_categories=None, dry_run=False, overwrite=False):
    """
    批量重命名多个类别的图像
    
    参数:
        selected_categories: 要处理的类别列表，如果为None则处理所有类别
        dry_run: 如果为True，只显示将要进行的操作而不实际重命名
        overwrite: 如果为True，覆盖已存在的具有新命名模式的文件
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
    total_success = 0
    total_skipped = 0
    total_errors = 0
    
    # 处理每个类别
    for category in selected_categories:
        success, skipped, errors = rename_images_in_category(category, dry_run, overwrite)
        total_success += success
        total_skipped += skipped
        total_errors += errors
    
    # 打印摘要
    print("\nRenaming Summary:")
    print(f"  Categories processed: {len(selected_categories)}")
    print(f"  Files successfully renamed: {total_success}")
    print(f"  Files skipped: {total_skipped}")
    print(f"  Errors encountered: {total_errors}")
    
    if dry_run:
        print("\nThis was a dry run. No files were actually renamed.")
        print("Run without --dry-run to perform the actual renaming.")


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
    
    print("\nSelect categories to rename (comma-separated numbers, or 'all' for all categories):")
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
    批量图像重命名工具的主函数
    """
    parser = argparse.ArgumentParser(description="Batch rename images to standardized format")
    parser.add_argument("--categories", nargs="+", help="Categories to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without renaming")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode for selecting categories")
    
    args = parser.parse_args()
    
    print("Batch Image Renaming Tool")
    print("=========================")
    
    # 获取要处理的类别
    selected_categories = None
    if args.interactive:
        selected_categories = select_categories_interactive()
    elif args.categories:
        selected_categories = args.categories
    
    # 执行重命名
    batch_rename_images(selected_categories, args.dry_run, args.overwrite)


if __name__ == "__main__":
    main() 