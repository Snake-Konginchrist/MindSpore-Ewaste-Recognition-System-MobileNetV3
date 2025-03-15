"""
图像预处理工具集命令行界面

此脚本提供了一个统一的命令行界面，用于访问所有图像预处理工具。
"""

import os
import sys
import argparse

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入各个工具模块
from rename_images import batch_rename_images, select_categories_interactive as select_categories_rename
from image_validator import batch_validate_images, select_categories_interactive as select_categories_validate
from image_converter import batch_convert_images, select_categories_interactive as select_categories_convert
from image_analyzer import batch_analyze_images, select_categories_interactive as select_categories_analyze


def print_header(title):
    """打印带有分隔线的标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def rename_tool():
    """重命名工具的交互式界面"""
    print_header("Image Renaming Tool")
    print("This tool will rename all images in the dataset to a standardized format:")
    print("category_number.ext (e.g., camera_001.jpg, keyboard_023.png)")
    
    # 询问用户是否要继续
    print("\nDo you want to proceed with renaming? (y/n)")
    choice = input("> ").strip().lower()
    if choice != 'y' and choice != 'yes':
        print("Operation cancelled.")
        return
    
    # 询问要处理的类别
    print("\nDo you want to rename images in specific categories or all? (specific/all)")
    choice = input("> ").strip().lower()
    
    selected_categories = None
    if choice == 'specific':
        selected_categories = select_categories_rename()
    
    # 询问是否进行干运行
    print("\nDo you want to perform a dry run first? (y/n)")
    print("(A dry run will show what would be renamed without actually changing any files)")
    dry_run = input("> ").strip().lower() in ['y', 'yes']
    
    # 询问是否覆盖
    overwrite = False
    if not dry_run:
        print("\nDo you want to overwrite existing files if they have the new naming format? (y/n)")
        overwrite = input("> ").strip().lower() in ['y', 'yes']
    
    # 确认设置
    print("\nSettings:")
    print(f"  Categories: {'All' if selected_categories is None else ', '.join(selected_categories)}")
    print(f"  Dry run: {'Yes' if dry_run else 'No'}")
    print(f"  Overwrite: {'Yes' if overwrite else 'No'}")
    
    print("\nConfirm these settings? (y/n)")
    choice = input("> ").strip().lower()
    if choice != 'y' and choice != 'yes':
        print("Operation cancelled.")
        return
    
    # 执行重命名
    print("\nStarting renaming process...\n")
    batch_rename_images(selected_categories, dry_run, overwrite)
    
    # 最终消息
    if not dry_run:
        print("\nRenaming completed!")
    else:
        print("\nDry run completed. Run again without dry run to perform actual renaming.")


def validate_tool():
    """验证工具的交互式界面"""
    print_header("Image Validation Tool")
    print("This tool will check all images in the dataset for issues and can fix or remove problematic files.")
    
    # 询问用户是否要继续
    print("\nDo you want to proceed with validation? (y/n)")
    choice = input("> ").strip().lower()
    if choice != 'y' and choice != 'yes':
        print("Operation cancelled.")
        return
    
    # 询问要处理的类别
    print("\nDo you want to validate images in specific categories or all? (specific/all)")
    choice = input("> ").strip().lower()
    
    selected_categories = None
    if choice == 'specific':
        selected_categories = select_categories_validate()
    
    # 询问处理方式
    print("\nWhat action do you want to take for problem images?")
    print("1. Report only (no changes)")
    print("2. Move to a separate directory")
    print("3. Delete problem images")
    print("4. Try to fix issues (e.g., file extensions)")
    
    action_choice = input("> ").strip()
    actions = {
        '1': 'report',
        '2': 'move',
        '3': 'delete',
        '4': 'fix'
    }
    action = actions.get(action_choice, 'report')
    
    # 询问是否修复扩展名
    fix_extensions = False
    if action == 'fix':
        print("\nDo you want to attempt to fix incorrect file extensions? (y/n)")
        fix_extensions = input("> ").strip().lower() in ['y', 'yes']
    
    # 询问是否显示详细信息
    print("\nDo you want to see detailed information about each problem image? (y/n)")
    verbose = input("> ").strip().lower() in ['y', 'yes']
    
    # 如果需要，询问存放问题图像的目录
    broken_dir = './datasets/broken_images'
    if action == 'move':
        print(f"\nWhere do you want to move problem images? (default: {broken_dir})")
        user_dir = input("> ").strip()
        if user_dir:
            broken_dir = user_dir
    
    # 确认设置
    print("\nSettings:")
    print(f"  Categories: {'All' if selected_categories is None else ', '.join(selected_categories)}")
    print(f"  Action: {action}")
    if action == 'move':
        print(f"  Broken images directory: {broken_dir}")
    if action == 'fix':
        print(f"  Fix extensions: {'Yes' if fix_extensions else 'No'}")
    print(f"  Verbose output: {'Yes' if verbose else 'No'}")
    
    print("\nConfirm these settings? (y/n)")
    choice = input("> ").strip().lower()
    if choice != 'y' and choice != 'yes':
        print("Operation cancelled.")
        return
    
    # 执行验证
    print("\nStarting validation process...\n")
    batch_validate_images(
        selected_categories=selected_categories,
        action=action,
        broken_dir=broken_dir,
        fix_extensions=fix_extensions,
        verbose=verbose
    )


def convert_tool():
    """转换工具的交互式界面"""
    print_header("Image Conversion Tool")
    print("This tool will convert images to a different format and/or resize them.")
    
    # 询问用户是否要继续
    print("\nDo you want to proceed with conversion? (y/n)")
    choice = input("> ").strip().lower()
    if choice != 'y' and choice != 'yes':
        print("Operation cancelled.")
        return
    
    # 询问要处理的类别
    print("\nDo you want to convert images in specific categories or all? (specific/all)")
    choice = input("> ").strip().lower()
    
    selected_categories = None
    if choice == 'specific':
        selected_categories = select_categories_convert()
    
    # 询问目标格式
    print("\nSelect target format:")
    print("1. PNG (lossless, larger files)")
    print("2. JPG (lossy compression, smaller files)")
    print("3. BMP (uncompressed, very large files)")
    print("4. GIF (limited colors, good for simple graphics)")
    
    format_choice = input("> ").strip()
    formats = {
        '1': 'png',
        '2': 'jpg',
        '3': 'bmp',
        '4': 'gif'
    }
    target_format = formats.get(format_choice, 'png')
    
    # 如果选择JPG，询问质量
    quality = 90
    if target_format == 'jpg':
        print("\nSelect JPEG quality (1-100, higher is better quality but larger file size):")
        try:
            quality = int(input("> ").strip())
            quality = max(1, min(100, quality))
        except:
            print("Invalid input, using default quality (90).")
            quality = 90
    
    # 询问是否调整大小
    print("\nDo you want to resize the images? (y/n)")
    resize_choice = input("> ").strip().lower()
    resize = None
    if resize_choice in ['y', 'yes']:
        print("\nEnter new size in format WIDTHxHEIGHT (e.g., 224x224):")
        size_str = input("> ").strip()
        try:
            width, height = size_str.lower().split('x')
            resize = (int(width), int(height))
        except:
            print("Invalid size format. Images will not be resized.")
    
    # 询问输出位置
    print("\nWhere do you want to save the converted images?")
    print("1. Create new directories with suffix (e.g., camera_converted)")
    print("2. Convert in-place (replace original files)")
    
    output_choice = input("> ").strip()
    in_place = (output_choice == '2')
    
    output_suffix = '_converted'
    keep_originals = True
    
    if not in_place:
        print("\nEnter suffix for output directories (default: _converted):")
        user_suffix = input("> ").strip()
        if user_suffix:
            output_suffix = user_suffix
        
        print("\nDo you want to keep the original files? (y/n)")
        keep_originals = input("> ").strip().lower() in ['y', 'yes']
    else:
        output_suffix = None
        keep_originals = False
    
    # 确认设置
    print("\nSettings:")
    print(f"  Categories: {'All' if selected_categories is None else ', '.join(selected_categories)}")
    print(f"  Target format: {target_format.upper()}")
    if target_format == 'jpg':
        print(f"  JPEG quality: {quality}")
    if resize:
        print(f"  Resize to: {resize[0]}x{resize[1]}")
    if in_place:
        print("  Output: In-place (replace original files)")
    else:
        print(f"  Output: New directories with suffix '{output_suffix}'")
        print(f"  Keep originals: {'Yes' if keep_originals else 'No'}")
    
    print("\nConfirm these settings? (y/n)")
    choice = input("> ").strip().lower()
    if choice != 'y' and choice != 'yes':
        print("Operation cancelled.")
        return
    
    # 执行转换
    print("\nStarting conversion process...\n")
    batch_convert_images(
        selected_categories=selected_categories,
        target_format=target_format,
        quality=quality,
        resize=resize,
        output_suffix=output_suffix,
        keep_originals=keep_originals
    )


def analyze_tool():
    """分析工具的交互式界面"""
    print_header("Image Analysis Tool")
    print("This tool will analyze images and generate statistics and visualizations.")
    
    # 询问用户是否要继续
    print("\nDo you want to proceed with analysis? (y/n)")
    choice = input("> ").strip().lower()
    if choice != 'y' and choice != 'yes':
        print("Operation cancelled.")
        return
    
    # 询问要处理的类别
    print("\nDo you want to analyze images in specific categories or all? (specific/all)")
    choice = input("> ").strip().lower()
    
    selected_categories = None
    if choice == 'specific':
        selected_categories = select_categories_analyze()
    
    # 询问是否使用样本
    print("\nDo you want to analyze all images or just a sample? (all/sample)")
    sample_choice = input("> ").strip().lower()
    
    sample_size = None
    if sample_choice == 'sample':
        print("\nHow many images do you want to sample from each category?")
        try:
            sample_size = int(input("> ").strip())
        except:
            print("Invalid input, analyzing all images.")
    
    # 询问是否显示详细信息
    print("\nDo you want to see detailed information about each image? (y/n)")
    verbose = input("> ").strip().lower() in ['y', 'yes']
    
    # 询问输出目录
    output_dir = './analysis_results'
    print(f"\nWhere do you want to save the analysis results? (default: {output_dir})")
    user_dir = input("> ").strip()
    if user_dir:
        output_dir = user_dir
    
    # 确认设置
    print("\nSettings:")
    print(f"  Categories: {'All' if selected_categories is None else ', '.join(selected_categories)}")
    if sample_size:
        print(f"  Sample size: {sample_size} images per category")
    else:
        print("  Sample size: All images")
    print(f"  Verbose output: {'Yes' if verbose else 'No'}")
    print(f"  Output directory: {output_dir}")
    
    print("\nConfirm these settings? (y/n)")
    choice = input("> ").strip().lower()
    if choice != 'y' and choice != 'yes':
        print("Operation cancelled.")
        return
    
    # 执行分析
    print("\nStarting analysis process...\n")
    batch_analyze_images(
        selected_categories=selected_categories,
        sample_size=sample_size,
        verbose=verbose,
        output_dir=output_dir
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Image Preprocessing Tools")
    parser.add_argument("tool", nargs="?", choices=["rename", "validate", "convert", "analyze"],
                        help="Tool to run (if not specified, shows menu)")
    
    args = parser.parse_args()
    
    if args.tool:
        # 直接运行指定的工具
        tools = {
            "rename": rename_tool,
            "validate": validate_tool,
            "convert": convert_tool,
            "analyze": analyze_tool
        }
        tools[args.tool]()
    else:
        # 显示菜单
        while True:
            print_header("Electronic Waste Image Preprocessing Tools")
            print("1. Rename Images - Standardize image filenames")
            print("2. Validate Images - Check for and fix image issues")
            print("3. Convert Images - Change format or resize images")
            print("4. Analyze Images - Generate statistics and visualizations")
            print("0. Exit")
            
            choice = input("\nSelect a tool (0-4): ").strip()
            
            if choice == '0':
                print("Exiting...")
                break
            elif choice == '1':
                rename_tool()
            elif choice == '2':
                validate_tool()
            elif choice == '3':
                convert_tool()
            elif choice == '4':
                analyze_tool()
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main() 