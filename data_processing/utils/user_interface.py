"""
用户交互模块
"""
import multiprocessing
import os
from data_processing.config import OUTPUT_DIR
from data_processing.core.dataset_processor import discover_categories


def select_categories(categories):
    """
    让用户选择要处理的类别
    
    Args:
        categories: 可用类别列表
    
    Returns:
        选择的类别列表
    """
    print("\nAvailable categories:")
    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")
    
    print("\nSelect categories to process (comma-separated numbers, or 'all' for all categories):")
    selection = input("> ").strip().lower()
    
    if selection == 'all' or selection == '':
        print("Processing all categories (will use simplified filename)")
        return categories
    
    try:
        # 解析用户输入的数字
        indices = [int(idx.strip()) for idx in selection.split(',') if idx.strip()]
        selected = [categories[idx-1] for idx in indices if 1 <= idx <= len(categories)]
        
        if not selected:
            print("No valid categories selected, processing all categories (will use simplified filename)")
            return categories
        
        # 检查是否选择了所有类别
        if set(selected) == set(categories):
            print("All categories selected (will use simplified filename)")
        else:
            print(f"Selected categories: {', '.join(selected)}")
        
        return selected
    except (ValueError, IndexError):
        print("Invalid input, processing all categories (will use simplified filename)")
        return categories


def select_processing_mode():
    """
    让用户选择处理模式
    
    Returns:
        处理模式: "parallel" 或 "sequential"
    """
    print("\nSelect processing mode:")
    print("1. Parallel processing (recommended, faster)")
    print("2. Sequential processing (if parallel processing has issues)")
    
    selection = input("> ").strip()
    
    if selection == '2':
        return "sequential"
    else:
        return "parallel"


def select_cpu_cores():
    """
    让用户选择使用多少个CPU核心进行并行处理
    
    Returns:
        用户选择的CPU核心数，如果输入无效则返回None（表示使用所有核心）
    """
    # 获取系统可用的CPU核心数
    total_cores = multiprocessing.cpu_count()
    
    print(f"\nYour system has {total_cores} CPU cores available.")
    print(f"Please enter the number of cores to use (1-{total_cores}), or press Enter to use all cores:")
    
    selection = input("> ").strip()
    
    if not selection:
        print(f"Will use all {total_cores} cores")
        return None  # 返回None表示使用所有核心
    
    try:
        cores = int(selection)
        if 1 <= cores <= total_cores:
            print(f"Will use {cores} cores")
            return cores
        else:
            print(f"Input out of range, will use all {total_cores} cores")
            return None
    except ValueError:
        print(f"Invalid input, will use all {total_cores} cores")
        return None


def generate_output_filenames(selected_categories):
    """
    根据选择的类别生成输出文件名
    
    Args:
        selected_categories: 选择的类别列表，如果为None则表示所有类别
        
    Returns:
        训练集、验证集和测试集的输出文件路径
    """
    if selected_categories:
        # 如果选择了特定类别，在文件名中反映这一点
        category_suffix = '_'.join([cat.lower() for cat in selected_categories])
        train_file = f"{OUTPUT_DIR}/{category_suffix}_train.mindrecord"
        val_file = f"{OUTPUT_DIR}/{category_suffix}_val.mindrecord"
        test_file = f"{OUTPUT_DIR}/{category_suffix}_test.mindrecord"
    else:
        # 处理所有类别，使用"ewaste"作为前缀
        train_file = f"{OUTPUT_DIR}/ewaste_train.mindrecord"
        val_file = f"{OUTPUT_DIR}/ewaste_val.mindrecord"
        test_file = f"{OUTPUT_DIR}/ewaste_test.mindrecord"
    
    return train_file, val_file, test_file


def get_user_selections(categories):
    """
    获取用户选择的类别和处理模式
    
    Args:
        categories: 可用类别列表
    
    Returns:
        选择的类别列表、处理模式和CPU核心数
    """
    # 选择要处理的类别
    selected_categories = select_categories(categories)
    
    # 选择处理模式
    processing_mode = select_processing_mode()
    
    # 如果选择并行处理，询问使用多少个CPU核心
    cpu_cores = None
    if processing_mode == "parallel":
        cpu_cores = select_cpu_cores()
    
    return selected_categories, processing_mode, cpu_cores 