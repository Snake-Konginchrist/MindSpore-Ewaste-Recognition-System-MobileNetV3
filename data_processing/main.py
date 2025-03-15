"""
主程序：电子垃圾图像数据处理
"""
import os
import multiprocessing
from data_processing.utils.user_interface import select_categories, select_processing_mode, generate_output_filenames, get_user_selections
from data_processing.utils.timing import Timer
from data_processing.core.dataset_processor import (
    get_image_files, 
    split_dataset, 
    prepare_mindrecord_writers,
    process_dataset_parallel,
    process_dataset_sequential,
    save_mindrecord_data,
    discover_categories
)
from data_processing.config import OUTPUT_DIR, DATASET_DIR


def main():
    """
    主函数，处理电子垃圾图像数据集
    """
    print("Welcome to the Electronic Waste Image Dataset Processor")
    
    # 自动发现类别
    categories, _ = discover_categories()
    
    if not categories:
        print("No categories found in the dataset directory.")
        return
    
    # 获取用户选择的类别、处理模式和CPU核心数
    selected_categories, processing_mode, cpu_cores = get_user_selections(categories)
    
    # 检查是否选择了所有类别
    if selected_categories and set(selected_categories) == set(categories):
        print("All categories selected, using simplified filename.")
        selected_categories = None
    elif not selected_categories:
        print("No categories selected, using all available categories.")
        selected_categories = None
    
    # 获取图像文件列表
    image_files, label_dict = get_image_files(DATASET_DIR, selected_categories)
    
    if not image_files:
        print("No image files found. Please check your dataset directory.")
        return
    
    # 生成输出文件名
    train_file, val_file, test_file = generate_output_filenames(selected_categories)
    
    # 根据处理模式选择处理方法
    if processing_mode == "parallel":
        # 如果用户没有指定CPU核心数，使用系统的所有核心
        if cpu_cores is None:
            cpu_cores = multiprocessing.cpu_count()
            
        print(f"Using parallel processing mode with {cpu_cores} workers")
        
        # 并行处理数据集
        process_dataset_parallel(image_files, train_file, val_file, test_file, cpu_cores)
    else:
        # 顺序处理数据集
        print("Using sequential processing mode")
        process_dataset_sequential(image_files, train_file, val_file, test_file)
    
    print("Dataset processing completed!")
    print(f"Training set: {train_file}")
    print(f"Validation set: {val_file}")
    print(f"Test set: {test_file}")
    
    return


if __name__ == "__main__":
    main() 