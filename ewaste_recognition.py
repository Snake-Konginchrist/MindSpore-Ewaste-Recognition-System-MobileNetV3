"""
电子垃圾识别系统主程序

此程序提供了一个用户友好的界面，用于训练和使用电子垃圾识别模型。
用户可以选择重新训练模型或使用现有模型进行识别。
"""

import os
import sys
import time
import multiprocessing
from PIL import Image
import mindspore as ms
from mindspore import context

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入项目模块
from train.train import train_model
from eval.evaluate import Predictor
from config.config import Config
from data_processing.utils.user_interface import select_processing_mode, select_cpu_cores
from data_processing.core.dataset_processor import discover_categories
from utils.dataset import find_mindrecord_files


def print_header(title):
    """打印带有分隔线的标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_model_exists():
    """
    检查模型文件是否存在
    
    Returns:
        bool: 模型文件是否存在
    """
    return os.path.exists(Config.best_model_path)


def select_action():
    """
    让用户选择要执行的操作
    
    Returns:
        str: 选择的操作 ("train", "detect", "exit")
    """
    print_header("E-Waste Recognition System")
    print("\nPlease select an operation:")
    print("1. Train New Model")
    print("2. Recognize E-Waste")
    print("0. Exit Program")
    
    while True:
        choice = input("\n> ").strip()
        if choice == "1":
            return "train"
        elif choice == "2":
            return "detect"
        elif choice == "0":
            return "exit"
        else:
            print("Invalid selection, please try again")


def select_dataset_type():
    """
    让用户选择数据集类型
    
    Returns:
        tuple: (use_mindrecord, train_file, val_file)
            use_mindrecord: 是否使用MindRecord数据集
            train_file: 训练集文件路径
            val_file: 验证集文件路径
    """
    print("\nPlease select dataset type:")
    print("1. Use original image folders (default)")
    print("2. Use preprocessed MindRecord dataset")
    print("0. Return to main menu")
    
    choice = input("\n> ").strip()
    if choice == "0":
        return None, None, None
    
    # 使用原始图像文件夹
    if choice != "2":
        return False, None, None
    
    # 使用MindRecord数据集
    # 查找可用的MindRecord文件
    train_files, val_files, test_files = find_mindrecord_files("./datasets")
    
    if not train_files or not val_files:
        print("\nError: No available MindRecord dataset files found")
        print("Please generate MindRecord datasets first, or choose to use original image folders")
        return False, None, None
    
    # 显示可用的训练集文件
    print("\nAvailable training set files:")
    for i, file_path in enumerate(train_files, 1):
        file_name = os.path.basename(file_path)
        print(f"{i}. {file_name}")
    
    # 选择训练集文件
    print("\nPlease select a training set file (enter number):")
    train_choice = input("> ").strip()
    try:
        train_idx = int(train_choice) - 1
        if train_idx < 0 or train_idx >= len(train_files):
            print("Invalid selection, will use the first training set file")
            train_idx = 0
    except ValueError:
        print("Invalid input, will use the first training set file")
        train_idx = 0
    
    train_file = train_files[train_idx]
    
    # 查找对应的验证集文件
    train_basename = os.path.basename(train_file)
    val_basename = train_basename.replace("_train", "_val")
    val_file = None
    
    for file_path in val_files:
        if os.path.basename(file_path) == val_basename:
            val_file = file_path
            break
    
    # 如果找不到对应的验证集文件，使用第一个验证集文件
    if val_file is None:
        print("\nNo corresponding validation set file found, will use the first validation set file")
        val_file = val_files[0]
    
    print(f"\nSelected dataset:")
    print(f"  Training set: {os.path.basename(train_file)}")
    print(f"  Validation set: {os.path.basename(val_file)}")
    
    return True, train_file, val_file


def train_with_options():
    """
    提供训练选项并执行训练
    """
    print_header("Model Training")
    
    # 选择数据集类型
    use_mindrecord, train_file, val_file = select_dataset_type()
    if use_mindrecord is None:  # 用户选择返回主菜单
        return
    
    # 选择处理模式
    print("\nPlease select processing mode:")
    print("1. Parallel processing (recommended, faster)")
    print("2. Sequential processing (if parallel processing has issues)")
    print("0. Return to main menu")
    
    choice = input("\n> ").strip()
    if choice == "0":
        return
    
    processing_mode = "parallel" if choice != "2" else "sequential"
    
    # 如果选择并行处理，询问CPU核心数
    cpu_cores = None
    if processing_mode == "parallel":
        # 获取可用的CPU核心数
        total_cores = multiprocessing.cpu_count()
        
        print(f"\nYour system has {total_cores} CPU cores available.")
        print(f"Please enter the number of cores to use (1-{total_cores}), or press Enter to use all cores:")
        print("0. Return to main menu")
        
        selection = input("\n> ").strip()
        if selection == "0":
            return
        
        if not selection:
            print(f"Will use all {total_cores} cores")
            cpu_cores = total_cores
        else:
            try:
                cores = int(selection)
                if 1 <= cores <= total_cores:
                    print(f"Will use {cores} cores")
                    cpu_cores = cores
                else:
                    print(f"Input out of range, will use all {total_cores} cores")
                    cpu_cores = total_cores
            except ValueError:
                print(f"Invalid input, will use all {total_cores} cores")
                cpu_cores = total_cores
    
    # 设置环境变量
    if processing_mode == "parallel" and cpu_cores is not None:
        os.environ['OMP_NUM_THREADS'] = str(cpu_cores)
        print(f"Setting parallel processing threads to: {cpu_cores}")
    
    # 确认训练设置
    print("\nTraining Settings:")
    print(f"  Dataset type: {'MindRecord' if use_mindrecord else 'Original image folders'}")
    if use_mindrecord:
        print(f"  Training set file: {os.path.basename(train_file)}")
        print(f"  Validation set file: {os.path.basename(val_file)}")
    else:
        print(f"  Training set directory: {Config.train_data_dir}")
        print(f"  Validation set directory: {Config.val_data_dir}")
    print(f"  Processing mode: {'Parallel' if processing_mode == 'parallel' else 'Sequential'}")
    if processing_mode == "parallel":
        print(f"  CPU cores: {cpu_cores}")
    print(f"  Training epochs: {Config.num_epochs}")
    print(f"  Batch size: {Config.batch_size}")
    print(f"  Learning rate: {Config.learning_rate}")
    
    print("\nConfirm to start training? (y/n)")
    confirm = input("> ").strip().lower()
    if confirm != 'y' and confirm != 'yes':
        print("Training cancelled")
        return
    
    # 开始训练
    print("\nStarting model training...")
    start_time = time.time()
    
    try:
        # 如果是顺序处理，将核心数设为1
        if processing_mode == "sequential":
            cpu_cores = 1
            
        # 调用训练函数，传递CPU核心数和数据集信息
        train_model(
            num_workers=cpu_cores,
            use_mindrecord=use_mindrecord,
            train_mindrecord=train_file,
            val_mindrecord=val_file
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\nTraining completed! Time used: {training_time:.2f} seconds")
        print(f"Model saved to: {Config.best_model_path}")
    except Exception as e:
        print(f"\nError during training: {str(e)}")


def detect_with_options():
    """
    提供识别选项并执行识别
    """
    print_header("E-Waste Recognition")
    
    # 检查模型是否存在
    if not check_model_exists():
        print(f"\nError: Model file not found at {Config.best_model_path}")
        print("Please train a model first or ensure the model file exists")
        
        print("\nTrain a model now? (y/n)")
        choice = input("> ").strip().lower()
        if choice == 'y' or choice == 'yes':
            train_with_options()
            # 训练后再次检查模型
            if not check_model_exists():
                print("\nModel training failed or not saved, cannot perform recognition")
                return
        else:
            return
    
    # 加载模型
    print("\nLoading model...")
    try:
        predictor = Predictor(Config.best_model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    # 选择识别方法
    print("\nPlease select recognition method:")
    print("1. Recognize a single image")
    print("2. Recognize all images in a folder")
    print("0. Return to main menu")
    
    choice = input("\n> ").strip()
    if choice == "0":
        return
    
    # 识别单张图片
    if choice == "1":
        print("\nPlease enter image path (or drag and drop image to this window):")
        image_path = input("> ").strip().strip('"\'')
        
        if not os.path.exists(image_path):
            print(f"Error: File does not exist at {image_path}")
            return
        
        try:
            # 显示图片信息
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"\nImage info: {os.path.basename(image_path)}, {width}x{height}, {img.format}")
            
            # 执行预测
            print("\nRecognizing...")
            result = predictor.predict(image_path)
            
            # 显示结果
            print("\nRecognition Result:")
            print(f"Class: {result['class_name']}")
            print(f"Confidence: {result['probability']:.2%}")
        except Exception as e:
            print(f"Error during recognition: {str(e)}")
    
    # 识别文件夹中的所有图片
    elif choice == "2":
        print("\nPlease enter image folder path:")
        folder_path = input("> ").strip().strip('"\'')
        
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"Error: Folder does not exist at {folder_path}")
            return
        
        # 获取文件夹中的所有图片
        image_files = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(folder_path, filename))
        
        if not image_files:
            print("No image files found in the folder")
            return
        
        print(f"\nFound {len(image_files)} images, starting recognition...")
        
        # 处理每张图片
        results = []
        for image_path in image_files:
            try:
                result = predictor.predict(image_path)
                results.append({
                    'filename': os.path.basename(image_path),
                    'class_name': result['class_name'],
                    'probability': result['probability']
                })
                print(f"Recognized: {os.path.basename(image_path)} -> {result['class_name']} ({result['probability']:.2%})")
            except Exception as e:
                print(f"Error recognizing {os.path.basename(image_path)}: {str(e)}")
        
        # 显示统计信息
        if results:
            print("\nRecognition Statistics:")
            class_counts = {}
            for result in results:
                class_name = result['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                percentage = count / len(results) * 100
                print(f"{class_name}: {count} images ({percentage:.1f}%)")


def main():
    """
    主函数
    """
    while True:
        action = select_action()
        
        if action == "exit":
            print("\nThank you for using the E-Waste Recognition System. Goodbye!")
            break
        elif action == "train":
            train_with_options()
        elif action == "detect":
            detect_with_options()


if __name__ == "__main__":
    main() 