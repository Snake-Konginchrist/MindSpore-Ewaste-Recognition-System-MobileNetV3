"""
数据集处理模块
"""
import os
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from mindspore.mindrecord import FileWriter
import multiprocessing

from data_processing.config import TRAIN_RATIO, VAL_RATIO, SUPPORTED_FORMATS, DATASET_DIR, DATASET_SUFFIX, TESTSET_SUFFIX
from data_processing.core.image_processor import process_image, process_image_for_parallel


def discover_categories(dataset_dir=DATASET_DIR, suffix=DATASET_SUFFIX):
    """
    自动发现数据集中的类别
    
    Args:
        dataset_dir: 数据集目录
        suffix: 数据集文件夹后缀
        
    Returns:
        类别列表和类别映射字典
    """
    categories = []
    
    # 遍历数据集目录
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        
        # 检查是否是目录且以指定后缀结尾
        if os.path.isdir(item_path) and item.endswith(suffix):
            # 提取类别名称（前缀）
            category = item[:-len(suffix)] if suffix else item
            categories.append(category)
    
    # 排序以确保一致性
    categories.sort()
    
    # 创建类别映射字典
    label_dict = {category: idx for idx, category in enumerate(categories)}
    
    return categories, label_dict


def get_image_files(image_dir, selected_labels=None):
    """
    获取指定目录下的图像文件列表
    
    Args:
        image_dir: 图像目录
        selected_labels: 选择处理的标签列表，如果为None则处理所有标签
        
    Returns:
        图像文件列表和标签映射字典
    """
    # 确保使用绝对路径
    image_dir = os.path.abspath(image_dir)
    
    # 自动发现类别
    categories, label_dict = discover_categories()
    
    if not categories:
        print("No categories found in the dataset directory.")
        return [], {}
    
    print(f"Discovered categories: {', '.join(categories)}")
    
    # 如果没有指定标签，使用所有发现的标签
    if selected_labels is None:
        selected_labels = categories
    else:
        # 验证选择的标签是否存在
        valid_labels = [label for label in selected_labels if label in label_dict]
        if not valid_labels:
            print("None of the selected categories exist in the dataset. Using all available categories.")
            selected_labels = categories
        else:
            selected_labels = valid_labels
    
    # 获取所有图像文件
    image_files = []
    skipped_files = 0
    
    for label in selected_labels:
        label_dir = os.path.join(image_dir, f"{label}{DATASET_SUFFIX}")
        if not os.path.exists(label_dir):
            print(f"Warning: Directory for category '{label}' not found.")
            continue
        
        print(f"Processing directory: {label_dir}")
        
        try:
            # 获取该类别下的所有图像文件
            for filename in os.listdir(label_dir):
                try:
                    if filename.lower().endswith(SUPPORTED_FORMATS):
                        file_path = os.path.join(label_dir, filename)
                        
                        # 检查文件是否可访问
                        if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
                            print(f"Warning: Cannot access file: {file_path}")
                            skipped_files += 1
                            continue
                            
                        # 存储完整路径和标签信息
                        image_files.append({
                            'path': file_path,
                            'filename': filename,
                            'label': label,
                            'label_id': label_dict[label]
                        })
                except UnicodeEncodeError:
                    print(f"Warning: Unicode encoding error with filename in {label_dir}")
                    skipped_files += 1
                    continue
                except Exception as e:
                    print(f"Warning: Error processing file in {label_dir}: {str(e)}")
                    skipped_files += 1
                    continue
        except Exception as e:
            print(f"Error listing directory {label_dir}: {str(e)}")
            continue
    
    if skipped_files > 0:
        print(f"Warning: Skipped {skipped_files} files due to access or encoding issues.")
    
    print(f"Found {len(image_files)} images in {len(selected_labels)} categories.")
    return image_files, label_dict


def split_dataset(image_files):
    """
    将图像文件列表分割为训练集、验证集和测试集
    
    Args:
        image_files: 图像文件列表
        
    Returns:
        训练集、验证集和测试集的文件列表
    """
    # 随机打乱文件列表
    random.shuffle(image_files)

    # 计算训练集、验证集和测试集的大小
    n_train = int(TRAIN_RATIO * len(image_files))
    n_val = int(VAL_RATIO * len(image_files))
    n_test = len(image_files) - n_train - n_val

    # 分割数据集
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[-n_test:]
    
    return train_files, val_files, test_files


def prepare_mindrecord_writers(train_file, val_file, test_file):
    """
    准备MindRecord文件写入器
    
    Args:
        train_file: 训练集输出文件路径
        val_file: 验证集输出文件路径
        test_file: 测试集输出文件路径
        
    Returns:
        训练集、验证集和测试集的MindRecord文件写入器
    """
    # 创建MindRecord文件写入器
    train_writer = FileWriter(train_file, shard_num=1)
    val_writer = FileWriter(val_file, shard_num=1)
    test_writer = FileWriter(test_file, shard_num=1)

    # 定义数据架构
    schema = {
        "image": {"type": "bytes"},
        "label": {"type": "int32"}
    }
    
    # 添加数据架构
    train_writer.add_schema(schema, "image_classification")
    val_writer.add_schema(schema, "image_classification")
    test_writer.add_schema(schema, "image_classification")
    
    return train_writer, val_writer, test_writer


def process_dataset_parallel(image_files, output_train, output_val, output_test, num_workers=None):
    """
    并行处理数据集
    
    Args:
        image_files: 图像文件列表
        output_train: 训练集输出文件
        output_val: 验证集输出文件
        output_test: 测试集输出文件
        num_workers: 并行处理的工作进程数，默认为CPU核心数
    """
    if not image_files:
        print("No image files to process.")
        return
    
    # 如果未指定工作进程数，使用CPU核心数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # 确保工作进程数至少为1
    num_workers = max(1, num_workers)
    
    print(f"Starting parallel processing with {num_workers} workers...")
    
    # 划分数据集
    train_files, val_files, test_files = split_dataset(image_files)
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_train)), exist_ok=True)
    
    # 处理训练集
    print(f"Processing training set ({len(train_files)} images)...")
    train_data = process_file_batch(train_files, num_workers)
    if train_data:
        write_mindrecord(train_data, output_train)
        print(f"Training set saved to {output_train}")
    else:
        print("Warning: No valid images found for training set.")
    
    # 处理验证集
    print(f"Processing validation set ({len(val_files)} images)...")
    val_data = process_file_batch(val_files, num_workers)
    if val_data:
        write_mindrecord(val_data, output_val)
        print(f"Validation set saved to {output_val}")
    else:
        print("Warning: No valid images found for validation set.")
    
    # 处理测试集
    print(f"Processing test set ({len(test_files)} images)...")
    test_data = process_file_batch(test_files, num_workers)
    if test_data:
        write_mindrecord(test_data, output_test)
        print(f"Test set saved to {output_test}")
    else:
        print("Warning: No valid images found for test set.")
    
    print("Dataset processing completed.")


def process_file_batch(files, num_workers):
    """
    并行处理一批文件
    
    Args:
        files: 文件列表
        num_workers: 工作进程数
    
    Returns:
        处理后的数据列表
    """
    processed_data = []
    error_count = 0
    success_count = 0
    
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # 使用tqdm显示进度条
            results = list(tqdm(
                pool.imap(process_image_for_parallel, [(file['path'], file, file['label_id']) for file in files]),
                total=len(files),
                desc="Processing images"
            ))
            
            # 过滤掉处理失败的结果
            for result in results:
                if result is not None:
                    processed_data.append(result)
                    success_count += 1
                else:
                    error_count += 1
    except Exception as e:
        print(f"Error during parallel processing: {str(e)}")
    
    # 打印处理统计信息
    total = len(files)
    print(f"Processing complete: {success_count}/{total} images processed successfully ({error_count} errors)")
    
    return processed_data


def process_dataset_serial(train_files, val_files, test_files):
    """
    串行处理数据集
    
    Args:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        test_files: 测试集文件列表
        
    Returns:
        处理后的训练集、验证集和测试集数据
    """
    # 准备数据
    train_data = []
    val_data = []
    test_data = []
    errors = 0
    
    # 合并所有文件
    all_files = train_files + val_files + test_files
    
    print("Using serial processing")
    
    # 使用tqdm创建进度条
    with tqdm(total=len(all_files), desc="Processing", unit="img") as pbar:
        # 串行处理图像
        for file_info in all_files:
            image_path = file_info['path']
            label_id = file_info['label_id']
            
            try:
                img_bytes = process_image(image_path)
                if img_bytes is None:
                    print(f"Warning: Could not read image {image_path}")
                    errors += 1
                    pbar.update(1)
                    continue
                
                data = {
                    "image": img_bytes,
                    "label": label_id
                }
                
                # 根据文件信息将数据添加到不同的数据集
                if file_info in train_files:
                    train_data.append(data)
                elif file_info in val_files:
                    val_data.append(data)
                elif file_info in test_files:
                    test_data.append(data)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                errors += 1
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "train": len(train_data), 
                "val": len(val_data), 
                "test": len(test_data), 
                "errors": errors
            })
    
    return train_data, val_data, test_data, errors


def save_mindrecord_data(writers, datasets):
    """
    保存数据到MindRecord文件
    
    Args:
        writers: 包含训练集、验证集和测试集写入器的元组
        datasets: 包含训练集、验证集和测试集数据的元组
    """
    train_writer, val_writer, test_writer = writers
    train_data, val_data, test_data = datasets
    
    print("\nSaving datasets...")
    
    # 保存训练集
    if train_data:
        train_writer.write_raw_data(train_data)
        train_writer.commit()
        print(f"Training data saved, {len(train_data)} images")
    
    # 保存验证集
    if val_data:
        val_writer.write_raw_data(val_data)
        val_writer.commit()
        print(f"Validation data saved, {len(val_data)} images")
    
    # 保存测试集
    if test_data:
        test_writer.write_raw_data(test_data)
        test_writer.commit()
        print(f"Test data saved, {len(test_data)} images")
    
    # 打印统计信息
    print(f"\nSummary:")
    print(f"Train/Val/Test split: {len(train_data)}/{len(val_data)}/{len(test_data)}")


def process_dataset_sequential(image_files, output_train, output_val, output_test):
    """
    顺序处理数据集
    
    Args:
        image_files: 图像文件列表
        output_train: 训练集输出文件
        output_val: 验证集输出文件
        output_test: 测试集输出文件
    """
    if not image_files:
        print("No image files to process.")
        return
    
    print("Starting sequential processing...")
    
    # 划分数据集
    train_files, val_files, test_files = split_dataset(image_files)
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_train)), exist_ok=True)
    
    # 处理训练集
    print(f"Processing training set ({len(train_files)} images)...")
    train_data = []
    for i, file_info in enumerate(tqdm(train_files, desc="Processing training set")):
        result = process_image_for_parallel((file_info['path'], file_info, file_info['label_id']))
        if result is not None:
            train_data.append(result)
    
    if train_data:
        write_mindrecord(train_data, output_train)
        print(f"Training set saved to {output_train}")
    else:
        print("Warning: No valid images found for training set.")
    
    # 处理验证集
    print(f"Processing validation set ({len(val_files)} images)...")
    val_data = []
    for i, file_info in enumerate(tqdm(val_files, desc="Processing validation set")):
        result = process_image_for_parallel((file_info['path'], file_info, file_info['label_id']))
        if result is not None:
            val_data.append(result)
    
    if val_data:
        write_mindrecord(val_data, output_val)
        print(f"Validation set saved to {output_val}")
    else:
        print("Warning: No valid images found for validation set.")
    
    # 处理测试集
    print(f"Processing test set ({len(test_files)} images)...")
    test_data = []
    for i, file_info in enumerate(tqdm(test_files, desc="Processing test set")):
        result = process_image_for_parallel((file_info['path'], file_info, file_info['label_id']))
        if result is not None:
            test_data.append(result)
    
    if test_data:
        write_mindrecord(test_data, output_test)
        print(f"Test set saved to {output_test}")
    else:
        print("Warning: No valid images found for test set.")
    
    print("Dataset processing completed.")


def write_mindrecord(data, output_file):
    """
    将数据写入MindRecord文件
    
    Args:
        data: 数据列表
        output_file: 输出文件路径
    """
    if not data:
        print(f"No data to write to {output_file}")
        return
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 如果文件已存在，先删除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # 创建MindRecord文件写入器
    writer = FileWriter(output_file, shard_num=1)
    
    # 定义数据架构
    schema = {
        "image": {"type": "bytes"},
        "label": {"type": "int32"}
    }
    
    # 添加架构
    writer.add_schema(schema)
    
    # 写入数据
    writer.write_raw_data(data)
    
    # 提交写入
    writer.commit()
    
    print(f"Successfully wrote {len(data)} records to {output_file}") 