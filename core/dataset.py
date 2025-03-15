"""
数据集处理模块：负责数据加载和预处理
"""
import os
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
from PIL import Image
import numpy as np

class EWasteDataset:
    """电子垃圾数据集类"""
    
    def __init__(self, data_dir, config, is_train=True):
        """
        初始化数据集
        Args:
            data_dir: 数据集路径
            config: 配置对象
            is_train: 是否为训练集
        """
        self.data_dir = data_dir
        self.config = config
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        self._load_dataset()
        
    def _load_dataset(self):
        """加载数据集，获取图片路径和标签"""
        for class_idx, class_name in enumerate(self.config.class_names):
            class_dir = os.path.join(self.data_dir, f"{class_name}_datasets")
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
    
    def __getitem__(self, index):
        """获取单个数据样本"""
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        # 读取图像
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            
        return img, label
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)


def create_dataset(data_dir, config, is_train=True, shuffle=True, num_workers=None):
    """
    创建数据集实例
    Args:
        data_dir: 数据集路径
        config: 配置对象
        is_train: 是否为训练集
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数，如果为None则使用配置中的设置
    Returns:
        dataset: MindSpore数据集对象
    """
    dataset_generator = EWasteDataset(data_dir, config, is_train)
    
    # 设置工作进程数
    if num_workers is None:
        num_workers = config.num_workers
    
    # 创建数据集
    dataset = ds.GeneratorDataset(
        dataset_generator,
        column_names=["image", "label"],
        shuffle=shuffle,
        num_parallel_workers=num_workers
    )
    
    # 数据增强和预处理
    transform_list = []
    
    # 训练集数据增强
    if is_train:
        if config.random_crop:
            transform_list.extend([
                vision.Resize((256, 256), Inter.BILINEAR),
                vision.RandomCrop((config.image_size, config.image_size)),
                vision.RandomHorizontalFlip(prob=0.5)
            ])
    else:
        transform_list.extend([
            vision.Resize((config.image_size, config.image_size), Inter.BILINEAR)
        ])
    
    # 通用预处理
    transform_list.extend([
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        vision.HWC2CHW()
    ])
    
    dataset = dataset.map(
        operations=transform_list,
        input_columns="image",
        num_parallel_workers=num_workers
    )
    
    # 设置批量大小
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    
    return dataset


def create_dataset_from_mindrecord(mindrecord_file, config, is_train=True, shuffle=True, num_workers=None):
    """
    从MindRecord文件创建数据集实例
    Args:
        mindrecord_file: MindRecord文件路径
        config: 配置对象
        is_train: 是否为训练集
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数，如果为None则使用配置中的设置
    Returns:
        dataset: MindSpore数据集对象
    """
    # 设置工作进程数
    if num_workers is None:
        num_workers = config.num_workers
    
    try:
        # 从MindRecord文件创建数据集
        dataset = ds.MindDataset(
            dataset_files=mindrecord_file,
            columns_list=["image", "label"],
            shuffle=shuffle,
            num_parallel_workers=num_workers
        )
        
        # 打印数据集信息
        print(f"MindDataset created successfully from {mindrecord_file}")
        print(f"Dataset size: {dataset.get_dataset_size()}")
        
        # 解码图像
        decode_op = vision.Decode()
        dataset = dataset.map(
            operations=decode_op,
            input_columns=["image"],
            num_parallel_workers=num_workers
        )
        
        # 确保标签是标量而不是元组 - 使用更直接的方式
        # 定义一个Python函数来处理标签
        def ensure_scalar_label(label):
            """确保标签是标量"""
            if isinstance(label, (tuple, list)):
                return np.array(label[0], dtype=np.int32)
            return np.array(label, dtype=np.int32)
        
        # 使用Python函数处理标签
        dataset = dataset.map(
            operations=ensure_scalar_label,
            input_columns=["label"],
            output_columns=["label"],
            python_multiprocessing=False,  # 避免多进程可能导致的问题
            num_parallel_workers=1  # 使用单线程处理标签
        )
        
        # 使用TypeCast确保数据类型正确
        type_cast_op = transforms.TypeCast(mstype.int32)
        dataset = dataset.map(
            operations=type_cast_op,
            input_columns=["label"],
            num_parallel_workers=num_workers
        )
        
        # 数据增强和预处理
        transform_list = []
        
        # 训练集数据增强
        if is_train:
            if config.random_crop:
                transform_list.extend([
                    vision.Resize((256, 256), Inter.BILINEAR),
                    vision.RandomCrop((config.image_size, config.image_size)),
                    vision.RandomHorizontalFlip(prob=0.5)
                ])
        else:
            transform_list.extend([
                vision.Resize((config.image_size, config.image_size), Inter.BILINEAR)
            ])
        
        # 通用预处理
        transform_list.extend([
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ])
        
        dataset = dataset.map(
            operations=transform_list,
            input_columns="image",
            num_parallel_workers=num_workers
        )
        
        # 设置批量大小
        dataset = dataset.batch(config.batch_size, drop_remainder=True)
        
        print(f"Dataset preprocessing completed successfully")
        return dataset
        
    except Exception as e:
        print(f"Error creating dataset from MindRecord: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def find_mindrecord_files(data_dir="./datasets"):
    """
    在指定目录中查找所有MindRecord文件
    
    Args:
        data_dir: 数据集目录
        
    Returns:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        test_files: 测试集文件列表
    """
    train_files = []
    val_files = []
    test_files = []
    
    # 遍历目录查找MindRecord文件
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".mindrecord"):
                file_path = os.path.join(data_dir, filename)
                
                # 根据文件名分类
                if "_train" in filename:
                    train_files.append(file_path)
                elif "_val" in filename:
                    val_files.append(file_path)
                elif "_test" in filename:
                    test_files.append(file_path)
    
    # 对文件列表排序
    train_files.sort()
    val_files.sort()
    test_files.sort()
    
    return train_files, val_files, test_files 