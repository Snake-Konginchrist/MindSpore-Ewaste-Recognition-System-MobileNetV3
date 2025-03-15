"""
训练脚本：包含训练循环和评估函数
"""
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.model import Model
from mindspore.common import set_seed
import numpy as np
import time

# 修改导入路径，适应新的文件结构
from core.mobilenetv3 import MobileNetV3
from core.dataset import create_dataset, create_dataset_from_mindrecord
from core.config import Config

def check_dataset(dataset, name="dataset"):
    """
    检查数据集的格式和内容
    
    Args:
        dataset: 要检查的数据集
        name: 数据集名称，用于打印信息
    """
    print(f"\nChecking {name}...")
    try:
        # 获取数据集大小
        dataset_size = dataset.get_dataset_size()
        print(f"{name} size: {dataset_size}")
        
        # 创建数据迭代器
        iterator = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
        
        # 检查第一个批次
        batch = next(iterator)
        print(f"First batch keys: {list(batch.keys())}")
        
        # 检查图像数据
        images = batch["image"]
        print(f"Image batch shape: {images.shape}")
        print(f"Image data type: {images.dtype}")
        print(f"Image min/max values: {np.min(images):.4f}/{np.max(images):.4f}")
        
        # 检查标签数据
        labels = batch["label"]
        print(f"Label batch shape: {labels.shape}")
        print(f"Label data type: {labels.dtype}")
        print(f"Label values: {labels}")
        
        print(f"{name} check completed successfully\n")
        return True
    except Exception as e:
        print(f"Error checking {name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def train_model(num_workers=None, use_mindrecord=False, train_mindrecord=None, val_mindrecord=None):
    """
    训练模型的主函数
    
    Args:
        num_workers: 并行处理的工作进程数，如果为None则使用Config中的设置
        use_mindrecord: 是否使用MindRecord数据集
        train_mindrecord: 训练集MindRecord文件路径
        val_mindrecord: 验证集MindRecord文件路径
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 设置随机种子
        set_seed(1)
        
        # 设置运行环境
        context.set_context(mode=context.GRAPH_MODE)
        ms.set_device(Config.device_target)
        
        # 设置数据加载的并行工作进程数
        if num_workers is None:
            dataset_num_workers = Config.num_workers
        else:
            dataset_num_workers = num_workers
        
        # 创建数据集
        if use_mindrecord and train_mindrecord and val_mindrecord:
            print(f"Using MindRecord dataset:")
            print(f"  Training set: {train_mindrecord}")
            print(f"  Validation set: {val_mindrecord}")
            
            train_dataset = create_dataset_from_mindrecord(
                train_mindrecord,
                Config,
                is_train=True,
                shuffle=True,
                num_workers=dataset_num_workers
            )
            
            val_dataset = create_dataset_from_mindrecord(
                val_mindrecord,
                Config,
                is_train=False,
                shuffle=False,
                num_workers=dataset_num_workers
            )
        else:
            print(f"Using original image dataset:")
            print(f"  Training set directory: {Config.train_data_dir}")
            print(f"  Validation set directory: {Config.val_data_dir}")
            
            train_dataset = create_dataset(
                Config.train_data_dir,
                Config,
                is_train=True,
                shuffle=True,
                num_workers=dataset_num_workers
            )
            
            val_dataset = create_dataset(
                Config.val_data_dir,
                Config,
                is_train=False,
                shuffle=False,
                num_workers=dataset_num_workers
            )
        
        # 检查数据集
        train_ok = check_dataset(train_dataset, "Training dataset")
        val_ok = check_dataset(val_dataset, "Validation dataset")
        
        if not train_ok or not val_ok:
            print("Dataset check failed, aborting training")
            return None
        
        # 创建模型实例
        network = MobileNetV3(
            num_classes=Config.num_classes,
            width_mult=1.0,
            mode='small'  # 使用small版本，适合较小的数据集
        )
        
        # 定义损失函数和优化器
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        
        # 使用余弦退火学习率调度器
        lr = nn.cosine_decay_lr(
            min_lr=0.0,
            max_lr=Config.learning_rate,
            total_step=Config.num_epochs * train_dataset.get_dataset_size(),
            step_per_epoch=train_dataset.get_dataset_size(),
            decay_epoch=Config.num_epochs
        )
        
        optimizer = nn.Momentum(
            network.trainable_params(),
            learning_rate=lr,
            momentum=Config.momentum,
            weight_decay=Config.weight_decay
        )
        
        # 定义训练模型
        model = Model(
            network,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics={'acc'}
        )
        
        # 设置模型检查点
        config_ck = CheckpointConfig(
            save_checkpoint_steps=train_dataset.get_dataset_size(),
            keep_checkpoint_max=5
        )
        
        if not os.path.exists(Config.model_save_dir):
            os.makedirs(Config.model_save_dir)
            
        ckpoint_cb = ModelCheckpoint(
            prefix="mobilenetv3",
            directory=Config.model_save_dir,
            config=config_ck
        )
        
        # 开始训练
        print("\nStarting model training...")
        model.train(
            Config.num_epochs,
            train_dataset,
            callbacks=[ckpoint_cb, LossMonitor(125), time_cb],
            dataset_sink_mode=False  # 在CPU上禁用数据下沉模式以获得更准确的时间统计
        )
        
        # 评估模型
        print("\nEvaluating model...")
        metrics = model.eval(val_dataset)
        print(f"Validation Accuracy: {metrics['acc']:.4f}")
        
        # 保存最佳模型
        best_ckpt = os.path.join(Config.model_save_dir, "mobilenetv3-final.ckpt")
        if os.path.exists(best_ckpt):
            # 复制最终检查点作为最佳模型
            import shutil
            shutil.copy(best_ckpt, Config.best_model_path)
            print(f"Best model saved to {Config.best_model_path}")
        
        # 计算并打印总用时
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "="*50)
        print(f"Training completed!")
        print(f"Total time: {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds")
        print(f"Number of epochs: {Config.num_epochs}")
        print(f"Validation accuracy: {metrics['acc']:.4f}")
        print("="*50 + "\n")
        
        return metrics['acc']
    
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    train_model() 