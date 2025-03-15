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

from models.mobilenetv3 import MobileNetV3
from utils.dataset import create_dataset, create_dataset_from_mindrecord
from config.config import Config

def train_model(num_workers=None, use_mindrecord=False, train_mindrecord=None, val_mindrecord=None):
    """
    训练模型的主函数
    
    Args:
        num_workers: 并行处理的工作进程数，如果为None则使用Config中的设置
        use_mindrecord: 是否使用MindRecord数据集
        train_mindrecord: 训练集MindRecord文件路径
        val_mindrecord: 验证集MindRecord文件路径
    """
    # 设置随机种子
    set_seed(1)
    
    # 设置运行环境
    context.set_context(mode=context.GRAPH_MODE, device_target=Config.device_target)
    
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
    model.train(
        Config.num_epochs,
        train_dataset,
        callbacks=[ckpoint_cb, LossMonitor(125)],
        dataset_sink_mode=True
    )
    
    # 评估模型
    metrics = model.eval(val_dataset)
    print(f"Validation Accuracy: {metrics['acc']:.4f}")
    
    # 保存最佳模型
    best_ckpt = os.path.join(Config.model_save_dir, "mobilenetv3-final.ckpt")
    if os.path.exists(best_ckpt):
        # 复制最终检查点作为最佳模型
        import shutil
        shutil.copy(best_ckpt, Config.best_model_path)
        print(f"Best model saved to {Config.best_model_path}")
    
    return metrics['acc']

if __name__ == '__main__':
    train_model() 