"""
训练脚本：包含训练循环和评估函数
"""
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import LossMonitor, Callback, TimeMonitor
from mindspore.train.model import Model
from mindspore.common import set_seed
import numpy as np
import time
import multiprocessing
import shutil

# 修改导入路径，适应新的文件结构
from core.mobilenetv3 import MobileNetV3
from core.dataset import create_dataset, create_dataset_from_mindrecord
from core.config import Config

class EarlyStoppingCallback(Callback):
    """
    早停回调：当验证指标不再改善时停止训练
    
    Args:
        patience: 在停止前等待的轮数
        monitor: 监控的指标，例如'acc'
        mode: 'max'表示更高的指标更好（如准确率），'min'表示更低的指标更好（如损失）
        min_delta: 被视为改进的最小变化量
        verbose: 是否打印早停信息
    """
    def __init__(self, patience=10, monitor='acc', mode='max', min_delta=0.001, verbose=True):
        super(EarlyStoppingCallback, self).__init__()
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = None
        self.best_epoch = 0
        
        if mode == 'max':
            self.monitor_op = lambda a, b: a > b + min_delta
        elif mode == 'min':
            self.monitor_op = lambda a, b: a < b - min_delta
        else:
            raise ValueError(f"Mode {mode} is not supported. Use 'max' or 'min'.")
    
    def on_train_begin(self, run_context):
        # 初始化最佳值
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        if self.verbose:
            print(f"Early stopping initialized: monitoring {self.monitor}, patience={self.patience}")
    
    def on_eval_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        metrics = cb_params.metrics
        
        # 获取监控的值
        if self.monitor in metrics:
            current = metrics[self.monitor]
        else:
            if self.verbose:
                print(f"Warning: {self.monitor} not found in metrics. Available metrics: {metrics.keys()}")
            return
        
        if self.verbose:
            print(f"Epoch {epoch}: {self.monitor} = {current:.4f}")
        
        # 检查是否有改进
        if self.monitor_op(current, self.best_value):
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved to {current:.4f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} did not improve. Patience: {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                run_context.request_stop()
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch}")
    
    def on_train_end(self, run_context):
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Training stopped early at epoch {self.stopped_epoch}")
            print(f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch}")

class ValidationCallback(Callback):
    """
    验证回调：在每个epoch结束后进行验证
    
    Args:
        model: 训练模型
        val_dataset: 验证数据集
        early_stop_cb: 早停回调实例
    """
    def __init__(self, model, val_dataset, early_stop_cb=None):
        super(ValidationCallback, self).__init__()
        self.model = model
        self.val_dataset = val_dataset
        self.early_stop_cb = early_stop_cb
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_ckpt_path = None
    
    def on_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        print(f"\nEvaluating after epoch {epoch}...")
        metrics = self.model.eval(self.val_dataset)
        print(f"Validation Accuracy: {metrics['acc']:.4f}")
        
        # 记录最佳精度并保存最佳模型
        if metrics['acc'] > self.best_acc:
            self.best_acc = metrics['acc']
            self.best_epoch = epoch
            print(f"New best accuracy: {self.best_acc:.4f}")
            
            # 保存当前最佳模型
            current_ckpt = os.path.join(Config.model_save_dir, f"mobilenetv3-{epoch}_{cb_params.batch_num}.ckpt")
            if os.path.exists(current_ckpt):
                self.best_ckpt_path = current_ckpt
                # 立即复制为best_model.ckpt
                try:
                    shutil.copy(current_ckpt, Config.best_model_path)
                    print(f"Best model saved to {Config.best_model_path}")
                except Exception as e:
                    print(f"Failed to save best model: {e}")
        
        # 手动触发早停回调
        if self.early_stop_cb:
            # 更新回调参数
            cb_params.metrics = metrics
            # 创建一个新的运行上下文
            from mindspore.train.callback import _RunContext
            early_stop_context = _RunContext(cb_params)
            # 调用早停回调
            self.early_stop_cb.on_eval_end(early_stop_context)
            # 检查是否请求停止
            if hasattr(early_stop_context, '_is_stop') and early_stop_context._is_stop:
                run_context.request_stop()
    
    def on_train_end(self, run_context):
        """训练结束时确保最佳模型已保存"""
        if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
            try:
                # 再次确保最佳模型被保存为best_model.ckpt
                shutil.copy(self.best_ckpt_path, Config.best_model_path)
                print(f"\nTraining completed. Best model (epoch {self.best_epoch}, accuracy {self.best_acc:.4f}) saved to {Config.best_model_path}")
            except Exception as e:
                print(f"Failed to save best model at training end: {e}")

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

def setup_parallel_environment(num_workers=None):
    """
    设置并行训练环境
    
    Args:
        num_workers: 要使用的CPU核心数，如果为None则使用所有可用核心
    
    Returns:
        int: 实际使用的CPU核心数
    """
    # 获取可用的CPU核心数
    total_cores = multiprocessing.cpu_count()
    
    # 如果未指定核心数，使用所有可用核心
    if num_workers is None:
        num_workers = total_cores
    else:
        # 确保核心数在有效范围内
        num_workers = max(1, min(num_workers, total_cores))
    
    # 设置环境变量
    os.environ['OMP_NUM_THREADS'] = str(num_workers)
    
    # 设置MindSpore上下文
    context.set_context(mode=context.GRAPH_MODE)
    ms.set_device(Config.device_target)
    
    # 如果是CPU设备，设置线程数
    if Config.device_target.upper() == "CPU":
        try:
            context.set_context(max_device_memory="0GB")  # 对CPU模式，设置为0
            context.set_context(max_call_depth=10000)
            context.set_context(enable_parallel_optimizer=True)  # 启用并行优化器
            
            # 设置CPU线程数
            context.set_context(thread_num=num_workers)
            print(f"Set CPU thread number to: {num_workers}")
        except Exception as e:
            print(f"Failed to set CPU thread number: {e}")
    
    print(f"Parallel environment setup completed. Using {num_workers}/{total_cores} CPU cores.")
    return num_workers

def train_model(num_workers=None, use_mindrecord=False, train_mindrecord=None, val_mindrecord=None, 
                early_stopping=True, patience=10):
    """
    训练模型的主函数
    
    Args:
        num_workers: 并行处理的工作进程数，如果为None则使用Config中的设置
        use_mindrecord: 是否使用MindRecord数据集
        train_mindrecord: 训练集MindRecord文件路径
        val_mindrecord: 验证集MindRecord文件路径
        early_stopping: 是否启用早停机制
        patience: 早停的耐心值，即在多少个epoch没有改进后停止训练
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 设置随机种子
        set_seed(1)
        
        # 设置并行环境
        actual_workers = setup_parallel_environment(num_workers)
        
        # 设置数据加载的并行工作进程数
        if num_workers is None:
            # 使用CPU核心数的75%作为工作进程数，避免系统过载
            dataset_num_workers = max(1, int(actual_workers * 0.75))
            print(f"Auto-setting dataset workers: {dataset_num_workers}")
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
        
        # 使用并行优化的动量优化器
        optimizer = nn.Momentum(
            network.trainable_params(),
            learning_rate=lr,
            momentum=Config.momentum,
            weight_decay=Config.weight_decay,
            use_nesterov=True  # 使用Nesterov加速梯度，通常能提高收敛速度
        )
        
        # 尝试启用混合精度训练
        try:
            from mindspore import amp
            network = amp.auto_mixed_precision(network, "O2")
            print("Auto mixed precision training enabled")
        except Exception as e:
            print(f"Failed to enable mixed precision training: {e}")
        
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
        
        # 设置回调
        callbacks = [LossMonitor(125), ckpoint_cb]
        
        # 添加时间监控回调
        time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
        callbacks.append(time_cb)
        
        # 添加早停和验证回调
        if early_stopping:
            early_stop_cb = EarlyStoppingCallback(
                patience=patience,
                monitor='acc',
                mode='max',
                min_delta=0.001,
                verbose=True
            )
            val_cb = ValidationCallback(model, val_dataset, early_stop_cb)
            callbacks.append(val_cb)
            print(f"Early stopping enabled with patience={patience}")
        else:
            val_cb = ValidationCallback(model, val_dataset)
            callbacks.append(val_cb)
        
        # 打印训练配置信息
        print("\nTraining configuration:")
        print(f"  CPU cores: {actual_workers}")
        print(f"  Dataset workers: {dataset_num_workers}")
        print(f"  Batch size: {Config.batch_size}")
        print(f"  Learning rate: {Config.learning_rate}")
        print(f"  Epochs: {Config.num_epochs}")
        print(f"  Early stopping: {'Enabled' if early_stopping else 'Disabled'}")
        print(f"  Best model will be saved to: {Config.best_model_path}")
        
        # 开始训练
        print("\nStarting model training...")
        model.train(
            Config.num_epochs,
            train_dataset,
            callbacks=callbacks,
            dataset_sink_mode=False  # 在CPU上禁用数据下沉模式以获得更准确的时间统计
        )
        
        # 评估模型
        print("\nEvaluating model...")
        metrics = model.eval(val_dataset)
        print(f"Final validation accuracy: {metrics['acc']:.4f}")
        
        # 如果训练过程中没有保存最佳模型，则在这里保存
        if not os.path.exists(Config.best_model_path) or val_cb.best_ckpt_path is None:
            # 查找最新的检查点
            ckpt_files = [f for f in os.listdir(Config.model_save_dir) if f.endswith('.ckpt')]
            if ckpt_files:
                # 按修改时间排序，获取最新的检查点
                latest_ckpt = sorted(ckpt_files, key=lambda x: os.path.getmtime(os.path.join(Config.model_save_dir, x)))[-1]
                latest_ckpt_path = os.path.join(Config.model_save_dir, latest_ckpt)
                
                # 复制为最佳模型
                shutil.copy(latest_ckpt_path, Config.best_model_path)
                print(f"Latest model checkpoint {latest_ckpt} saved as best model to {Config.best_model_path}")
        
        # 计算并打印总用时
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "="*50)
        print(f"Training completed!")
        print(f"Total time: {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds")
        print(f"Number of epochs: {Config.num_epochs}")
        print(f"Best validation accuracy: {val_cb.best_acc:.4f} (epoch {val_cb.best_epoch})")
        print(f"Best model saved to: {Config.best_model_path}")
        print(f"CPU cores used: {actual_workers}")
        print("="*50 + "\n")
        
        return metrics['acc']
    
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # 使用早停机制，设置耐心值为10
    train_model(early_stopping=True, patience=10)

    # 或者禁用早停机制
    # train_model(early_stopping=False) 