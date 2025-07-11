# API 文档

## 核心模块

### core.mobilenetv3

#### MobileNetV3 类

```python
class MobileNetV3(nn.Cell):
    """
    MobileNetV3 网络模型
    
    参数:
        num_classes (int): 分类类别数量
        width_multiplier (float): 网络宽度倍数
        reduced_tail (bool): 是否使用简化尾部
        dilated (bool): 是否使用空洞卷积
    """
```

**主要方法**:
- `construct(x)`: 前向传播
- `features(x)`: 特征提取
- `classifier(x)`: 分类器

### core.dataset

#### EwasteDataset 类

```python
class EwasteDataset:
    """
    电子垃圾数据集类
    
    参数:
        data_dir (str): 数据目录路径
        transform (callable): 数据变换函数
        is_training (bool): 是否为训练模式
    """
```

**主要方法**:
- `__getitem__(index)`: 获取单个样本
- `__len__()`: 返回数据集大小
- `get_categories()`: 获取类别列表

### core.train

#### 训练函数

```python
def train_model(model, train_loader, val_loader, config):
    """
    训练模型
    
    参数:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置
    
    返回:
        trained_model: 训练完成的模型
    """
```

### core.evaluate

#### 评估函数

```python
def evaluate_model(model, test_loader):
    """
    评估模型性能
    
    参数:
        model: 模型实例
        test_loader: 测试数据加载器
    
    返回:
        accuracy: 准确率
        predictions: 预测结果
    """
```

## 数据处理模块

### data_processing.core.dataset_processor

#### DatasetProcessor 类

```python
class DatasetProcessor:
    """
    数据集处理器
    
    参数:
        datasets_dir (str): 数据集目录
        testsets_dir (str): 测试集目录
    """
```

**主要方法**:
- `discover_categories()`: 发现数据集类别
- `process_dataset(parallel=False, cpu_cores=None)`: 处理数据集
- `get_dataset_info()`: 获取数据集信息

### data_processing.core.image_processor

#### 图像处理函数

```python
def process_image(image_path, target_size=(224, 224)):
    """
    处理单张图像
    
    参数:
        image_path (str): 图像路径
        target_size (tuple): 目标尺寸
    
    返回:
        processed_image: 处理后的图像
    """
```

```python
def process_image_for_parallel(args):
    """
    并行处理图像
    
    参数:
        args (tuple): 包含图像路径和处理参数的元组
    
    返回:
        processed_image: 处理后的图像
    """
```

## 用户界面模块

### ui.ewaste_ui

#### EwasteRecognitionUI 类

```python
class EwasteRecognitionUI(QMainWindow):
    """
    电子垃圾识别图形界面
    
    参数:
        model_path (str): 模型文件路径
    """
```

**主要方法**:
- `setup_ui()`: 设置用户界面
- `load_model()`: 加载模型
- `recognize_image(image_path)`: 识别图像
- `dragEnterEvent(event)`: 拖拽进入事件
- `dropEvent(event)`: 拖拽释放事件

### data_processing.utils.user_interface

#### 用户界面工具函数

```python
def select_processing_mode():
    """
    选择处理模式
    
    返回:
        bool: True为并行处理，False为顺序处理
    """
```

```python
def select_cpu_cores():
    """
    选择CPU核心数
    
    返回:
        int: 选择的CPU核心数
    """
```

## 图像预处理模块

### image_preprocessing.image_analyzer

#### ImageAnalyzer 类

```python
class ImageAnalyzer:
    """
    图像分析器
    
    参数:
        data_dir (str): 数据目录
    """
```

**主要方法**:
- `analyze_dataset()`: 分析数据集
- `generate_report()`: 生成分析报告
- `plot_statistics()`: 绘制统计图表

### image_preprocessing.image_validator

#### ImageValidator 类

```python
class ImageValidator:
    """
    图像验证器
    
    参数:
        data_dir (str): 数据目录
    """
```

**主要方法**:
- `validate_images()`: 验证图像
- `fix_corrupted_images()`: 修复损坏图像
- `generate_validation_report()`: 生成验证报告

### image_preprocessing.image_converter

#### ImageConverter 类

```python
class ImageConverter:
    """
    图像转换器
    
    参数:
        input_dir (str): 输入目录
        output_dir (str): 输出目录
    """
```

**主要方法**:
- `convert_format(target_format)`: 转换格式
- `resize_images(target_size)`: 调整图像大小
- `batch_process()`: 批量处理

## 配置模块

### core.config

#### 配置类

```python
class Config:
    """
    系统配置类
    
    属性:
        num_classes (int): 类别数量
        image_size (tuple): 图像尺寸
        batch_size (int): 批次大小
        learning_rate (float): 学习率
        num_epochs (int): 训练轮数
        device (str): 设备类型
    """
```

## 主程序

### ewaste_recognition

#### 主函数

```python
def main():
    """
    主程序入口
    
    功能:
        - 显示主菜单
        - 处理用户选择
        - 启动相应功能
    """
```

```python
def train_new_model():
    """
    训练新模型
    
    功能:
        - 配置训练参数
        - 启动模型训练
        - 保存训练结果
    """
```

```python
def recognize_ewaste():
    """
    识别电子垃圾
    
    功能:
        - 加载训练好的模型
        - 处理输入图像
        - 返回识别结果
    """
```

## 使用示例

### 基本使用

```python
# 导入模块
from core.mobilenetv3 import MobileNetV3
from core.dataset import EwasteDataset
from core.train import train_model

# 创建模型
model = MobileNetV3(num_classes=11)

# 创建数据集
dataset = EwasteDataset(data_dir="datasets/", is_training=True)

# 训练模型
trained_model = train_model(model, train_loader, val_loader, config)
```

### 图像识别

```python
# 导入模块
from core.evaluate import evaluate_model
from ui.ewaste_ui import EwasteRecognitionUI

# 创建界面
ui = EwasteRecognitionUI(model_path="checkpoints/best_model.ckpt")

# 识别图像
result = ui.recognize_image("test_image.jpg")
print(f"识别结果: {result}")
```

### 数据处理

```python
# 导入模块
from data_processing.core.dataset_processor import DatasetProcessor

# 创建处理器
processor = DatasetProcessor("datasets/", "testsets/")

# 处理数据集
processor.process_dataset(parallel=True, cpu_cores=4)
```

## 注意事项

1. **模型加载**: 确保模型文件路径正确
2. **数据格式**: 图像数据需要符合指定格式
3. **内存管理**: 大数据集处理时注意内存使用
4. **并行处理**: 根据硬件配置选择合适的并行参数 