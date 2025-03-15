# 电子垃圾图像识别分类系统

基于 MindSpore 框架和 MobileNetV3 网络的电子垃圾图像分类系统，用于识别和分类不同类型的电子废弃物。

## 项目概述

本项目实现了一个轻量级的电子垃圾图像分类系统，可以识别包括相机、机箱、键盘、笔记本电脑等在内的多种电子废弃物。系统采用了 MobileNetV3-Small 架构，在保证准确率的同时，具有较小的模型体积和较快的推理速度。

### 支持的电子垃圾类别

- 相机 (Camera)
- 机箱 (Chassis)
- 键盘 (Keyboard)
- 笔记本电脑 (Laptop)
- 显示器 (Monitor)
- 鼠标 (Mouse)
- 收音机 (Radio)
- 路由器 (Router)
- 智能手机 (Smartphone)
- 电话 (Telephone)
- 其他 (Others)

## 技术架构

### MobileNetV3 网络结构

MobileNetV3 是 Google 提出的移动端神经网络架构，本项目使用其 Small 版本，具有以下特点：

1. **创新点**：
   - 结合 NAS（神经架构搜索）和 NetAdapt 算法
   - 引入 Hard Swish 激活函数
   - 改进的 SE（Squeeze-and-Excitation）模块
   - 优化的网络结构

2. **核心模块**：

   a. **倒置残差块 (Inverted Residual Block)**：
   ```
   输入 → 1x1 PW扩展 → DW卷积 → SE模块 → 1x1 PW压缩 → 输出
   ```
   
   b. **SE 注意力模块**：
   ```
   特征图 → 全局平均池化 → 1x1 Conv降维 → ReLU → 1x1 Conv升维 → Hard Sigmoid → 特征重标定
   ```
   
   c. **Hard Swish 激活函数**：
   ```
   H-Swish(x) = x * ReLU6(x + 3) / 6
   ```

3. **网络架构详细参数**：

   | 层 | 输入大小 | 算子 | 扩展因子 | 输出通道 | SE | HS | 步长 |
   |---|---------|------|----------|----------|----|----|-----|
   | 0 | 224×224 | Conv2d 3×3 | - | 16 | - | √ | 2 |
   | 1 | 112×112 | bneck 3×3 | 1 | 16 | √ | - | 2 |
   | 2 | 56×56 | bneck 3×3 | 4.5 | 24 | - | - | 2 |
   | 3 | 28×28 | bneck 3×3 | 3.67 | 24 | - | - | 1 |
   | 4 | 28×28 | bneck 5×5 | 4 | 40 | √ | √ | 2 |
   | 5 | 14×14 | bneck 5×5 | 6 | 40 | √ | √ | 1 |
   | 6 | 14×14 | bneck 5×5 | 6 | 40 | √ | √ | 1 |
   | 7 | 14×14 | bneck 5×5 | 3 | 48 | √ | √ | 1 |
   | 8 | 14×14 | bneck 5×5 | 3 | 48 | √ | √ | 1 |
   | 9 | 14×14 | bneck 5×5 | 6 | 96 | √ | √ | 2 |
   | 10 | 7×7 | bneck 5×5 | 6 | 96 | √ | √ | 1 |
   | 11 | 7×7 | bneck 5×5 | 6 | 96 | √ | √ | 1 |

### 项目结构

```
.
├── config/
│   └── config.py                # 配置文件
├── models/
│   └── mobilenetv3.py           # MobileNetV3 模型定义
├── utils/
│   └── dataset.py               # 数据集处理工具
├── train/
│   └── train.py                 # 训练脚本
├── eval/
│   └── evaluate.py              # 评估和预测脚本
├── data_processing/
│   ├── core/
│   │   ├── dataset_processor.py # 数据集处理核心
│   │   └── image_processor.py   # 图像处理核心
│   └── utils/
│       └── user_interface.py    # 用户界面工具
├── image_preprocessing/
│   ├── image_analyzer.py        # 图像分析工具
│   ├── image_converter.py       # 图像格式转换工具
│   ├── image_renamer.py         # 图像重命名工具
│   ├── image_validator.py       # 图像验证工具
│   └── preprocess_cli.py        # 预处理命令行界面
├── ewaste_recognition.py        # 主程序
├── requirements.txt             # 项目依赖
└── README.md                    # 项目文档
```

## 环境要求

- Python 3.7+
- MindSpore 2.2.0
- CUDA 支持（可选，用于 GPU 加速）
- 其他依赖见 requirements.txt

## 安装和使用

1. **环境配置**：
   ```bash
   pip install -r requirements.txt
   ```

2. **数据准备**：
   - 将数据集按以下结构组织：
     ```
     datasets/
     ├── camera_datasets/
     ├── chassis_datasets/
     ├── keyboard_datasets/
     └── ...
     
     testsets/
     ├── camera_testsets/
     ├── chassis_testsets/
     ├── keyboard_testsets/
     └── ...
     ```

3. **启动主程序**：
   ```bash
   python ewaste_recognition.py
   ```

4. **训练模型**：
   - 通过主程序界面选择 "Train New Model"
   - 或直接运行训练脚本：
     ```bash
     python train/train.py
     ```

5. **识别电子垃圾**：
   - 通过主程序界面选择 "Recognize E-Waste"
   - 可以识别单张图片或整个文件夹中的图片

## 新增功能

### MindRecord 数据集支持

系统现在支持使用 MindSpore 的 MindRecord 格式数据集，具有以下优势：

1. **更高效的数据加载**：
   - 优化的二进制存储格式
   - 减少数据加载时间
   - 提高训练效率

2. **灵活的数据选择**：
   - 用户可以在训练时选择使用原始图像文件夹或 MindRecord 数据集
   - 系统会自动列出所有可用的 MindRecord 文件供用户选择

3. **使用方法**：
   - 在训练界面选择 "Use preprocessed MindRecord dataset"
   - 从列表中选择要使用的训练集文件
   - 系统会自动匹配对应的验证集文件

### 图像预处理工具

项目新增了一套完整的图像预处理工具，位于 `image_preprocessing` 目录：

1. **图像重命名工具**：
   - 批量重命名图像文件为标准格式
   - 支持按类别处理
   - 提供交互式选择和确认机制

2. **图像验证工具**：
   - 检查并修复损坏或有问题的图像
   - 支持移动、删除或修复问题图像
   - 生成验证报告

3. **图像格式转换工具**：
   - 将图像转换为不同格式（JPG、PNG 等）
   - 支持调整图像大小
   - 批量处理多个类别

4. **图像分析工具**：
   - 生成数据集统计信息和可视化
   - 分析图像尺寸、格式、亮度等特性
   - 输出分析报告和图表

5. **预处理命令行界面**：
   - 统一的命令行界面访问所有预处理工具
   - 交互式操作指引
   - 使用方法：
     ```bash
     python image_preprocessing/preprocess_cli.py
     ```
   - 或指定特定工具：
     ```bash
     python image_preprocessing/preprocess_cli.py [rename|validate|convert|analyze]
     ```

## 并行处理支持

系统支持并行和顺序处理模式：

1. **并行处理**：
   - 利用多核 CPU 加速数据处理和训练
   - 用户可以指定使用的 CPU 核心数
   - 适合大多数现代计算机

2. **顺序处理**：
   - 单线程执行，适合资源受限的环境
   - 在并行处理出现问题时可作为备选方案

## 模型特点

1. **轻量级设计**：
   - 采用深度可分离卷积
   - 使用 1x1 卷积进行通道数调整
   - 模型参数量小，适合移动端部署

2. **优化的激活函数**：
   - 使用 Hard Swish 替代 ReLU
   - 降低计算量，提高性能

3. **注意力机制**：
   - 集成 SE 模块
   - 自适应特征重标定
   - 提高特征表达能力

4. **训练优化**：
   - 使用余弦退火学习率调度
   - 批归一化层用于稳定训练
   - 数据增强提高模型鲁棒性

## 数据预处理

1. **图像变换**：
   - 调整图像大小至 224×224
   - 随机裁剪和水平翻转
   - 标准化处理

2. **数据增强**：
   - 随机裁剪
   - 随机水平翻转
   - 标准化（均值和标准差）

## 训练策略

1. **优化器**：
   - Momentum 优化器
   - 权重衰减：0.0001
   - 动量因子：0.9

2. **学习率策略**：
   - 余弦退火调度
   - 初始学习率：0.001
   - 动态调整

3. **训练参数**：
   - 批次大小：32
   - 训练轮数：100
   - 使用模型检查点保存最佳模型

## 用户界面

系统提供了友好的英文用户界面：

1. **主菜单**：
   - Train New Model（训练新模型）
   - Recognize E-Waste（识别电子垃圾）
   - Exit Program（退出程序）

2. **训练选项**：
   - 选择数据集类型（原始图像文件夹或 MindRecord 数据集）
   - 选择处理模式（并行或顺序）
   - 指定 CPU 核心数
   - 确认训练设置

3. **识别选项**：
   - 识别单张图片
   - 识别整个文件夹中的图片
   - 显示识别结果和统计信息

## 注意事项

1. 确保数据集按照指定目录结构组织
2. GPU 训练需要正确配置 CUDA 环境
3. 可根据实际需求调整配置文件中的参数
4. 建议使用大规模数据集进行训练
5. 使用图像预处理工具可以提高数据集质量和训练效果

## 参考文献

1. Howard, A., et al. (2019). "Searching for MobileNetV3." arXiv preprint arXiv:1905.02244.
2. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018.
3. Hu, J., et al. (2018). "Squeeze-and-Excitation Networks." CVPR 2018. 