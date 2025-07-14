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

## 快速开始

### 环境要求

- Python 3.11
- MindSpore 2.6.0
- PyQt5 (用于图形界面)
- CUDA 支持（可选，用于 GPU 加速）

### 安装依赖

#### 方式一：使用 pip（推荐）
```bash
pip install -r requirements.txt
```

#### 方式二：使用 UV 包管理器（更快）

**安装 UV：**

**macOS/Linux：**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows：**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**安装项目依赖：**
```bash
uv sync
```

> **UV 优势**：更快的依赖解析和安装速度，自动管理虚拟环境，支持依赖锁定。
> 
> 详细使用说明请参考：[UV 使用指南](UV_USAGE.md)

### 数据准备

1. 下载数据集：[百度网盘链接](https://pan.baidu.com/s/14GTB3pxmQ-c0dcADk1cH3A) (提取码: fasf)
2. 解压到项目根目录，保持以下目录结构：
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

### 使用方式

#### 命令行界面

##### 使用 pip 安装的依赖
```bash
# 启动主程序
python ewaste_recognition.py

# 启动图形界面
python ewaste_recognition.py --gui
```

##### 使用 UV 安装的依赖
```bash
# 启动主程序
uv run ewaste_recognition.py

# 启动图形界面
uv run ewaste_recognition.py --gui
```

#### 图形界面功能
- 拖放图片进行识别
- 实时显示识别结果和置信度
- 支持批量处理

## 项目结构

```
.
├── core/                        # 核心功能模块
│   ├── config.py                # 核心配置
│   ├── dataset.py               # 数据集处理
│   ├── evaluate.py              # 评估和预测
│   ├── mobilenetv3.py           # MobileNetV3 模型定义
│   └── train.py                 # 训练脚本
├── ui/                          # 图形用户界面
│   └── ewaste_ui.py             # 图形界面实现
├── data_processing/             # 数据处理模块
├── image_preprocessing/         # 图像预处理工具
├── ewaste_recognition.py        # 主程序
├── requirements.txt             # 项目依赖
└── README.md                    # 项目文档
```

## 主要功能

- **模型训练**：支持自定义数据集训练
- **图像识别**：单张图片或批量识别
- **图形界面**：基于 PyQt5 的现代化界面
- **并行处理**：多核 CPU 加速支持
- **早停机制**：智能训练终止
- **数据预处理**：完整的图像处理工具链

## 技术特点

- **轻量级设计**：采用 MobileNetV3-Small 架构
- **高效推理**：深度可分离卷积优化
- **注意力机制**：SE 模块增强特征表达
- **混合精度**：自动混合精度训练支持

## 详细文档

- [技术架构详解](docs/ARCHITECTURE.md) - MobileNetV3 网络结构和技术细节
- [使用指南](docs/USAGE.md) - 详细的使用说明和配置指南
- [数据集说明](docs/DATASET.md) - 数据集信息和使用方法
- [API 文档](docs/API.md) - 核心模块 API 参考
- [预处理工具](docs/PREPROCESSING.md) - 图像预处理工具使用说明

## 许可证

本项目采用 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0) 进行许可。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 交流群
- **彩旗开源交流QQ群**: 1022820973
- 欢迎加入群聊，与开发者交流技术问题和使用心得

## 参考文献

1. Howard, A., et al. (2019). "Searching for MobileNetV3." arXiv preprint arXiv:1905.02244.
2. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018.
3. Hu, J., et al. (2018). "Squeeze-and-Excitation Networks." CVPR 2018. 

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Snake-Konginchrist/MindSpore-Ewaste-Recognition-System-MobileNetV3&type=Date)](https://www.star-history.com/#Snake-Konginchrist/MindSpore-Ewaste-Recognition-System-MobileNetV3&Date)