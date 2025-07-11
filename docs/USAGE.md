# 使用指南

## 环境配置

### 系统要求
- Python 3.7+
- MindSpore 2.2.0+
- PyQt5 (用于图形界面)
- CUDA 支持（可选，用于 GPU 加速）

### 安装依赖
```bash
pip install -r requirements.txt
```

## 数据准备

### 数据集下载
1. 下载数据集：[百度网盘链接](https://pan.baidu.com/s/14GTB3pxmQ-c0dcADk1cH3A) (提取码: fasf)
2. 解压到项目根目录

### 目录结构
```
datasets/
├── camera_datasets/
├── chassis_datasets/
├── keyboard_datasets/
├── laptop_datasets/
├── monitor_datasets/
├── mouse_datasets/
├── radio_datasets/
├── router_datasets/
├── smartphone_datasets/
├── telephone_datasets/
└── others_datasets/

testsets/
├── camera_testsets/
├── chassis_testsets/
├── keyboard_testsets/
├── laptop_testsets/
├── monitor_testsets/
├── mouse_testsets/
├── radio_testsets/
├── router_testsets/
├── smartphone_testsets/
├── telephone_testsets/
└── others_testsets/
```

## 使用方式

### 命令行界面

#### 启动主程序
```bash
python ewaste_recognition.py
```

#### 启动图形界面
```bash
python ewaste_recognition.py --gui
```

#### 主菜单选项
1. **Train New Model（训练新模型）**
   - 选择数据集类型（原始图像文件夹或 MindRecord 数据集）
   - 选择处理模式（并行或顺序）
   - 指定 CPU 核心数
   - 配置早停参数
   - 确认训练设置

2. **Recognize E-Waste（识别电子垃圾）**
   - 识别单张图片
   - 识别整个文件夹中的图片
   - 显示识别结果和统计信息

3. **Launch Graphical Interface（启动图形界面）**
   - 启动基于 PyQt5 的图形界面

4. **Exit Program（退出程序）**

### 图形用户界面

#### 功能特点
- **拖放识别**：直接将图片拖放到界面中进行识别
- **实时反馈**：即时显示识别结果和置信度
- **用户友好设计**：清晰的视觉提示和操作反馈

#### 界面组件
- 拖放区域用于上传图片
- 识别结果实时显示
- 图像信息展示
- 操作按钮（选择图片、识别、清除）
- 状态栏提供操作反馈

## 训练配置

### 训练选项
- **数据集类型选择**：
  - 原始图像文件夹
  - MindRecord 数据集

- **处理模式**：
  - 并行处理（推荐）
  - 顺序处理

- **CPU 核心数**：
  - 自动检测全部核心
  - 手动指定核心数

- **早停机制**：
  - 启用/禁用早停
  - 设置耐心值（默认10）
  - 监控指标选择

### 训练参数
- 批次大小：32
- 训练轮数：100（支持早停）
- 学习率：0.001
- 优化器：Momentum

## 识别功能

### 单张图片识别
1. 选择图片文件
2. 系统自动进行预处理
3. 显示识别结果和置信度

### 批量识别
1. 选择包含图片的文件夹
2. 系统自动处理所有图片
3. 生成识别结果报告

### 支持格式
- PNG
- JPG/JPEG
- BMP
- 其他常见图像格式

## 高级功能

### MindRecord 数据集支持
- 更高效的数据加载
- 优化的二进制存储格式
- 减少数据加载时间

### 并行处理优化
- 多核 CPU 加速
- 多线程数据加载
- 自动混合精度训练

### 早停机制
- 智能训练终止
- 自动监控验证集性能
- 避免过拟合

## 注意事项

1. **环境配置**：
   - 确保 MindSpore 正确安装
   - GPU 训练需要正确配置 CUDA 环境
   - 图形界面需要安装 PyQt5 库

2. **数据准备**：
   - 确保数据集按照指定目录结构组织
   - 建议使用图像预处理工具提高数据质量

3. **训练建议**：
   - 使用大规模数据集进行训练
   - 启用并行处理提高效率
   - 使用早停机制避免过拟合

4. **性能优化**：
   - 根据硬件配置调整 CPU 核心数
   - 使用 MindRecord 格式提高数据加载效率
   - 启用混合精度训练减少内存占用

## 故障排除

### 常见问题
1. **模块导入错误**：检查依赖是否正确安装
2. **CUDA 错误**：确认 CUDA 环境配置
3. **内存不足**：减少批次大小或启用混合精度
4. **数据加载慢**：使用 MindRecord 格式或并行处理

### 日志查看
- 训练日志保存在控制台输出
- 错误信息会显示详细的堆栈跟踪
- 建议保存日志文件以便调试 