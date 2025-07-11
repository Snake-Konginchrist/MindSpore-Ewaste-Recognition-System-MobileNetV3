# 数据集说明

## 电子垃圾智能分类数据集 (E-Waste Classification Dataset)

本项目使用专为电子废弃物自动识别与分类设计的图像数据集，具有以下特点：

### 数据集组成

- **11个类别的电子垃圾图像**
- 每类包含多角度、多光照条件下的图像
- 经过精心标注和预处理

### 技术细节

- **类别数量**: 11
- **类别列表**: camera, chassis, keyboard, laptop, monitor, mouse, radio, router, smartphone, telephone, others
- **图像格式**: JPG/PNG
- **预处理后分辨率**: 224×224

### 应用场景

- 智能回收站自动分类
- 回收工厂自动化
- 环保教育
- 循环经济研究
- 资源回收优化

### 数据集分割

- **训练集**: 70%
- **验证集**: 15%
- **测试集**: 15%

## 获取数据集

### 下载链接
- **百度网盘链接**: [https://pan.baidu.com/s/14GTB3pxmQ-c0dcADk1cH3A](https://pan.baidu.com/s/14GTB3pxmQ-c0dcADk1cH3A)
- **提取码**: fasf

### 目录结构
下载后请将数据集解压到项目根目录，保持以下目录结构：

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

## 数据集开源计划

### 即将上线平台
- **魔搭社区 (ModelScope)**: [即将上线]
- **Hugging Face**: [即将上线]

### 更新说明
- 上线后将在此更新直接下载链接
- 欢迎关注项目更新获取最新数据集资源

## 数据预处理

### 图像变换
- 调整图像大小至 224×224
- 随机裁剪和水平翻转
- 标准化处理

### 数据增强
- 随机裁剪
- 随机水平翻转
- 标准化（均值和标准差）

## MindRecord 数据集支持

### 优势
- **更高效的数据加载**：优化的二进制存储格式
- **减少数据加载时间**：提高训练效率
- **灵活的数据选择**：用户可以选择使用原始图像文件夹或 MindRecord 数据集

### 使用方法
- 在训练界面选择 "Use preprocessed MindRecord dataset"
- 从列表中选择要使用的训练集文件
- 系统会自动匹配对应的验证集文件

## 数据质量要求

### 图像要求
- 分辨率：建议不低于 224×224
- 格式：JPG、PNG、BMP 等常见格式
- 质量：清晰、无严重损坏

### 标注要求
- 每个类别至少包含 100 张图像
- 图像角度多样化
- 光照条件多样化
- 背景环境多样化

## 自定义数据集

### 准备步骤
1. 按类别创建文件夹
2. 将图像放入对应类别文件夹
3. 确保图像格式正确
4. 使用预处理工具优化数据

### 推荐工具
- 使用项目提供的图像预处理工具
- 批量重命名和格式转换
- 图像质量验证和修复

## 数据集统计

### 类别分布
- 每个类别图像数量应相对均衡
- 建议每类至少 100 张训练图像
- 验证集和测试集按比例分配

### 图像特征
- 平均分辨率：224×224
- 主要格式：JPG、PNG
- 色彩空间：RGB

## 注意事项

1. **数据完整性**：确保所有图像文件完整可读
2. **类别平衡**：避免某些类别数据过多或过少
3. **数据质量**：使用预处理工具检查图像质量
4. **备份保存**：建议保留原始数据备份 