# 电子垃圾分类数据集上传到魔搭平台指南

本文档提供了使用`upload_dataset_to_modelscope.py`脚本将电子垃圾分类数据集上传到魔搭(ModelScope)平台的详细指南。

## 前提条件

在开始之前，请确保您已经：

1. 安装了Python 3.7或更高版本
2. 注册了魔搭平台账号 (https://www.modelscope.cn)
3. 获取了魔搭平台的访问令牌(Access Token)
4. 安装了必要的依赖库

### 安装依赖

```bash
pip install modelscope tqdm pillow pandas
```

## 数据集要求

您的电子垃圾分类数据集应当按照以下目录结构组织：

```
datasets/
├── camera_datasets/      # 相机类别的图片
├── chassis_datasets/     # 机箱类别的图片
├── keyboard_datasets/    # 键盘类别的图片
└── ...                   # 其他类别
```

每个类别文件夹的命名格式应为`{类别名称}_datasets`，例如`camera_datasets`。

## 使用方法

### 1. 获取魔搭平台访问令牌

1. 登录魔搭平台 (https://www.modelscope.cn)
2. 点击右上角的用户头像，选择"设置"
3. 在左侧菜单中选择"访问令牌"
4. 点击"创建新令牌"，输入令牌名称并选择权限
5. 创建后复制并保存令牌字符串

### 2. 运行上传脚本

基本用法：

```bash
python upload_to_modelscope.py --dataset_dir <数据集目录路径>
```

完整参数说明：

```bash
python upload_to_modelscope.py \
    --dataset_dir <数据集目录路径> \
    --output_dir <临时处理目录> \
    --dataset_name <数据集名称> \
    --description <数据集描述>
```

参数说明：
- `--dataset_dir`：必需，原始数据集目录的路径
- `--output_dir`：可选，临时处理目录的路径，默认为`./modelscope_dataset`
- `--dataset_name`：可选，数据集名称，默认为`ewaste_classification`
- `--description`：可选，数据集描述，默认为`电子垃圾图像分类数据集，包含多种电子废弃物类别`

### 3. 输入访问令牌

运行脚本后，如果您没有通过环境变量设置访问令牌，系统会提示您输入：

```
请输入您的魔搭平台访问令牌 (Access Token): 
```

输入您在第1步获取的访问令牌。

或者，您可以通过设置环境变量来避免每次输入令牌：

```bash
# Linux/Mac
export MODELSCOPE_API_TOKEN=<您的访问令牌>

# Windows (CMD)
set MODELSCOPE_API_TOKEN=<您的访问令牌>

# Windows (PowerShell)
$env:MODELSCOPE_API_TOKEN=<您的访问令牌>
```

### 4. 上传过程

脚本会执行以下步骤：

1. 扫描数据集目录，识别所有类别
2. 将数据集划分为训练集(80%)和验证集(20%)
3. 创建必要的CSV文件和元数据文件
4. 登录魔搭平台
5. 创建新的数据集
6. 上传所有文件到魔搭平台

上传完成后，您将看到类似以下的输出：

```
数据集上传成功! 数据集ID: <用户名>/<数据集名称>
您可以在魔搭平台查看您的数据集: https://www.modelscope.cn/datasets/<用户名>/<数据集名称>/summary
```

## 数据集格式说明

上传到魔搭平台的数据集包含以下文件：

1. `train.csv`：训练集数据，包含图片路径、标签ID和类别名称
2. `dev.csv`：验证集数据，格式同上
3. `<dataset_name>.json`：数据集元数据，包含类别信息等
4. `dataset_infos.json`：魔搭平台所需的数据集信息文件
5. 所有图片文件，按照`train_<类别>_<原文件名>`和`dev_<类别>_<原文件名>`命名

## 注意事项

1. 上传大型数据集可能需要较长时间，请确保网络连接稳定
2. 默认情况下，数据集会被设置为私有，您可以在魔搭平台上修改其可见性
3. 如果上传过程中断，您可能需要在魔搭平台上删除未完成的数据集后重新上传
4. 确保您的数据集符合魔搭平台的使用条款和内容政策

## 常见问题

### Q: 上传失败并显示认证错误
A: 请确认您的访问令牌是否有效，或者尝试重新生成一个新的令牌。

### Q: 上传过程中断
A: 检查网络连接并重新运行脚本。如果问题持续存在，尝试减小数据集大小或使用更稳定的网络连接。

### Q: 找不到类别
A: 确保您的类别文件夹命名格式正确，应为`{类别名称}_datasets`。

### Q: 上传后无法在魔搭平台找到数据集
A: 登录魔搭平台，进入"我的数据集"页面查看。如果仍然找不到，检查上传日志中的数据集ID是否正确。

## 后续步骤

成功上传数据集后，您可以：

1. 在魔搭平台上编辑数据集信息和标签
2. 设置数据集的可见性（私有/公开）
3. 使用该数据集训练模型
4. 与其他用户分享您的数据集

## 参考资料

- [魔搭平台官方文档](https://www.modelscope.cn/docs)
- [ModelScope Python SDK文档](https://github.com/modelscope/modelscope)
- [数据集格式规范](https://www.modelscope.cn/docs/dataset_create) 