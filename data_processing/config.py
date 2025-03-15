"""
数据处理配置文件
"""

# 支持的图像格式
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg')

# 数据集分割比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1  # 这个值实际上是 1 - TRAIN_RATIO - VAL_RATIO

# 图像处理参数
IMAGE_SIZE = 224  # 调整后的图像大小

# 输出目录
OUTPUT_DIR = './datasets'

# 数据集目录
DATASET_DIR = './datasets'

# 数据集文件夹后缀
DATASET_SUFFIX = '_datasets'
TESTSET_SUFFIX = '_testsets' 