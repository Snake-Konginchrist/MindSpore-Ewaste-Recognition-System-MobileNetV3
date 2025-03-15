"""
配置文件：包含模型训练和数据处理的相关参数
"""

class Config:
    # 数据集配置
    num_classes = 11  # 电子垃圾类别数量
    image_size = 224  # 输入图像大小
    batch_size = 32
    num_workers = 4
    
    # 训练配置
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 0.0001
    momentum = 0.9
    
    # 模型配置
    dropout = 0.2
    
    # 数据集路径
    train_data_dir = "./datasets"
    val_data_dir = "./testsets"
    
    # 模型保存路径
    model_save_dir = "./checkpoints"
    best_model_path = "./checkpoints/best_model.ckpt"
    
    # 设备配置
    device_target = "CPU"  # 可选 "GPU", "CPU", "Ascend"
    
    # 数据增强配置
    random_crop = True
    random_horizontal_flip = True
    normalize = True
    
    # 类别映射
    class_names = [
        "camera", "chassis", "keyboard", "laptop",
        "monitor", "mouse", "radio", "router",
        "smartphone", "telephone", "others"
    ] 