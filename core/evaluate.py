"""
评估脚本：用于模型评估和预测
"""
import os
import numpy as np
from PIL import Image
import mindspore as ms
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter

# 修改导入路径，适应新的文件结构
from core.mobilenetv3 import MobileNetV3
from core.config import Config

class Predictor:
    """预测类"""
    def __init__(self, checkpoint_path):
        """
        初始化预测器
        Args:
            checkpoint_path: 模型检查点路径
        """
        # 设置运行环境
        context.set_context(mode=context.GRAPH_MODE)
        ms.set_device(Config.device_target)
        
        # 创建模型
        self.network = MobileNetV3(
            num_classes=Config.num_classes,
            width_mult=1.0,
            mode='small'
        )
        
        # 加载模型参数
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(self.network, param_dict)
        self.network.set_train(False)
        
        # 图像预处理
        self.transform = [
            vision.Resize((Config.image_size, Config.image_size), Inter.BILINEAR),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            vision.HWC2CHW()
        ]
    
    def preprocess_image(self, image_path):
        """
        预处理图像
        Args:
            image_path: 图像路径
        Returns:
            处理后的图像数据
        """
        # 读取图像
        with Image.open(image_path) as img:
            img = img.convert('RGB')
        
        # 应用预处理
        for transform_op in self.transform:
            img = transform_op(img)
        
        # 添加批次维度
        img = np.expand_dims(img, 0)
        return ms.Tensor(img, ms.float32)
    
    def predict(self, image_path):
        """
        预测单张图像
        Args:
            image_path: 图像路径
        Returns:
            预测的类别和概率
        """
        # 预处理图像
        img = self.preprocess_image(image_path)
        
        # 执行预测
        output = self.network(img)
        probabilities = ms.ops.Softmax()(output)
        
        # 获取预测结果
        pred_class = int(output.argmax(1).asnumpy())
        pred_prob = float(probabilities[0][pred_class].asnumpy())
        
        return {
            'class_name': Config.class_names[pred_class],
            'class_id': pred_class,
            'probability': pred_prob
        }

def evaluate_model(checkpoint_path, test_dir):
    """
    评估模型性能
    Args:
        checkpoint_path: 模型检查点路径
        test_dir: 测试数据集目录
    """
    predictor = Predictor(checkpoint_path)
    total = 0
    correct = 0
    
    # 遍历测试集
    for class_name in Config.class_names:
        class_dir = os.path.join(test_dir, f"{class_name}_testsets")
        if not os.path.exists(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                result = predictor.predict(img_path)
                
                total += 1
                if result['class_name'] == class_name:
                    correct += 1
                    
    accuracy = correct / total if total > 0 else 0
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Correct Predictions: {correct}")
    print(f"Total Test Samples: {total}")

if __name__ == '__main__':
    # 使用示例
    checkpoint_path = Config.best_model_path
    if os.path.exists(checkpoint_path):
        # 评估模型
        evaluate_model(checkpoint_path, Config.val_data_dir)
        
        # 单张图片预测示例
        predictor = Predictor(checkpoint_path)
        test_image = "path_to_your_test_image.jpg"  # 替换为实际的测试图片路径
        if os.path.exists(test_image):
            result = predictor.predict(test_image)
            print(f"\nPrediction Results:")
            print(f"Class: {result['class_name']}")
            print(f"Confidence: {result['probability']:.4f}")
    else:
        print(f"Model checkpoint not found: {checkpoint_path}") 