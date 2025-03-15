"""
Image processing module
"""
import cv2
import numpy as np
import os
from data_processing.config import IMAGE_SIZE

def process_image(image_path):
    """
    Process a single image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Processed image bytes, or None if processing fails
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"Error: File does not exist: {image_path}")
            return None
            
        # 检查文件大小是否为0
        if os.path.getsize(image_path) == 0:
            print(f"Error: File is empty: {image_path}")
            return None
        
        # 使用绝对路径
        abs_path = os.path.abspath(image_path)
        
        # 尝试读取图像
        img = cv2.imread(abs_path)
        
        if img is None:
            print(f"Error: Could not read image: {abs_path}")
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # 调整图像大小
        _, img_encode = cv2.imencode('.jpg', img)
        img_bytes = img_encode.tobytes()
        
        return img_bytes
    except UnicodeEncodeError as e:
        print(f"Unicode encoding error with file: {image_path}")
        print(f"Try using absolute path or renaming the file to ASCII characters only")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def process_image_for_parallel(args):
    """
    Function to process a single image for parallel processing
    
    Args:
        args: Tuple containing image path, file info, and label ID
    
    Returns:
        Dictionary with processing results or None if processing fails
    """
    image_path, file_info, label_id = args
    
    try:
        img_bytes = process_image(image_path)
        if img_bytes is None:
            return None
        
        # 确保标签是整数而不是元组
        if isinstance(label_id, tuple):
            print(f"Warning: label_id is a tuple {label_id} for {image_path}, converting to int")
            label_id = label_id[0] if label_id else 0
        elif not isinstance(label_id, int):
            print(f"Warning: label_id is not an int ({type(label_id)}) for {image_path}, converting to int")
            label_id = int(label_id) if label_id is not None else 0
        
        # 打印调试信息
        print(f"Processed image: {image_path}, label: {label_id} (type: {type(label_id)})")
            
        return {
            "image": img_bytes,
            "label": label_id,
            "filename": file_info
        }
    except Exception as e:
        print(f"Error in parallel processing for {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 