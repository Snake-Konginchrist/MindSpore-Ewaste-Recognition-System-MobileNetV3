"""
图像统计和可视化工具

此脚本用于分析数据集中的图像，生成统计信息和可视化结果，
帮助了解数据集的特性和分布情况。
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, Counter
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从数据处理模块导入配置
from data_processing.config import DATASET_DIR, DATASET_SUFFIX, SUPPORTED_FORMATS
from data_processing.core.dataset_processor import discover_categories


def analyze_image(image_path):
    """
    分析单个图像的特性
    
    参数:
        image_path: 图像文件路径
        
    返回:
        包含图像特性的字典或None（如果分析失败）
    """
    try:
        # 使用PIL读取图像
        with Image.open(image_path) as pil_img:
            width, height = pil_img.size
            format_type = pil_img.format
            mode = pil_img.mode
            
            # 计算文件大小（KB）
            file_size = os.path.getsize(image_path) / 1024
        
        # 使用OpenCV读取图像以进行更多分析
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            return None
        
        # 转换为RGB进行分析
        if len(cv_img.shape) == 3:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # 计算平均RGB值
            avg_r = np.mean(rgb_img[:, :, 0])
            avg_g = np.mean(rgb_img[:, :, 1])
            avg_b = np.mean(rgb_img[:, :, 2])
            # 计算亮度
            gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_img)
            # 计算对比度
            contrast = np.std(gray_img)
        else:
            # 灰度图像
            avg_r = avg_g = avg_b = np.mean(cv_img)
            brightness = avg_r
            contrast = np.std(cv_img)
        
        # 计算模糊度（使用拉普拉斯算子）
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        blur_score = np.var(laplacian)
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'format': format_type,
            'mode': mode,
            'file_size': file_size,
            'avg_r': avg_r,
            'avg_g': avg_g,
            'avg_b': avg_b,
            'brightness': brightness,
            'contrast': contrast,
            'blur_score': blur_score
        }
    except Exception as e:
        print(f"Error analyzing {image_path}: {str(e)}")
        return None


def analyze_category(category, sample_size=None, verbose=False):
    """
    分析特定类别的所有图像
    
    参数:
        category: 类别名称
        sample_size: 要分析的样本数量，如果为None则分析所有图像
        verbose: 是否显示详细信息
        
    返回:
        包含类别统计信息的字典
    """
    # 构建类别目录路径
    category_dir = os.path.join(DATASET_DIR, f"{category}{DATASET_SUFFIX}")
    
    if not os.path.exists(category_dir):
        print(f"Error: Directory for category '{category}' not found at {category_dir}")
        return None
    
    # 获取目录中的所有图像文件
    image_files = []
    for filename in os.listdir(category_dir):
        if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in {category_dir}")
        return None
    
    # 如果指定了样本大小，随机选择样本
    if sample_size and sample_size < len(image_files):
        import random
        image_files = random.sample(image_files, sample_size)
    
    print(f"Analyzing {len(image_files)} images in category '{category}'")
    
    # 初始化统计数据
    stats = {
        'count': len(image_files),
        'formats': Counter(),
        'modes': Counter(),
        'widths': [],
        'heights': [],
        'aspect_ratios': [],
        'file_sizes': [],
        'brightness': [],
        'contrast': [],
        'blur_scores': [],
        'avg_colors': [],
        'invalid_images': []
    }
    
    # 分析每个文件
    for filename in tqdm(image_files, desc=f"Analyzing {category}"):
        image_path = os.path.join(category_dir, filename)
        result = analyze_image(image_path)
        
        if result:
            # 更新统计数据
            stats['formats'][result['format']] += 1
            stats['modes'][result['mode']] += 1
            stats['widths'].append(result['width'])
            stats['heights'].append(result['height'])
            stats['aspect_ratios'].append(result['aspect_ratio'])
            stats['file_sizes'].append(result['file_size'])
            stats['brightness'].append(result['brightness'])
            stats['contrast'].append(result['contrast'])
            stats['blur_scores'].append(result['blur_score'])
            stats['avg_colors'].append((result['avg_r'], result['avg_g'], result['avg_b']))
            
            if verbose:
                print(f"  {filename}: {result['width']}x{result['height']}, {result['format']}, {result['file_size']:.1f}KB")
        else:
            stats['invalid_images'].append(filename)
            if verbose:
                print(f"  Invalid image: {filename}")
    
    # 计算汇总统计数据
    if stats['widths']:
        stats['avg_width'] = np.mean(stats['widths'])
        stats['avg_height'] = np.mean(stats['heights'])
        stats['avg_aspect_ratio'] = np.mean(stats['aspect_ratios'])
        stats['avg_file_size'] = np.mean(stats['file_sizes'])
        stats['avg_brightness'] = np.mean(stats['brightness'])
        stats['avg_contrast'] = np.mean(stats['contrast'])
        stats['avg_blur_score'] = np.mean(stats['blur_scores'])
        
        # 计算平均颜色
        avg_colors = np.array(stats['avg_colors'])
        stats['avg_color'] = np.mean(avg_colors, axis=0)
    
    return stats


def batch_analyze_images(selected_categories=None, sample_size=None, verbose=False, output_dir='./analysis_results'):
    """
    批量分析多个类别的图像
    
    参数:
        selected_categories: 要处理的类别列表，如果为None则处理所有类别
        sample_size: 每个类别要分析的样本数量，如果为None则分析所有图像
        verbose: 是否显示详细信息
        output_dir: 输出目录，用于保存可视化结果
    """
    # 发现可用类别
    categories, _ = discover_categories()
    
    if not categories:
        print("No categories found in the dataset directory.")
        return
    
    print(f"Available categories: {', '.join(categories)}")
    
    # 如果未指定类别，使用所有类别
    if selected_categories is None:
        selected_categories = categories
    else:
        # 验证选定的类别
        valid_categories = [cat for cat in selected_categories if cat in categories]
        if not valid_categories:
            print("None of the selected categories exist in the dataset.")
            return
        selected_categories = valid_categories
    
    print(f"Selected categories: {', '.join(selected_categories)}")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 初始化汇总统计数据
    all_stats = {}
    category_counts = []
    category_names = []
    avg_widths = []
    avg_heights = []
    avg_file_sizes = []
    avg_brightness = []
    
    # 处理每个类别
    for category in selected_categories:
        stats = analyze_category(category, sample_size, verbose)
        if stats:
            all_stats[category] = stats
            
            # 收集汇总数据
            category_counts.append(stats['count'])
            category_names.append(category)
            if 'avg_width' in stats:
                avg_widths.append(stats['avg_width'])
                avg_heights.append(stats['avg_height'])
                avg_file_sizes.append(stats['avg_file_size'])
                avg_brightness.append(stats['avg_brightness'])
    
    # 打印汇总统计信息
    print("\nDataset Analysis Summary:")
    print(f"  Total categories: {len(all_stats)}")
    total_images = sum(stats['count'] for stats in all_stats.values())
    print(f"  Total images: {total_images}")
    
    # 生成可视化结果
    if output_dir:
        # 1. 类别分布饼图
        plt.figure(figsize=(10, 6))
        plt.pie(category_counts, labels=category_names, autopct='%1.1f%%')
        plt.title('Image Distribution by Category')
        plt.savefig(os.path.join(output_dir, 'category_distribution.png'))
        plt.close()
        
        # 2. 平均图像尺寸条形图
        if avg_widths:
            plt.figure(figsize=(12, 6))
            x = np.arange(len(category_names))
            width = 0.35
            plt.bar(x - width/2, avg_widths, width, label='Width')
            plt.bar(x + width/2, avg_heights, width, label='Height')
            plt.xlabel('Category')
            plt.ylabel('Pixels')
            plt.title('Average Image Dimensions by Category')
            plt.xticks(x, category_names, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'avg_dimensions.png'))
            plt.close()
            
            # 3. 平均文件大小条形图
            plt.figure(figsize=(12, 6))
            plt.bar(category_names, avg_file_sizes)
            plt.xlabel('Category')
            plt.ylabel('File Size (KB)')
            plt.title('Average File Size by Category')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'avg_file_size.png'))
            plt.close()
            
            # 4. 平均亮度条形图
            plt.figure(figsize=(12, 6))
            plt.bar(category_names, avg_brightness)
            plt.xlabel('Category')
            plt.ylabel('Brightness (0-255)')
            plt.title('Average Image Brightness by Category')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'avg_brightness.png'))
            plt.close()
        
        print(f"Visualization results saved to {output_dir}")
    
    # 返回所有统计数据
    return all_stats


def select_categories_interactive():
    """
    交互式选择要处理的类别
    
    返回:
        选定的类别列表或None表示所有类别
    """
    # 发现可用类别
    categories, _ = discover_categories()
    
    if not categories:
        print("No categories found in the dataset directory.")
        return None
    
    print("\nAvailable categories:")
    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")
    
    print("\nSelect categories to analyze (comma-separated numbers, or 'all' for all categories):")
    selection = input("> ").strip().lower()
    
    if selection == 'all' or selection == '':
        print("Processing all categories")
        return None  # None表示所有类别
    
    try:
        # 解析用户输入
        indices = [int(idx.strip()) for idx in selection.split(',') if idx.strip()]
        selected = [categories[idx-1] for idx in indices if 1 <= idx <= len(categories)]
        
        if not selected:
            print("No valid categories selected, processing all categories")
            return None
        
        print(f"Selected categories: {', '.join(selected)}")
        return selected
    except (ValueError, IndexError):
        print("Invalid input, processing all categories")
        return None


def main():
    """
    图像分析工具的主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze images in the dataset")
    parser.add_argument("--categories", nargs="+", help="Categories to process (default: all)")
    parser.add_argument("--sample-size", type=int, help="Number of images to sample from each category")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    parser.add_argument("--output-dir", default="./analysis_results", help="Directory to save visualization results")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode for selecting categories")
    
    args = parser.parse_args()
    
    print("Image Analysis Tool")
    print("==================")
    
    # 获取要处理的类别
    selected_categories = None
    if args.interactive:
        selected_categories = select_categories_interactive()
    elif args.categories:
        selected_categories = args.categories
    
    # 执行分析
    batch_analyze_images(
        selected_categories=selected_categories,
        sample_size=args.sample_size,
        verbose=args.verbose,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main() 