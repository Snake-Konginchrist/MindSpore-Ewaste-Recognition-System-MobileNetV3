#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电子垃圾分类数据集上传到魔搭ModelScope平台的脚本
"""

import os
import json
import argparse
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
import csv
from modelscope.hub.api import HubApi
from modelscope.utils.constant import DownloadMode
import time
import hashlib

class EWasteDatasetUploader:
    """电子垃圾数据集上传到魔搭平台的工具类"""
    
    def __init__(self, dataset_dir, output_dir, dataset_name, dataset_description):
        """
        初始化上传工具
        
        Args:
            dataset_dir: 原始数据集目录
            output_dir: 临时处理目录
            dataset_name: 数据集名称
            dataset_description: 数据集描述
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.dataset_description = dataset_description
        self.class_names = []
        self.api = HubApi()
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def prepare_dataset(self):
        """准备数据集，将原始数据转换为魔搭平台支持的格式"""
        print("正在准备数据集...")
        
        # 创建train.csv和dev.csv文件
        train_csv_path = os.path.join(self.output_dir, "train.csv")
        dev_csv_path = os.path.join(self.output_dir, "dev.csv")
        
        # 获取所有类别
        self._discover_categories()
        
        # 处理训练集和验证集
        train_data = []
        dev_data = []
        
        # 遍历所有类别
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.dataset_dir, f"{class_name}_datasets")
            if not os.path.exists(class_dir):
                continue
                
            # 获取该类别下的所有图片
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 划分训练集和验证集 (80% 训练, 20% 验证)
            split_idx = int(len(image_files) * 0.8)
            train_files = image_files[:split_idx]
            dev_files = image_files[split_idx:]
            
            # 处理训练集
            for img_file in train_files:
                img_path = os.path.join(class_dir, img_file)
                # 复制图片到输出目录
                new_img_path = os.path.join(self.output_dir, f"train_{class_name}_{img_file}")
                shutil.copy(img_path, new_img_path)
                
                # 添加到CSV数据
                train_data.append({
                    "image_path": os.path.basename(new_img_path),
                    "label": class_idx,
                    "class_name": class_name
                })
            
            # 处理验证集
            for img_file in dev_files:
                img_path = os.path.join(class_dir, img_file)
                # 复制图片到输出目录
                new_img_path = os.path.join(self.output_dir, f"dev_{class_name}_{img_file}")
                shutil.copy(img_path, new_img_path)
                
                # 添加到CSV数据
                dev_data.append({
                    "image_path": os.path.basename(new_img_path),
                    "label": class_idx,
                    "class_name": class_name
                })
        
        # 写入CSV文件
        self._write_csv(train_csv_path, train_data)
        self._write_csv(dev_csv_path, dev_data)
        
        # 创建数据集信息文件
        self._create_dataset_info()
        
        print(f"数据集准备完成，共 {len(train_data)} 个训练样本，{len(dev_data)} 个验证样本")
        return train_csv_path, dev_csv_path
    
    def _discover_categories(self):
        """发现数据集中的所有类别"""
        categories = []
        
        # 遍历数据集目录
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            
            # 检查是否是目录且以_datasets结尾
            if os.path.isdir(item_path) and item.endswith("_datasets"):
                # 提取类别名称
                category = item[:-9]  # 去掉_datasets后缀
                categories.append(category)
        
        # 排序以确保一致性
        categories.sort()
        self.class_names = categories
        
        print(f"发现以下类别: {', '.join(self.class_names)}")
    
    def _write_csv(self, csv_path, data):
        """将数据写入CSV文件"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "label", "class_name"])
            writer.writeheader()
            writer.writerows(data)
    
    def _create_dataset_info(self):
        """创建数据集信息文件"""
        dataset_info = {
            "dataset_name": self.dataset_name,
            "description": self.dataset_description,
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        # 创建与数据集同名的JSON文件
        info_path = os.path.join(self.output_dir, f"{self.dataset_name}.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        # 创建dataset_infos.json文件
        dataset_infos = {
            "default": {
                "features": {
                    "image_path": {"_type": "Value"},
                    "label": {"_type": "Value"},
                    "class_name": {"_type": "Value"}
                },
                "splits": {
                    "train": {
                        "name": "train",
                        "dataset_name": self.dataset_name
                    },
                    "validation": {
                        "name": "validation",
                        "dataset_name": self.dataset_name
                    }
                }
            }
        }
        
        infos_path = os.path.join(self.output_dir, "dataset_infos.json")
        with open(infos_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_infos, f, ensure_ascii=False, indent=2)
    
    def upload_to_modelscope(self):
        """上传数据集到魔搭平台"""
        print("正在上传数据集到魔搭平台...")
        
        try:
            # 准备数据集
            self.prepare_dataset()
            
            # 登录魔搭平台
            self._login_modelscope()
            
            # 创建数据集
            dataset_id = self._create_modelscope_dataset()
            
            # 上传数据文件
            self._upload_files(dataset_id)
            
            print(f"数据集上传成功! 数据集ID: {dataset_id}")
            print(f"您可以在魔搭平台查看您的数据集: https://www.modelscope.cn/datasets/{dataset_id}/summary")
            
            return dataset_id
        except Exception as e:
            print(f"上传数据集时发生错误: {str(e)}")
            raise
    
    def _login_modelscope(self):
        """登录魔搭平台"""
        # 检查环境变量中是否有访问令牌
        access_token = os.environ.get('MODELSCOPE_API_TOKEN')
        
        if not access_token:
            # 如果环境变量中没有，则提示用户输入
            access_token = input("请输入您的魔搭平台访问令牌 (Access Token): ")
        
        # 设置访问令牌
        self.api.login(access_token)
        print("魔搭平台登录成功")
    
    def _create_modelscope_dataset(self):
        """在魔搭平台创建数据集"""
        # 获取当前用户名
        user_info = self.api.get_user_info()
        username = user_info.get('name', 'anonymous')
        
        # 创建数据集ID
        dataset_id = f"{username}/{self.dataset_name}"
        
        # 创建数据集
        response = self.api.create_dataset(
            dataset_name=self.dataset_name,
            dataset_type='datasets',
            visibility='private',  # 可以设置为'public'使其公开
            description=self.dataset_description,
            dataset_id=dataset_id
        )
        
        print(f"数据集创建成功: {dataset_id}")
        return dataset_id
    
    def _upload_files(self, dataset_id):
        """上传数据文件到魔搭平台"""
        # 获取输出目录中的所有文件
        files = os.listdir(self.output_dir)
        
        # 上传每个文件
        for file in tqdm(files, desc="上传文件"):
            file_path = os.path.join(self.output_dir, file)
            
            # 上传文件
            self.api.upload_dataset_file(
                dataset_id=dataset_id,
                file_path=file_path,
                target_path=file
            )
        
        print(f"共上传 {len(files)} 个文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='上传电子垃圾分类数据集到魔搭平台')
    parser.add_argument('--dataset_dir', type=str, required=True, help='原始数据集目录')
    parser.add_argument('--output_dir', type=str, default='./modelscope_dataset', help='临时处理目录')
    parser.add_argument('--dataset_name', type=str, default='ewaste_classification', help='数据集名称')
    parser.add_argument('--description', type=str, default='电子垃圾图像分类数据集，包含多种电子废弃物类别', help='数据集描述')
    
    args = parser.parse_args()
    
    # 创建上传工具并上传数据集
    uploader = EWasteDatasetUploader(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_description=args.description
    )
    
    # 上传数据集
    dataset_id = uploader.upload_to_modelscope()
    
    print(f"数据集上传完成! 数据集ID: {dataset_id}")


if __name__ == "__main__":
    main() 