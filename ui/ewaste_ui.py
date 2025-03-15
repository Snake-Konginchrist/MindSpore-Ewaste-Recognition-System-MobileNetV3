"""
电子垃圾识别系统图形界面

此模块提供了一个用户友好的图形界面，用于电子垃圾识别。
用户可以通过拖拽或选择文件的方式上传图片进行识别。
"""

import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                            QFrame, QSplitter, QSizePolicy, QMessageBox)
from PyQt5.QtCore import Qt, QMimeData, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QDrag, QDragEnterEvent, QDropEvent, QPalette, QColor

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入项目模块
from core.evaluate import Predictor
from core.config import Config
from PIL import Image
import numpy as np


class DropArea(QLabel):
    """
    可拖拽区域组件
    """
    # 添加自定义信号
    image_loaded_signal = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drag and drop image here\nor click to select")
        self.setWordWrap(True)
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 设置样式
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaaaaa;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 10px;
                font-size: 16px;
            }
            QLabel:hover {
                border-color: #3498db;
                background-color: #e8f4fc;
            }
        """)
        
        self.setScaledContents(False)
        self.image_path = None
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """处理拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #3498db;
                    border-radius: 5px;
                    background-color: #e8f4fc;
                    padding: 10px;
                    font-size: 16px;
                }
            """)
    
    def dragLeaveEvent(self, event):
        """处理拖拽离开事件"""
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaaaaa;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 10px;
                font-size: 16px;
            }
            QLabel:hover {
                border-color: #3498db;
                background-color: #e8f4fc;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """处理拖拽释放事件"""
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            
            # 获取文件路径
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if self.is_valid_image(file_path):
                    self.load_image(file_path)
                    # 发送信号而不是直接调用方法
                    self.image_loaded_signal.emit(file_path)
                    break
        
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaaaaa;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 10px;
                font-size: 16px;
            }
            QLabel:hover {
                border-color: #3498db;
                background-color: #e8f4fc;
            }
        """)
    
    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path and self.is_valid_image(file_path):
            self.load_image(file_path)
            # 发送信号而不是直接调用方法
            self.image_loaded_signal.emit(file_path)
    
    def is_valid_image(self, file_path):
        """检查文件是否为有效图像"""
        try:
            Image.open(file_path)
            return True
        except:
            return False
    
    def load_image(self, file_path):
        """加载并显示图像"""
        self.image_path = file_path
        pixmap = QPixmap(file_path)
        
        # 保持纵横比缩放图像
        if pixmap.width() > pixmap.height():
            pixmap = pixmap.scaledToWidth(min(self.width() - 20, 400))
        else:
            pixmap = pixmap.scaledToHeight(min(self.height() - 20, 400))
        
        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignCenter)


class ResultDisplay(QFrame):
    """
    识别结果显示组件
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(300)
        
        # 创建布局
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("Recognition Result")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        
        # 结果标签
        self.class_label = QLabel("Class: -")
        self.class_label.setFont(QFont("Arial", 14))
        self.class_label.setWordWrap(True)
        
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setFont(QFont("Arial", 14))
        
        # 图像信息
        self.image_info_label = QLabel("Image: -")
        self.image_info_label.setFont(QFont("Arial", 12))
        self.image_info_label.setWordWrap(True)
        
        # 添加到布局
        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addWidget(self.class_label)
        layout.addWidget(self.confidence_label)
        layout.addSpacing(10)
        layout.addWidget(self.image_info_label)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_result(self, class_name, confidence, image_path):
        """更新识别结果"""
        self.class_label.setText(f"Class: {class_name}")
        self.confidence_label.setText(f"Confidence: {confidence:.2%}")
        
        # 更新图像信息
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                file_name = os.path.basename(image_path)
                self.image_info_label.setText(f"Image: {file_name}\nSize: {width}x{height}")
        except:
            self.image_info_label.setText(f"Image: {os.path.basename(image_path)}")
    
    def clear_result(self):
        """清除识别结果"""
        self.class_label.setText("Class: -")
        self.confidence_label.setText("Confidence: -")
        self.image_info_label.setText("Image: -")


class EWasteRecognitionUI(QMainWindow):
    """
    电子垃圾识别系统主窗口
    """
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("E-Waste Recognition System")
        self.setMinimumSize(800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("E-Waste Recognition System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        
        # 创建拖拽区域和结果显示区域
        content_layout = QHBoxLayout()
        
        # 创建拖拽区域和结果显示区域
        self.drop_area = DropArea(self)  # 直接设置父窗口为self
        self.result_display = ResultDisplay(self)  # 直接设置父窗口为self
        
        # 连接信号
        self.drop_area.image_loaded_signal.connect(self.on_image_loaded)
        
        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.drop_area)
        splitter.addWidget(self.result_display)
        splitter.setSizes([400, 400])
        
        content_layout.addWidget(splitter)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        
        self.recognize_button = QPushButton("Recognize")
        self.recognize_button.clicked.connect(self.recognize_image)
        self.recognize_button.setEnabled(False)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_all)
        
        # 添加调试按钮
        self.debug_button = QPushButton("Debug Info")
        self.debug_button.clicked.connect(self.show_debug_info)
        
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.recognize_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.debug_button)
        
        # 添加到主布局
        main_layout.addWidget(title_label)
        main_layout.addSpacing(10)
        main_layout.addLayout(content_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(button_layout)
        
        central_widget.setLayout(main_layout)
        
        # 设置状态栏
        self.statusBar().showMessage("Ready")
    
    def load_model(self):
        """加载模型"""
        self.statusBar().showMessage("Loading model...")
        try:
            # 检查模型文件是否存在
            if not os.path.exists(Config.best_model_path):
                self.statusBar().showMessage(f"Model file not found: {Config.best_model_path}")
                return
                
            self.predictor = Predictor(Config.best_model_path)
            self.statusBar().showMessage("Model loaded successfully")
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def select_image(self):
        """选择图像"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path and os.path.exists(file_path):
            self.drop_area.load_image(file_path)
            self.on_image_loaded(file_path)
    
    def on_image_loaded(self, image_path):
        """图像加载完成的回调"""
        print(f"Image loaded: {image_path}")  # 调试信息
        self.recognize_button.setEnabled(True)
        self.statusBar().showMessage(f"Image loaded: {os.path.basename(image_path)}")
        
        # 显示确认对话框
        # QMessageBox.information(self, "Image Loaded", f"Image loaded successfully: {os.path.basename(image_path)}\nRecognize button should now be enabled.")
    
    def recognize_image(self):
        """识别图像"""
        if not self.drop_area.image_path:
            self.statusBar().showMessage("No image selected")
            return
        
        if not self.predictor:
            self.statusBar().showMessage("Model not loaded")
            return
        
        try:
            self.statusBar().showMessage("Recognizing...")
            result = self.predictor.predict(self.drop_area.image_path)
            
            class_name = result['class_name']
            probability = result['probability']
            
            self.result_display.update_result(class_name, probability, self.drop_area.image_path)
            self.statusBar().showMessage(f"Recognition complete: {class_name} ({probability:.2%})")
        except Exception as e:
            self.statusBar().showMessage(f"Recognition failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def clear_all(self):
        """清除所有内容"""
        self.drop_area.setText("Drag and drop image here\nor click to select")
        self.drop_area.setPixmap(QPixmap())  # 清除图像
        self.drop_area.image_path = None
        self.result_display.clear_result()
        self.recognize_button.setEnabled(False)
        self.statusBar().showMessage("Ready")
    
    def show_debug_info(self):
        """显示调试信息"""
        debug_info = f"""
        Debug Information:
        
        Model Path: {Config.best_model_path}
        Model File Exists: {os.path.exists(Config.best_model_path)}
        Model Loaded: {self.predictor is not None}
        
        Image Path: {self.drop_area.image_path if self.drop_area.image_path else 'None'}
        Image File Exists: {os.path.exists(self.drop_area.image_path) if self.drop_area.image_path else 'N/A'}
        
        Recognize Button Enabled: {self.recognize_button.isEnabled()}
        """
        
        QMessageBox.information(self, "Debug Information", debug_info)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = EWasteRecognitionUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 