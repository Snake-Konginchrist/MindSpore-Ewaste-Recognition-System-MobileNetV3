"""
时间计算模块
"""
import time


def format_time(seconds):
    """
    将秒数格式化为小时:分钟:秒格式
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化后的时间字符串
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


class Timer:
    """计时器类，用于计算程序运行时间"""
    
    def __init__(self):
        """初始化计时器"""
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        return self
    
    def get_elapsed_time(self):
        """获取经过的时间（秒）"""
        if self.start_time is None:
            return 0
        
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time
    
    def get_formatted_time(self):
        """获取格式化后的时间字符串"""
        return format_time(self.get_elapsed_time()) 