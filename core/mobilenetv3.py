"""
MobileNetV3 模型定义
"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import TruncatedNormal
import mindspore as ms
import numpy as np

def _make_divisible(v, divisor, min_value=None):
    """确保通道数是8的倍数"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class HSigmoid(nn.Cell):
    """Hard Sigmoid激活函数"""
    def __init__(self):
        super(HSigmoid, self).__init__()
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        return self.relu6(x + 3.) / 6.

class HSwish(nn.Cell):
    """Hard Swish激活函数"""
    def __init__(self):
        super(HSwish, self).__init__()
        self.hsigmoid = HSigmoid()

    def construct(self, x):
        return x * self.hsigmoid(x)

class SEModule(nn.Cell):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.fc = nn.SequentialCell([
            nn.Conv2d(channel, channel // reduction, 1, 1, pad_mode='pad', padding=0),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, 1, pad_mode='pad', padding=0),
            HSigmoid()
        ])

    def construct(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x, (2, 3))
        y = self.fc(y)
        return x * y

class InvertedResidual(nn.Cell):
    """MobileNetV3倒置残差块"""
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        self.identity = stride == 1 and inp == oup

        activation = HSwish() if use_hs else nn.ReLU()

        layers = []
        if inp != hidden_dim:
            # pw
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, pad_mode='pad', padding=0))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(activation)

        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                     pad_mode='pad', padding=kernel_size//2, group=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            activation
        ])

        if use_se:
            layers.append(SEModule(hidden_dim))

        # pw-linear
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, 1, pad_mode='pad', padding=0),
            nn.BatchNorm2d(oup)
        ])

        self.conv = nn.SequentialCell(layers)

    def construct(self, x):
        if self.identity:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV3(nn.Cell):
    """
    MobileNetV3 网络结构
    
    Args:
        num_classes: 分类数量
        width_mult: 宽度乘数，用于调整网络大小
        mode: 'small' 或 'large'，选择MobileNetV3的版本
    """
    def __init__(self, num_classes=1000, width_mult=1.0, mode='small'):
        super(MobileNetV3, self).__init__()
        
        # 设置平均池化的轴
        self.axis = (2, 3)  # 在H和W维度上进行平均池化
        
        # 设置输入通道数和最后一个通道数
        input_channel = 16
        last_channel = 1280
        
        # 根据模式选择不同的网络结构
        if mode == 'large':
            # kernel_size, exp_size, out_channels, use_SE, use_HS, stride
            self.cfgs = [
                [3, 16, 16, False, False, 1],
                [3, 64, 24, False, False, 2],
                [3, 72, 24, False, False, 1],
                [5, 72, 40, True, False, 2],
                [5, 120, 40, True, False, 1],
                [5, 120, 40, True, False, 1],
                [3, 240, 80, False, True, 2],
                [3, 200, 80, False, True, 1],
                [3, 184, 80, False, True, 1],
                [3, 184, 80, False, True, 1],
                [3, 480, 112, True, True, 1],
                [3, 672, 112, True, True, 1],
                [5, 672, 160, True, True, 2],
                [5, 960, 160, True, True, 1],
                [5, 960, 160, True, True, 1],
            ]
            init_conv_out = 16
            final_conv_out = 1280
        else:  # small
            self.cfgs = [
                [3, 16, 16, True, False, 2],
                [3, 72, 24, False, False, 2],
                [3, 88, 24, False, False, 1],
                [5, 96, 40, True, True, 2],
                [5, 240, 40, True, True, 1],
                [5, 240, 40, True, True, 1],
                [5, 120, 48, True, True, 1],
                [5, 144, 48, True, True, 1],
                [5, 288, 96, True, True, 2],
                [5, 576, 96, True, True, 1],
                [5, 576, 96, True, True, 1],
            ]
            init_conv_out = 16
            final_conv_out = 1024

        self.conv_stem = nn.SequentialCell([
            nn.Conv2d(3, init_conv_out, 3, 2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(init_conv_out),
            HSwish()
        ])

        # 构建中间层
        self.blocks = []
        input_channel = init_conv_out
        for k, exp, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(exp * width_mult, 8)
            self.blocks.append(
                InvertedResidual(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
            )
            input_channel = output_channel

        self.blocks = nn.SequentialCell(self.blocks)

        # 构建分类头
        self.conv_head = nn.SequentialCell([
            nn.Conv2d(input_channel, final_conv_out, 1, 1, pad_mode='pad', padding=0),
            nn.BatchNorm2d(final_conv_out),
            HSwish()
        ])

        self.avgpool = ops.ReduceMean(keep_dims=True)
        
        self.classifier = nn.SequentialCell([
            nn.Dense(final_conv_out, num_classes),
        ])

        self._initialize_weights()

    def construct(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.avgpool(x, self.axis)  # 使用类属性axis
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """初始化模型权重"""
        for name, cell in self.cells_and_names():
            try:
                if isinstance(cell, nn.Conv2d):
                    # 使用默认初始化方式
                    cell.weight.set_data(
                        ms.common.initializer.initializer(
                            'TruncatedNormal', 
                            cell.weight.shape,
                            ms.float32
                        )
                    )
                    
                    if cell.bias is not None:
                        cell.bias.set_data(
                            ms.common.initializer.initializer(
                                'TruncatedNormal',
                                cell.bias.shape,
                                ms.float32
                            )
                        )
                
                elif isinstance(cell, nn.Dense):
                    # 使用默认初始化方式
                    cell.weight.set_data(
                        ms.common.initializer.initializer(
                            'TruncatedNormal', 
                            cell.weight.shape,
                            ms.float32
                        )
                    )
                    
                    if cell.bias is not None:
                        cell.bias.set_data(
                            ms.common.initializer.initializer(
                                'TruncatedNormal',
                                cell.bias.shape,
                                ms.float32
                            )
                        )
            
            except Exception as e:
                print(f"Error initializing weights for {name}: {str(e)}")
                print(f"Cell type: {type(cell)}")
                if hasattr(cell, 'weight'):
                    print(f"Weight shape type: {type(cell.weight.shape)}")
                    print(f"Weight shape: {cell.weight.shape}")
                raise 