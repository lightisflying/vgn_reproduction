from builtins import super

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def get_network(name): #获得网络，此处为自己编写的卷积神经网络
    models = {
        "conv": ConvNet(),
    }
    return models[name.lower()]


def load_network(path, device):
    """Construct the neural network and load parameters from the specified file.

    Args:
        path: Path to the model parameters. The name must conform to `vgn_name_[_...]`.

    """
    model_name = path.stem.split("_")[1]  # 从文件名中提取模型名称
    net = get_network(model_name).to(device)  # 创建指定模型的网络实例，并将其移动到指定设备上
    net.load_state_dict(torch.load(path, map_location=device))  # 从文件加载模型参数
    return net

"""
in_channels/out_channels: 输入/出通道数
kernel_size: 卷积核（就是图像处理中的那个滤波器）大小
stride/padding: 步长/补位（输入的每一条边补充0的层数）
kernel_size的第一维度的值是每次处理的图像帧数，后面是卷积核的大小
在3D卷积中，一个kernel可以同时对于时间维度上的多帧图像进行卷积，具体对几帧就是由kernel的第一个维度的参数来确定。
"""
def conv(in_channels, out_channels, kernel_size):# https://blog.csdn.net/qq_41744950/article/details/123573949 Conv3d的理解
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

# 步长大于1时有下采样的效果，如此时stride=2时feature map的尺寸会减小一半
def conv_stride(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2)


class ConvNet(nn.Module):# 卷积神经网络的编写
 #[TODO1]可尝试调整/分析这里的参数/内容
    def __init__(self):
        super().__init__()#
        #下面的Encoder应该就是第一部分那个映射网络
        #Decoder是第二部分
        self.encoder = Encoder(1, [16, 32, 64], [5, 3, 3])  # 创建编码器实例，输入通道数为1，输出通道数分别为16、32、64，卷积核大小分别为5、3、3
        self.decoder = Decoder(64, [64, 32, 16], [3, 3, 5])  # 创建解码器实例，输入通道数为64，输出通道数分别为64、32、16，卷积核大小分别为3、3、5
        self.conv_qual = conv(16, 1, 5)  # 创建质量卷积层实例，输入通道数为16，输出通道数为1，卷积核大小为5
        self.conv_rot = conv(16, 4, 5)  # 创建旋转卷积层实例，输入通道数为16，输出通道数为4，卷积核大小为5
        self.conv_width = conv(16, 1, 5)  # 创建宽度卷积层实例，输入通道数为16，输出通道数为1，卷积核大小为5

    def forward(self, x):
        x = self.encoder(x)  # 运行编码器
        x = self.decoder(x)  # 运行解码器
        qual_out = torch.sigmoid(self.conv_qual(x))  # 经过质量卷积层，并通过Sigmoid函数进行激活
        """
        传统神经网络中最常用的两个激活函数，Sigmoid系（Logistic-Sigmoid、Tanh-Sigmoid）被视为神经网络的核心所在。
        从数学上来看，非线性的Sigmoid函数对中央区的信号增益较大，对两侧区的信号增益小，在信号的特征空间映射上，有很好的效果。
        从神经科学上来看，中央区酷似神经元的兴奋态，两侧区酷似神经元的抑制态，
        因而在神经网络学习方面，可以将重点特征推向中央区，将非重点特征推向两侧区。
        """
        rot_out = F.normalize(self.conv_rot(x), dim=1)  # 经过旋转卷积层，并进行归一化处理（对每一行进行归一化）
        width_out = self.conv_width(x)  # 经过宽度卷积层
        return qual_out, rot_out, width_out
        # 输出分别为1维，4维，1维


class Encoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv_stride(in_channels, filters[0], kernels[0])  # 创建步长卷积层实例，输入通道数为in_channels，输出通道数为filters[0]，卷积核大小为kernels[0]
        self.conv2 = conv_stride(filters[0], filters[1], kernels[1])  # 创建步长卷积层实例，输入通道数为filters[0]，输出通道数为filters[1]，卷积核大小为kernels[1]
        self.conv3 = conv_stride(filters[1], filters[2], kernels[2])  # 创建步长卷积层实例，输入通道数为filters[1]，输出通道数为filters[2]，卷积核大小为kernels[2]

    def forward(self, x):
      """
      RELU函数：max(0,x)
      1） 解决了gradient vanishing问题 (在正区间)
      2）计算速度非常快，只需要判断输入是否大于0
      3）收敛速度远快于sigmoid和tanh
      """

      x = self.conv1(x)  # 运行第一个卷积层
      x = F.relu(x)  # 运行ReLU激活函数

      x = self.conv2(x)  # 运行第二个卷积层
      x = F.relu(x)  # 运行ReLU激活函数

      x = self.conv3(x)  # 运行第三个卷积层
      x = F.relu(x)  # 运行ReLU激活函数

      return x


class Decoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv(in_channels, filters[0], kernels[0])  # 创建卷积层实例，输入通道数为in_channels，输出通道数为filters[0]，卷积核大小为kernels[0]
        self.conv2 = conv(filters[0], filters[1], kernels[1])  # 创建卷积层实例，输入通道数为filters[0]，输出通道数为filters[1]，卷积核大小为kernels[1]
        self.conv3 = conv(filters[1], filters[2], kernels[2])  # 创建卷积层实例，输入通道数为filters[1]，输出通道数为filters[2]，卷积核大小为kernels[2]

    def forward(self, x):
        x = self.conv1(x)  # 运行第一个卷积层
        x = F.relu(x)  # 运行ReLU激活函数

        x = F.interpolate(x, 10)  # 进行上采样，尺寸放大为原来的10倍
        x = self.conv2(x)  # 运行第二个卷积层
        x = F.relu(x)  # 运行ReLU激活函数

        x = F.interpolate(x, 20)  # 进行上采样，尺寸放大为原来的20倍
        x = self.conv3(x)  # 运行第三个卷积层
        x = F.relu(x)  # 运行ReLU激活函数

        x = F.interpolate(x, 40)  # 进行上采样，尺寸放大为原来的40倍
        return x


def count_num_trainable_parameters(net):# 计算需要保存梯度相关信息的参数个数，保证梯度回传能够正常实现吧？
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
