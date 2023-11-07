# 神经网络类的构建框架————详细请见CIFAR10文件
from torch import nn
import torch


class NetWork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


network = NetWork()
x = torch.tensor(1.0)
print(network(x))
