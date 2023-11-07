# 二维卷积函数的使用————conv2d
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision

dataset_trans = transforms.Compose([transforms.ToTensor()])
trans_set = torchvision.datasets.CIFAR10(root="dataset", train=False,
                                         transform=dataset_trans, download=True)
test_loader = DataLoader(dataset=trans_set, batch_size=64)


class MyNetWork(nn.Module):
    def __init__(self):
        super(MyNetWork, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6,
                            kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


network = MyNetWork()
print(network)

writer = SummaryWriter("logs")

step = 0
for data in test_loader:
    imgs, targets = data
    output = network(imgs)
    print(output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1
writer.close()
