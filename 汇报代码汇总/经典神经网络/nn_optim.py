# 有优化器的整体演示 十分重要的全面演示
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision

dataset_trans = transforms.Compose([transforms.ToTensor()])
trans_set = torchvision.datasets.CIFAR10(root="dataset", train=False,
                                         transform=dataset_trans, download=True)
test_loader = DataLoader(dataset=trans_set, batch_size=64, drop_last=True)


class MyNetWork(nn.Module):
    def __init__(self):
        super(MyNetWork, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32,
                            kernel_size=5, stride=1, padding=2)
        self.pool1 = MaxPool2d(kernel_size=2, ceil_mode=True)
        self.conv2 = Conv2d(in_channels=3, out_channels=32,
                            kernel_size=5, stride=1, padding=2)
        self.pool2 = MaxPool2d(kernel_size=2, ceil_mode=True)
        self.conv3 = Conv2d(in_channels=3, out_channels=32,
                            kernel_size=5, stride=1, padding=2)
        self.pool3 = MaxPool2d(kernel_size=2, ceil_mode=True)
        self.flatten = Flatten()
        self.line1 = Linear(1024, 64)
        self.line2 = Linear(64, 10)
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32,
                   kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=True),
            Conv2d(in_channels=32, out_channels=32,
                   kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=True),
            Conv2d(in_channels=32, out_channels=64,
                   kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, ceil_mode=True),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
            )

    def forward(self, x):
        x = self.model1(x)
        return x


network = MyNetWork()
print(network)
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(network.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.
    for number in test_loader:
        imgs, targets = number
        outputs = network(imgs)
        result = loss(outputs, targets)
        optim.zero_grad()
        result.backward()
        optim.step()
        running_loss = result + running_loss
    print(running_loss)


