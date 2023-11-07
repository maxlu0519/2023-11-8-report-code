# 非线性函数————激活函数
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision

dataset_trans = transforms.Compose([transforms.ToTensor()])
trans_set = torchvision.datasets.CIFAR10(root="dataset", train=False,
                                         transform=dataset_trans, download=True)
test_loader = DataLoader(dataset=trans_set, batch_size=64)
m_input = torch.tensor([[1, -0.5],
                        [-1, 3]], dtype=torch.float)

m_input = torch.reshape(m_input, (-1, 1, 2, 2))
print(m_input.shape)


class MyNetWork(nn.Module):
    def __init__(self):
        super(MyNetWork, self).__init__()
        self.relu1 = ReLU(inplace=False)
        self.sigmiod1 = Sigmoid()

    def forward(self, x):
        output = self.sigmiod1(x)
        return output


network = MyNetWork()
print(network(m_input))

writer = SummaryWriter("logs")

step = 0
for data in test_loader:
    imgs, target = data
    output = network(imgs)
    print(output.shape)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1
writer.close()