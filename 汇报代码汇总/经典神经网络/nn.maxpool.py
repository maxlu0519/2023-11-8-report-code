# 池化核演示
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision

dataset_trans = transforms.Compose([transforms.ToTensor()])
trans_set = torchvision.datasets.CIFAR10(root="dataset", train=False,
                                         transform=dataset_trans, download=True)
test_loader = DataLoader(dataset=trans_set, batch_size=64)
m_input = torch.tensor([[1, 2, 0, 3, 1],
                        [0, 1, 2, 3, 1],
                        [1, 2, 1, 0, 0],
                        [5, 2, 3, 1, 1],
                        [2, 1, 0, 1, 1]], dtype=torch.float)

m_input = torch.reshape(m_input, (-1, 1, 5, 5))
print(m_input.shape)


class MyNetWork(nn.Module):
    def __init__(self):
        super(MyNetWork, self).__init__()
        self.pool1 = MaxPool2d(kernel_size=[3, 3], ceil_mode=True)

    def forward(self, x):
        out = self.pool1(x)
        return out


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