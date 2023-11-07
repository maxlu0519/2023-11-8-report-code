import torchvision
from PIL import Image
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import time


class MyNetWork(nn.Module):
    def __init__(self):
        super(MyNetWork, self).__init__()
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


write = SummaryWriter("logs")
val = 0
for i in range(13):
    image_path = f"myphoto/{i}.jpg"
    image = Image.open(image_path)
    image = image.convert("RGB")
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor()])
    image = transform(image)
    model = torch.load('model_save/network_model_26.pth', map_location=torch.device("cuda:0"))
    image = torch.reshape(image, (1, 3, 32, 32))
    write.add_images("image_scal", image, i)
    model.eval()
    with torch.no_grad():
        output = model(image.cuda())
    print(output)
    num = output.argmax(1)
    if num == 0:
        print("It is an airplane photo")
    elif num == 1:
        print("It is an automobile photo")
    elif num == 2:
        print("It is a bird photo")
    elif num == 3:
        print("It is a cat photo")
    elif num == 4:
        print("It is a deer photo")
    elif num == 5:
        print("It is a dog photo")
    elif num == 6:
        print("It is a frog photo")
    elif num == 7:
        print("It is a horse photo")
    elif num == 8:
        print("It is a ship photo")
    elif num == 9:
        print("It is a struck photo")

    if num == 3:
        val += 1

write.close()
