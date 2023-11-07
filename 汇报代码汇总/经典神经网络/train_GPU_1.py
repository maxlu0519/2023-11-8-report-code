# 准备数据，加载数据，准备模型，设置损失函数，设置优化器，开始训练，最后验证，结果聚合展示
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import time

# 导入数据集
dataset_trans = transforms.Compose([transforms.ToTensor()])
trans_set = torchvision.datasets.CIFAR10(root="dataset", train=True,
                                         transform=dataset_trans, download=True)
test_set = torchvision.datasets.CIFAR10(root="dataset", train=False,
                                        transform=dataset_trans, download=True)
print(f"训练集的长度为：{len(trans_set)}, \n测试集的长度为: {len(test_set)}.")

# 加载数据集
train_loader = DataLoader(dataset=trans_set, batch_size=64, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, drop_last=True)


# 搭建神经网络
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
            Sigmoid(),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# if __name__ == '__main__':
#     network = MyNetWork()
#     input = torch.ones((64, 3, 32, 32))
#     output = network(input)
#     print(output.shape)

# 创建网络模型
network = MyNetWork()
if torch.cuda.is_available():
    network = network.cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
# optimizer = optimizer.cuda()
# 设置参数
# 记录总训练次数
step_train_all = 0
# 记录总测试次数
step_test_all = 0
# 训练轮数
epoch = 100
# 添加tensorboard
write = SummaryWriter("logs_train")
# 设置时间
start_time = time.time()
for i in range(epoch):
    print(f"\t第{i + 1}轮训练开始：")

    # 训练步骤开始
    network.train()
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = network(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_train_all = step_train_all + 1
        if step_train_all % 100 == 0:
            end_time = time.time()
            print(f"训练时长为：{end_time-start_time}s")
            print(f"训练次数在{step_train_all}次时，Loss : {loss.item()}")
            write.add_scalar("train_loss", loss.item(), step_train_all)
        # 测试
    network.eval()
    loss_test_all = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = network(imgs)
            loss = loss_fn(outputs, targets)
            loss_test_all += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
    print(f"整体测试loss为{loss_test_all}")
    print(f"测试集整体正确率为：{accuracy/len(trans_set)*100}%")
    write.add_scalar("test_loss", loss_test_all, step_test_all)
    write.add_scalar("test_accuracy", accuracy/len(trans_set)*100, step_test_all)
    step_test_all += 1

    # 保存每次的模型
    torch.save(network, f"model_save/network_model_{i}.pth")
    print("模型已保存")
write.close()
