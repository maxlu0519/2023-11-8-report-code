# 损失函数演示
import torch
from torch.nn import L1Loss, MSELoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float)
targets = torch.tensor([1, 2, 5], dtype=torch.float)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
loss2 = MSELoss()
result = loss(inputs, targets)
result2 = loss2(inputs, targets)

print(result)
print(result2)
