import numpy as np  # 导入numpy库并简写为np
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ  # 导入量子门H, X, Y, Z, RX, RY, RZ
from mindquantum.core.circuit import Circuit

print('Gate name:', X)
print(X.matrix())

print('Gate name:', Y)
print(Y.matrix())


encoder = Circuit()                              # 初始化量子线路
encoder += H.on(0)                               # H门作用在第0位量子比特
encoder += X.on(1, 0)                            # X门作用在第1位量子比特且受第0位量子比特控制
encoder += RY('theta').on(2)                     # RY(theta)门作用在第2位量子比特

print(encoder)                                   # 打印Encoder
encoder.summary()                                # 总结Encoder量子线路

encoder.svg().to_file(filename='circuit_chapter_1.svg')