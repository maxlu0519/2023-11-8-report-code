import numpy as np                                  # 导入numpy库并简写为np
from mindquantum.core.circuit import Circuit        # 导入Circuit模块，用于搭建量子线路
from mindquantum.core.gates import H, RX, RY, RZ    # 导入量子门H, RX, RY, RZ
from mindquantum.core.operators import QubitOperator           # 导入QubitOperator模块，用于构造泡利算符
from mindquantum.core.operators import Hamiltonian             # 导入Hamiltonian模块，用于构建哈密顿量
from mindquantum.simulator import Simulator
from mindquantum.framework import MQLayer          # 导入MQLayer
import mindspore as ms                             # 导入mindspore
from mindspore.nn import Adam, TrainOneStepCell                   # 导入Adam模块和TrainOneStepCell模块

"""
搭建Encoder
"""
# pylint: disable=W0104
encoder = Circuit()                   # 初始化量子线路
encoder += H.on(0)                    # H门作用在第0位量子比特
encoder += RX(f'alpha{0}').on(0)      # RX(alpha_0)门作用在第0位量子比特
encoder += RY(f'alpha{1}').on(0)      # RY(alpha_1)门作用在第0位量子比特
encoder += RZ(f'alpha{2}').on(0)      # RZ(alpha_2)门作用在第0位量子比特
encoder = encoder.no_grad()           # Encoder作为整个量子神经网络的第一层，不用对编码线路中的梯度求导数，因此加入no_grad()
encoder.as_encoder()                  # 将量子线路设置为encoder线路
# encoder.summary()                     # 总结Encoder
encoder.svg().to_file('circuit_chapter_one_bit_neural_1.svg')
# print(encoder)

# 设置随机参数
alpha0, alpha1, alpha2 = 0.2, 0.3, 0.4              # alpha0, alpha1, alpha2为已知的固定值，分别赋值0.2, 0.3 和0.4
state = encoder.get_qs(pr={'alpha0': alpha0, 'alpha1': alpha1, 'alpha2': alpha2}, ket=True)
# print(state)
"""
搭建Ansatz
"""
# pylint: disable=W0104
ansatz = Circuit()                           # 初始化量子线路
ansatz += RX(f'theta{0}').on(0)              # RX(theta_0)门作用在第0位量子比特
ansatz += RY(f'theta{1}').on(0)              # RY(theta_1)门作用在第0位量子比特
ansatz.as_ansatz()                           # 将量子线路设置成待训练线路
# ansatz.summary()
# print(ansatz)                                # 打印量子线路

# 初始化参数
theta0, theta1 = 0, 0                        # 对theta0, theta1进行赋值，设为初始值0, 0
state = ansatz.get_qs(pr=dict(zip(ansatz.params_name, [theta0, theta1])), ket=True)
# print(state)

"""
构建完整的混合电路
"""
# pylint: disable=W0104
circuit = encoder.as_encoder() + ansatz.as_ansatz()                   # 完整的量子线路由Encoder和Ansatz组成
print(circuit)
circuit.svg().to_file('VQE电路绘制.svg')  # 绘制SVG格式的量子线路图片
"""
构建哈密顿量
"""
ham = Hamiltonian(QubitOperator('Z0', -1))                     # 对第0位量子比特执行泡利Z算符测量，且将系数设置为-1，构建对应的哈密顿量
# print(ham)

"""
生成变分量子线路模拟算子
"""
encoder_names = encoder.params_name                   # Encoder中所有参数组成的数组，encoder.para_name系统会自动生成
ansatz_names = ansatz.params_name                     # Ansatz中所有参数组成的数组，ansatz.para_name系统会自动生成

print('encoder_names = ', encoder.params_name, '\nansatz_names =', ansatz.params_name)

"""
制备算子
"""
# 生成一个基于mqvector后端的模拟器，并设置模拟器的比特数为量子线路的比特数。
sim = Simulator('mqvector', circuit.n_qubits)

# 获取模拟器基于当前量子态的量子线路演化以及期望、梯度求解算子
grad_ops = sim.get_expectation_with_grad(ham, circuit)

# Encoder中的alpha0, alpha1, alpha2这三个参数组成的数组，
# 将其数据类型转换为float32，并储存在encoder_data中。
# MindSpore Quantum支持多样本的batch训练，Encoder数组是两个维度，
# 第一个维度为样本，第二个维度为特征（即参数）
encoder_data = np.array([[alpha0, alpha1, alpha2]]).astype(np.float32)

# Ansatz中的theta0, theta1这两个参数组成的数组，将其数据类型转换为float32，
# 并储存在ansatzr_data中，Ansatz数据只有一个维度，特征（即参数）
ansatz_data = np.array([theta0, theta1]).astype(np.float32)

# 根据Encoder和Ansatz的数据，输出变分量子线路的测量值，Encoder中的参数的导数和Ansatz中的参数的导数
measure_result, encoder_grad, ansatz_grad = grad_ops(encoder_data, ansatz_data)

print('Measurement result: ', measure_result)
print('Gradient of encoder parameters: ', encoder_grad)
print('Gradient of ansatz parameters: ', ansatz_grad)
"""
神经网络搭建
"""
ms.set_seed(1)                                     # 设置生成随机数的种子
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

QuantumNet = MQLayer(grad_ops)
print(QuantumNet)

# from mindspore import nn
#
# class MQLayer(nn.Cell):
#     def __init__(self, expectation_with_grad, weight='normal'):
#         super(MQLayer, self).__init__()
#         self.evolution = MQOps(expectation_with_grad)
#         weight_size = len(
#             self.evolution.expectation_with_grad.ansatz_params_name)
#         self.weight = Parameter(initializer(weight,
#                                             weight_size,
#                                             dtype=ms.float32),
#                                 name='ansatz_weight')
#
#     def construct(self, x):
#         return self.evolution(x, self.weight)

opti = Adam(QuantumNet.trainable_params(), learning_rate=0.5)     # 需要优化的是Quantumnet中可训练的参数，学习率设为0.5
net = TrainOneStepCell(QuantumNet, opti)

for i in range(200):
    res = net(ms.Tensor(encoder_data))
    if i % 10 == 0:
        print(i, ': ', res)

"""
打印此时Ansatz中的参数并输出量子线路在最优参数时的量子态
"""
theta0, theta1 = QuantumNet.weight.asnumpy()

print(QuantumNet.weight.asnumpy())

pr = {'alpha0': alpha0, 'alpha1': alpha1, 'alpha2': alpha2, 'theta0': theta0, 'theta1': theta1}
state = circuit.get_qs(pr=pr, ket=True)

print(state)

"""
打印保真度
"""
state = circuit.get_qs(pr=pr)
fid = np.abs(np.vdot(state, [1, 0]))**2            # 保真度fidelity为向量内积的绝对值的模平方，即计算此时量子态对应的向量与|0>态对应的向量[1,0]的内积的模平方

print(fid)