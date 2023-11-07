import numpy as np  # 导入numpy库并简写为np
from mindquantum import Measure
from mindquantum.core.circuit import Circuit  # 导入Circuit模块，用于搭建量子线路
from mindquantum.core.gates import X, H, RY  # 导入量子门H, X, RY
from mindquantum.simulator import Simulator  # 从mindquantum.simulator中导入Simulator类


"""
量子模拟器构建
"""
sim = Simulator('mqvector', 2)  # 声明一个两比特的mqvector模拟器
# print(sim)  # 展示模拟器状态

sim.apply_gate(H.on(0))  # 作用一个Hadamard门到0号比特上
# print(sim)  # 展示模拟器状态

circ = Circuit()  # 声明一个空的量子线路
circ += H.on(1)  # 向其中添加一个hadamard门，并作用到1号比特上
circ += RY('a').on(0)  # 向其中添加一个参数化的RY门，并作用到0号比特上
# print(circ)

sim.apply_circuit(circ, pr={'a': np.pi})  # 作用一个量子线路，当线路是一个参数化量子线路时，我们还需要提供参数值。
# print(sim)

sim.set_qs(np.array([3 ** 0.5, 0, 0, 6 ** 0.5]))  # 设置模拟器状态，无需归一化
# print(sim.get_qs(True))  # 查看模拟器状态

sim.reset()  # 复位模拟器
# print(sim.get_qs())  # 查看模拟器状态
"""
量子电路测量
"""
encoder = Circuit()  # 初始化量子线路
encoder += H.on(0)  # H门作用在第0位量子比特
encoder += X.on(1, 0)  # X门作用在第1位量子比特且受第0位量子比特控制
encoder += Measure('q0').on(0)  # 在0号量子比特作用一个测量，并将该测量命名为'q0'
encoder += Measure('q1').on(1)  # 在1号量子比特作用一个测量，并将该测量命名为'q1'
encoder.svg().to_file('circuit_chapter_121.svg')  # 绘制SVG格式的量子线路图片
encoder.summary()

sim.reset()
result = sim.sampling(encoder, shots=1000)  # 对上面定义的线路采样1000次
"""
sampling(circuit, pr=None, shots=1, seed=None)是MindSpore Quantum中提供的对模拟器进行线路采样方法，它接受四个参数：

circuit (Circuit)：希望进行采样的量子线路，注意，该线路中必须包含至少一个测量操作（即采样点）。

pr (Union[None, dict, ParameterResolver])：parameter resolver，当 circuit是含参线路时，需要给出参数的值。

shots (int)：采样的次数，默认为1。

seed：采样时的随机种子，默认为一个随机数，可以不用提供
"""
print(result)
"""
含参量子电路构建和测量
"""
# 含参线路采样：
# para_circ = Circuit()  # 初始化量子线路
# para_circ += H.on(0)  # H门作用在第0位量子比特
# para_circ += X.on(1, 0)  # X门作用在第1位量子比特且受第0位量子比特控制
# para_circ += RY('theta').on(1)  # RY(theta)门作用在第2位量子比特
# para_circ += Measure('0').on(0)  # 在0号量子比特作用一个测量，并将该测量命名为'q0'
# para_circ += Measure('q1').on(1)  # 在1号量子比特作用一个测量，并将该测量命名为'q1'
# para_circ.svg().to_file('circuit_chapter_122.svg')  # 绘制SVG格式的量子线路图片
# sim.reset()
# result = sim.sampling(para_circ, {'theta': 0}, shots=1000)  # 将上面定义的线路参数'theta'赋值为0采样1000次
# # theta为0时可看作I门，即对量子不做任何操作
# result.svg()
