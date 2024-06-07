import time
import numpy as np
# import decimal  #用numpy计算，弃用decimal
# decimal.getcontext().prec = 50

import matplotlib.pyplot as plt
"""
Consider an example: g(w) = w**3 - 5
analytical solution: g(w) = 0; w**3 = 5;  5^(1/3) ≈ 1.71.

Now, suppose that we can only observe the input w and the output   g̃(w) = g(w) + η,

"""

w_k = [0]  # w_1 = 0
g_tilde = []

# eta = np.random.normal(size=10) # η 高斯噪声
# print("eta:",eta)
eta_list = [] #plot用
def calculate_g_tilde(w):
    eta = np.random.normal()
    eta_list.append(eta)
    g_tilde =np.array(w**3) - 5 + eta

    # g_tilde = decimal.Decimal(w ** 3) - 5
    return (g_tilde)

for a_k in range(2,550):  # a_k 要从2开始
    g_tilde.append( calculate_g_tilde(w_k[-1]) ) # g_k
    # print("g_tilde",g_tilde)
    w_k.append( w_k[-1] - np.array(1/a_k) * g_tilde[-1] )
    # print("w_k" ,w_k)
print("w_k",w_k)  #w_k[-1]是结果
print('---------------------------')
print("实际结果：",np.cbrt(5))  #numpy开立方
print("迭代最后结果：",w_k[-1])



# 绘制第一个图表
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(w_k)+1), w_k,  marker='o',markerfacecolor='none',   # 空心，设置填充色为透明
         markeredgecolor='blue',   # 边框颜色为蓝色
         markersize=10,
         linestyle='-', color='blue', label='Estimated root w_k')
plt.xlabel('Iteration index k', fontsize = 12)
plt.ylabel('Estimated root w_k', fontsize = 12)

# 绘制第二个图表
plt.figure(figsize=(8, 5))
plt.plot(range(len(eta_list)), eta_list,  marker='o',markerfacecolor='none',   # 空心，设置填充色为透明
         markeredgecolor='green',   # 边框颜色为蓝色
         markersize=10,
         linestyle='-',  color='green', label='Observation noise')
plt.xlabel('Iteration index k', fontsize = 12)
plt.ylabel('Observation noise', fontsize = 12)

# 添加图例
plt.legend()

# 显示图表
plt.show()