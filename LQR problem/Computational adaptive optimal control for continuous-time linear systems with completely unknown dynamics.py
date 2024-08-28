import numpy as np
from scipy.linalg import solve_continuous_are


# Sys parameters
# 定义系统矩阵
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[1.0], [0.0]])
Q = np.eye(2)  # diag(1,1)
R = np.eye(1)  # diag(1)

# 直接求解Riccati方程
P = solve_continuous_are(A, B, Q, R)

# 计算最优反馈增益矩阵K
K = np.linalg.inv(R) @ B.T @ P

print("P*:\n",P)
print("直接求解方法得到的反馈增益矩阵K:")
print(K)


print("=======================================================================")
print("计算增益矩阵的迭代过程:")
import numpy as np
from scipy.linalg import solve_continuous_are

# 定义系统矩阵
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[1.0], [0.0]])
Q = np.eye(2)  # diag(1,1)
R = np.eye(1)  # diag(1)

# 初始化参数
A_hat = np.zeros_like(A)
B_hat = np.eye(A.shape[0], B.shape[1])
P = np.eye(A.shape[0])
K = np.zeros((B.shape[1], A.shape[0]))

# 迭代次数
iterations = 10


def get_new_data():
    # 模拟数据收集函数，返回新的状态和输入数据
    x_t = np.random.rand(A.shape[0])
    u_t = np.random.rand(B.shape[1])
    return x_t, u_t


for k in range(iterations):
    # 求解Riccati方程
    P = Q + K.T @ R @ K + (A_hat - B_hat @ K).T @ P @ (A_hat - B_hat @ K)

    # 更新控制增益矩阵
    K = np.linalg.inv(R) @ B_hat.T @ P @ A_hat

    # 模拟数据收集和更新A_hat
    x_t, u_t = get_new_data()
    z_t = np.hstack((x_t, u_t)).reshape(1, -1)  # 确保z_t是二维数组

    # 使用最小二乘法更新A_hat
    A_hat = np.linalg.lstsq(z_t.T, x_t.reshape(-1, 1), rcond=None)[0].T

print("迭代方法得到的反馈增益矩阵K:")
print(K)
