import numpy as np
import matplotlib.pyplot as plt
import time
import math


def proximal_gradient_method(A, b, lamda, x_true):
    # 记录运行时间
    start_time = time.time()
    # 初始化解的数组
    solutions = []
    # 初始化解，X是200维的变量
    x = np.zeros(200)
    # 学习率
    learning_rate = 0.0001
    for _ in range(3000):
        cur_x = x.copy()
        # 计算梯度
        gradient = np.zeros(200)
        for i in range(10):
            gradient = gradient + A[i].T.dot(A[i].dot(x) - b[i])
        x = (np.sign(x - learning_rate * gradient) *
             np.maximum(np.abs(x - learning_rate * gradient)
                        - lamda * learning_rate, 0))
        solutions.append(x)

        if np.linalg.norm(x - cur_x, ord=2) < 1e-5:
            break

    end_time = time.time()
    diff_time = end_time - start_time

    distances_true = [np.linalg.norm(iterate - x_true, ord=2) for iterate in solutions]
    distances_opt = [np.linalg.norm(iterate - x, ord=2) for iterate in solutions]
    plt.plot(distances_true, label='distance to x_true')
    plt.plot(distances_opt, label='distance to x_optimal')
    plt.title('proximal gradient method')
    plt.xlabel('iteration')
    plt.ylabel('distance')
    plt.grid()
    plt.legend()
    plt.show()


def admm(A,b,lamda,x_true):
    C = 1
    x = np.zeros(200)
    y = np.zeros(200)
    v = np.zeros(200)

    results = []
    start_time = time.time()
    for _ in range(3000):
        x_cur = x.copy()

        # 更新x
        x = np.linalg.inv(np.sum([A[i].T.dot(A[i]) for i in range(10)],
                                 axis=0) + C * np.eye((200)))
        x = x.dot(np.sum([A[i].T.dot(b[i]) for i in range(10)], axis=0)
                  + C * y - v)
        # 更新y
        y = np.sign(x + v/C) * np.maximum(np.abs(x + v/ C) - lamda / C, 0)
        results.append(x)

        # 判断收敛
        if np.linalg.norm(x - x_cur, ord=2) < 1e-5:
            break

    # 计算每步解与真实解之间以及最优解之间的距离
    distances_true = [np.linalg.norm(result - x_true, ord=2) for result in results]
    distances_opt = [np.linalg.norm(result - x, ord=2) for result in results]

    end_time = time.time()
    diff_time = end_time - start_time
    # 绘制距离变化图
    plt.plot(distances_true, label='distance to x_true')
    plt.plot(distances_opt, label='distance to x_optimal')
    plt.title('alternating direction method of multipliers')
    plt.xlabel('iteration')
    plt.ylabel('distance')
    plt.grid()
    plt.legend()
    plt.show()


def subgradient(A,b,lamda,x_true):
    start_time = time.time()
    x = np.zeros(200)  # 初始解
    learning_rate = 0.001
    results = []  # 记录每步的解

    for _ in range(3000):
        x_cur = x.copy()

        # 次梯度
        g = np.empty_like(x)
        for i, data in enumerate(x):
            if data == 0:
                g[i] = 2 * np.random.random() - 1  # [-1, 1]
            else:
                g[i] = np.sign(x[i])
        g *= lamda
        g += np.sum([A[i].T.dot(A[i].dot(x) - b[i]) for i in range(10)],
                    axis=0)
        # 更新x
        x = x - learning_rate * g

        results.append(x)

        # 判断收敛
        if np.linalg.norm(x - x_cur, ord=2) < 1e-5:
            break
    end_time = time.time()
    diff_time = end_time - start_time

    # 计算每步解与真实解之间以及最优解之间的距离
    distances_true = [np.linalg.norm(result - x_true, ord=2) for result in results]
    distances_opt = [np.linalg.norm(result - x, ord=2) for result in results]

    # 绘制距离变化图
    plt.figure()
    plt.plot(distances_true, label='distance to x_true')
    plt.plot(distances_opt, label='distance to x_optimal')
    plt.title('subgradient')
    plt.xlabel('iteration')
    plt.ylabel('distance')
    plt.grid()
    plt.legend()
    plt.show()


# 生成A: 均值为0⽅差为1的⾼斯分布
A = np.array([np.random.normal(0, 1, (5, 200))for _ in range(10)])
# 生成x的真值
# 200维
x_true = np.zeros(200)
# 稀疏度为5
nonzero = np.random.choice(200, 5, replace=False)
# 非零元素服从均值为0方差为1的高斯分布
x_true[nonzero] = np.random.normal(0, 1, 5)
# 生成b:bi=Aix+ei
# ei为5维的测量噪声,服从均值为0方差为0.1的高斯分布。
b = np.array([A[i].dot(x_true) + np.random.normal(0, 0.1, 5) for i in range(10)])

# 求解
type = 2
lamda = 10.0
if type == 0:
    proximal_gradient_method(A,b,lamda,x_true)
elif type == 1:
    admm(A,b,lamda,x_true)
elif type == 2:
    subgradient(A,b,lamda,x_true)
