import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def Predict(X_fixed, Z, w, b, index):
    Z[index] = sign(np.dot(X_fixed[index], w) + b)
    return Z[index]


def Cost(X_fixed, Y, index, w, b):  # 单个数据点的损失
    linear_com = np.dot(w, X_fixed[index]) + b
    cost = -1 * Y[index] * linear_com  # - yi * (w * xi + b)
    #print(w, X[index], Y[index], linear_com, cost)
    return cost


def Loss(X_fixed, Y, row,w ,b):  # 求损失函数
    sum = 0
    count = 0
    for i in range(0, row):
        if Cost(X_fixed, Y, i, w, b) >= 0:
            sum = sum + Cost(X_fixed, Y, i, w, b)
            count = count + 1  # 误分类的总数
    return sum / count


def Gradient_Descent(X_fixed, Y, w, b, n, row):
    for index in range(0, row):
        #print(Y[index])
        if Cost(X_fixed, Y, index,w, b) >= 0:  # 误分类点，更新参数
            b = b + n * Y[index]
            w = w + n * np.dot(Y[index], X_fixed[index])
            #print(n, Y[index], np.dot(Y[index], X[index]), X[index])
    return w, b


def Iterate(time, X_fixed, Y, Z, w, b, n, row, loss_list):  # 迭代
    for i in range(0, time):
        loss_list.append(Loss(X_fixed, Y, row,w,b))
        w, b = Gradient_Descent(X_fixed, Y, w, b, n, row)  # 更新所有参数
    sum = 0
    for i in range(0, row):
        tmp = Predict(X_fixed, Z, w, b, i)
        if tmp >= 0 and Y[i] == 1:
            sum = sum + 1
        elif tmp < 0 and Y[i] == -1:
            sum = sum + 1
    print("正确率：", sum / row)
    return loss_list, w, b


if __name__ == '__main__':
    data = pd.read_csv("data.csv", encoding="utf-8")
    data = pd.DataFrame(data)
    X = data.values[:, :-1]  # 前两列数据
    detail = X.shape
    row = detail[0]  # 行数
    col = detail[1]  # 列数
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    X_fixed = np.zeros((row, col))  # 存储归一化后的数据，初始化为全0
    for i in range(0, 400):  # 这个循环是归一化过程
        #print(X[i])
        X_fixed[i][0] = round((X[i][0] - min_values[0]) / (max_values[0] - min_values[0]), 6)
        X_fixed[i][1] = round((X[i][1] - min_values[1]) / (max_values[1] - min_values[1]), 6)
        #print(X[i][0] - min_values[0], max_values[0] - min_values[0], X[i][1] - min_values[1], max_values[1] - min_values[1], X_fixed[i][0], X_fixed[i][1], X_fixed[i])
    Y = data.values[:, -1]  # 标准的分类结果
    for i in range(400):
        if Y[i] == 0:
            Y[i] = -1  # 将标签0改成-1. 方便后续运算
    Z = np.zeros(row)  # 算法得出的分类结果
    #for i in range(400):
        #print(X_fixed[i], Y[i])

    # 使用新数据执行感知机算法: 随机初始化，计算输出，梯度下降，判断收敛。如果收敛，结束。
    # 初始化权重和偏置
    w = np.ones(col)
    b = 0
    n = 0.01
    loss_list = list()
    # 开始迭代
    loss_list, w, b = Iterate(3000, X_fixed, Y, Z, w, b, n, row, loss_list)
    #print(w,b)
    # 绘制数据可视化图，loss曲线图，计算分类准确率
    # 生成直线
    x_line = np.linspace(min(X_fixed[:, 0]), max(X_fixed[:, 0]), 100)
    y_line = -1 * w[0]/w[1] * x_line - b / w[1]
    # 绘制散点图
    figure1 = plt.figure(1)
    for i in range(row):
        if Y[i] == -1:
            plt.scatter(X_fixed[i][0], X_fixed[i][1], color='purple')
        else:
            plt.scatter(X_fixed[i][0], X_fixed[i][1], color='yellow')
    # 绘制一次函数的图像
    plt.plot(x_line, y_line, color='black', label='Linear Function')

    # 标题，x轴，y轴
    plt.xlabel('Age')
    plt.ylabel('Money')
    fig2 = plt.figure(2)
    y = loss_list
    x = list(range(3000))
    plt.scatter(x, y, marker='.', color='red')
    plt.xlabel("Time")
    plt.ylabel("Loss")
    # 显示
    plt.show()


