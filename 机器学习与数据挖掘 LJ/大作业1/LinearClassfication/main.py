from ucimlrepo import fetch_ucirepo
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import scipy
import time

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


# def sigmoid(x):
#     x=np.array(x,dtype=np.float64)
#     return 1 / (1 + np.exp(-x))


def Predict(X_normalized, Z, w, b, index):
    Z[index] = sign(np.dot(X_normalized.iloc[index].values, w) + b)
    return Z[index]


def Cost(X_normalized, Y, index, w, b):  # 单个数据点的损失
    if index < 0 or index >= len(X_normalized):
        raise ValueError("Index is out of bounds.")
    linear_com = np.dot(w, X_normalized.iloc[index].values) + b
    # p = sigmoid(linear_com)
    # p = p.astype('float')
    # cost = - (Y.iloc[index].values * np.log(p) + (1 - Y.iloc[index].values) * np.log(1 - p))  # Cross-Entropy Loss
    cost = max(0, 1 - Y.iloc[index].values * linear_com)
    return cost


def Loss(X_normalized, Y, row,w ,b):  # 求损失函数
    sum = 0
    count = 0
    for i in range(0, row):
        if Cost(X_normalized, Y, i, w, b) >= 0:
            sum = sum + Cost(X_normalized, Y, i, w, b)
            count = count + 1  # 误分类的总数
    return sum / count


def Gradient_Descent(X_normalized, Y, w, b, n, row):
    for index in range(0, row):
        if Cost(X_normalized, Y, index,w, b) >= 0:  # 误分类点，更新参数
            b = b + np.dot(n, Y.iloc[index].values)
            w = w + n * np.dot(Y.iloc[index].values[0], X_normalized.iloc[index].values)
    return w, b


def Iterate(time, X_normalized, Y, Z, w, b, n, row, loss_list):  # 迭代
    for i in range(0, time):
        if i % 100 ==0:
            print(i)
        loss_list.append(Loss(X_normalized, Y, row,w,b))
        w, b = Gradient_Descent(X_normalized, Y, w, b, n, row)  # 更新所有参数
    sum = 0
    for i in range(0, row):
        tmp = Predict(X_normalized, Z, w, b, i)
        if tmp >= 0 and Y.iloc[i].item() == 1:
            sum = sum + 1
        elif tmp < 0 and Y.iloc[i].item() == -1:
            sum = sum + 1
    print("正确率：", sum / row)
    return loss_list, w, b


if __name__ == '__main__':
    # fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # metadata
    print(breast_cancer_wisconsin_diagnostic.metadata)

    # variable information
    print(breast_cancer_wisconsin_diagnostic.variables)

    detail = X.shape
    row = detail[0]
    save_row = row
    col = detail[1]

    # normalization
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

    # targets
    for index, row in y.iterrows():
        if row['Diagnosis'] == 'M':
            y.at[index, 'Diagnosis'] = 1
        else:
            y.at[index, 'Diagnosis'] = -1

    # 使用新数据执行线性分类：预测，计算损失函数，梯度下降
    # 初始化权重和偏置
    z = np.zeros(save_row)
    w = np.ones(col)
    print(w)
    b = 0
    n = 0.01
    loss_list = list()
    time_start = time.time()

    # 开始迭代
    loss_list, w, b = Iterate(500, X_normalized, y, z, w, b, n, save_row, loss_list)

    # 计算分类准确率
    time_end = time.time()
    time = time_end - time_start
    print("执行时间：", time)
    y = loss_list
    x = list(range(500))
    plt.scatter(x, y, marker='.', color='red')
    plt.xlabel("Time")
    plt.ylabel("Loss")

    # 显示
    plt.show()
