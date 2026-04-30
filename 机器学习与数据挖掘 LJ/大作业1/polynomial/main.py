from ucimlrepo import fetch_ucirepo
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import scipy


def train_and_plot(X_train, X_test, y_train, y_test, feature_index, feature_name):
    # 仅使用一个特征进行训练
    X_train_feature = X_train.iloc[:, feature_index].values.reshape(-1, 1)
    X_test_feature = X_test.iloc[:, feature_index].values.reshape(-1, 1)
    poly = PolynomialFeatures(degree=5)
    X_poly_train = poly.fit_transform(X_train_feature)
    model = LogisticRegression(max_iter=500)
    model.fit(X_poly_train, y_train)

    # 生成测试数据点
    x_range = np.linspace(X_train_feature.min(), X_train_feature.max(), 300).reshape(-1, 1)
    # 转换为多项式特征
    x_poly_range = poly.transform(x_range)
    # 预测
    y_prob = model.predict_proba(x_poly_range)[:, 1]
    # 画曲线
    plt.plot(x_range, y_prob, label=f'Feature: {feature_name}')
    # # 画数据点
    # plt.scatter(X_train_feature, y_train, color='red', s=10, alpha=1)


# def train(degree,X_train,X_test,y_train,y_test):
#     # 生成多项式特征
#     poly = PolynomialFeatures(degree)
#     # 将输入数据转换为多项式特征
#     X_poly_train = poly.fit_transform(X_train)
#     # 使用在训练集上学到的特征组合和参数，将 X_test 转换为相同的多项式特征矩阵
#     X_poly_test = poly.transform(X_test)
#     # 模型实例化&训练
#     model = LogisticRegression()
#     model.fit(X_poly_train, y_train)
#
#     # 使用已训练的逻辑回归模型对测试数据进行分类
#     y_predict = model.predict(X_poly_test)
#     # 模型评估
#     accuracy = accuracy_score(y_test, y_predict)
#
#     # 绘制拟合曲线
#     # plt.scatter(X_train, y_train, color='blue', label='训练数据', s=10)  # 原始训练数据
#     # plt.scatter(X_test, y_test, color='red', label='测试数据', s=10)  # 原始测试数据
#     return accuracy


if __name__ == '__main__':
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
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
    y = y.values.ravel()
    y = y.astype('int')

    # train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=10)
    plt.figure(figsize=(15, 10))
    for feature_index, feature_name in enumerate(X.columns):
        train_and_plot(X_train, X_test, y_train, y_test, feature_index, feature_name)

    plt.title('每个特征的逻辑回归拟合曲线')
    plt.xlabel('特征值')
    plt.ylabel('预测概率')
    plt.legend()
    plt.grid()
    plt.ylim(0.0, 1.0)
    plt.show()

    # accuracies = []
    # degrees = range(1,6)
    # for degree in degrees:
    #     acc = train(degree,X_train,X_test,y_train,y_test)
    #     print("参数量:",degree,", 准确率：", acc)
    #     accuracies.append(acc)
    #
    # plt.plot(degrees, accuracies, marker='o')
    # plt.title('参数量-准确率图像')
    # plt.xlabel('参数量（阶数）')
    # plt.ylabel('准确率')
    # plt.xticks(degrees)
    # plt.grid()
    # plt.ylim(0.9, 1)  # 设置y轴范围
    # plt.show()


