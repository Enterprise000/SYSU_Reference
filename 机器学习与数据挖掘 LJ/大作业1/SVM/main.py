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
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import scipy


# matplotlib画图中中文显示会有问题，需要这两行设置默认字体可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# class SVM:
#     def __init__(self, C=0.1, tol=1e-3, max_passes=3):
#         self.C = C          # 正则化参数
#         self.tol = tol      # 容忍度
#         self.max_passes = max_passes  # 最大遍历次数
#         self.alpha = None   # Lagrange 乘子
#         self.b = 0          # 偏置
#         self.X = None       # 输入特征
#         self.y = None       # 标签
#
#     def fit(self, X, y):
#         self.X = X
#         self.y = y
#         m, n = X.shape
#         self.alpha = np.zeros(m)
#
#         passes = 0
#         while passes < self.max_passes:
#             print(passes)
#             num_changed_alphas = 0
#             for i in range(m):
#                 # 计算预测值
#                 f_xi = np.dot((self.alpha * self.y), self.kernel(X.to_numpy(), X.iloc[i].values)) + self.b
#                 # 得到误差
#                 E_i = f_xi - self.y[i]
#
#                 # 如果当前样本的预测存在较大误差且其权重尚未达到上限，或者样本被过度分类且其权重尚未降到 0，就需要进行更新
#                 if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (self.y[i] * E_i > self.tol and self.alpha[i] > 0):
#                     # 选择第二个乘子 j
#                     j = self.select_j(i, m)
#                     # 得到样本j的预测值
#                     f_xj = np.dot((self.alpha * self.y), self.kernel(X.to_numpy(), X.iloc[j].values)) + self.b
#                     # 得到样本j的误差
#                     E_j = f_xj - self.y[j]
#
#                     # 保存旧的乘子值
#                     alpha_i_old = self.alpha[i]
#                     alpha_j_old = self.alpha[j]
#
#                     # 根据两个乘子的类别选定更新上下界
#                     if self.y[i] != self.y[j]:
#                         L = max(0, alpha_j_old - alpha_i_old)
#                         H = min(self.C, self.C + alpha_j_old - alpha_i_old)
#                     else:
#                         L = max(0, alpha_i_old + alpha_j_old - self.C)
#                         H = min(self.C, alpha_i_old + alpha_j_old)
#
#                     if L == H:
#                         continue
#
#                     # 计算两个样本点之间的相似度
#                     eta = self.kernel(X.iloc[i].values, X.iloc[i].values) + self.kernel(X.iloc[j].values, X.iloc[j].values) - 2 * self.kernel(X.iloc[i].values, X.iloc[j].values)
#                     if eta <= 0:
#                         continue
#
#                     # 更新乘子 j
#                     self.alpha[j] += self.y[j] * (E_i - E_j) / eta
#                     self.alpha[j] = np.clip(self.alpha[j], L, H)
#
#                     # 更新乘子 i
#                     self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
#
#                     # 更新偏置 b
#                     b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X.iloc[i].values, X.iloc[i].values) - self.y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X.iloc[i].values, X.iloc[j].values)
#                     b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X.iloc[i].values, X.iloc[j].values) - self.y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X.iloc[j].values,X.iloc[j].values)
#                     if 0 < self.alpha[i] < self.C:
#                         self.b = b1
#                     elif 0 < self.alpha[j] < self.C:
#                         self.b = b2
#                     else:
#                         self.b = (b1 + b2) / 2
#                     # print("b:",self.b)
#
#                     num_changed_alphas += 1
#             if num_changed_alphas == 0:
#                 print("pass: ", passes)
#                 passes += 1
#             else:
#                 passes = 0
#
#     def kernel(self, x1, x2):
#         # 线性核函数
#         return np.dot(x1, x2.T)
#
#     def select_j(self, i, m):
#         j = i
#         while j == i:
#             j = np.random.randint(0, m)
#         return j
#
#     def predict(self, X):
#         return np.sign(np.dot((self.alpha * self.y), self.kernel(self.X.to_numpy(), X.to_numpy())) + self.b)


# Press the green button in the gutter to run the script.
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
    y = y.values.ravel()
    y = y.astype('int')
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # svm = SVM(C = 0.1)
    # svm.fit(x_train, y_train)
    # result = svm.predict(x_test)
    predictor = svm.SVC(C=100.0, decision_function_shape='ovo', kernel='rbf')
    # 进行训练
    predictor.fit(x_train, y_train)
    # 预测结果
    result = predictor.predict(x_test)
    # 进行评估
    acc = accuracy_score(y_test, result)
    cm = confusion_matrix(y_test, result)
    TP = cm[1, 1]  # 真正类
    TN = cm[0, 0]  # 真负类
    FP = cm[0, 1]  # 假正类
    FN = cm[1, 0]  # 假负类
    precision = precision_score(y_test, result, average='weighted')
    recall = recall_score(y_test, result, average='weighted')
    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)
    print("Precision: ",precision)
    print("Recall: ", recall)
    print("F-score: {0:.2f}".format(f1_score(result, y_test, average='micro')))
    print("Accuracy:", acc)

