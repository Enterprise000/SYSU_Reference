import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 检测加载是否成功
print(x_train.shape, y_train.shape)

start = time.time()
# 图像处理
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_flat)
x_test = scaler.fit_transform(x_test_flat)

# 训练
svm = svm.SVC(kernel='rbf', gamma='scale')
svm.fit(x_train, y_train)

# 测试
y_predict = svm.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
end = time.time()
print("accuracy: ", accuracy)
print("runtime: ", end - start)
