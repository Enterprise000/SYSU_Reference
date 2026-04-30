import tensorflow as tf
import time

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 检测加载是否成功
print(x_train.shape, y_train.shape)

start = time.time()
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 3000  # 每批次大小
for i in range(0, 12000, batch_size):
    batch = x_train[i:i + batch_size]
    batch_y = y_train[i:i + batch_size]

    # 将批次数据调整形状为 (batch_size, 28, 28, 1)
    batch = tf.expand_dims(batch, axis=-1)
    # 调整批次数据大小为 224x224
    batch_resized = tf.image.resize(batch, [224, 224])

    # 如果需要将灰度图像转换为 RGB
    batch_rgb = tf.image.grayscale_to_rgb(batch_resized)

    print(batch_rgb.shape)  # 输出每批次的形状
    print(i)

    model.fit(batch_rgb, batch_y, epochs=2)


batch_test = x_test[0:3000]
batch_test_y = y_test[0:3000]

batch_test = tf.expand_dims(batch_test, axis=-1)

batch_test_resized = tf.image.resize(batch_test, [224, 224])

batch_test_rgb = tf.image.grayscale_to_rgb(batch_test_resized)

test_loss, test_accuracy = model.evaluate(batch_test_rgb, batch_test_y)

end = time.time()
print("Test Accuracy: ", test_accuracy)
print("runtime: ", end - start)
