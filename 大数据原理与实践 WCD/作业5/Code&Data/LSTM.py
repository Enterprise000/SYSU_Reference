import pandas as pd
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error

# 数据处理
columns = ['Date','Close','High','Low','Open','Volume']
data = pd.read_csv(f"MSFT_data.csv", skiprows=3, names=columns,index_col="Date",parse_dates=True)
data = data["Close"]
data.ffill(inplace=True)
# 去掉最后七行的数据
data_train = data[:-7]
data_train = data_train.values.reshape(-1,1)
print(data_train.shape)
print(data_train[:5])
# 只有最后七行的数据
data_true = data[-7:]
time_step = 30
X = []
y = []
# 按照时间步分组，X是时间步内的股票价格，y是要预测的1个值
for i in range(time_step,len(data_train)):
    X.append(data_train[i-time_step:i,0])
    y.append(data_train[i,0])
X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0],X.shape[1],1)
print(X.shape)
print(y.shape)

# 训练模型
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.LSTM(units=100,return_sequences=True, input_shape=(X.shape[1],1)))
model.add(tensorflow.keras.layers.Dense(units=1))
optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.3)
model.compile(optimizer=optimizer,loss='mean_squared_error')
model.summary()
model.fit(X,y,epochs=700,batch_size=100)

# 进行预测
# 获取砍掉最后7天之后，最后30天的数据
data_test = X[-30:]
print(data_test.shape)
predict_price = model.predict(data_test)
predict_price = predict_price[0][:7]
print(type(predict_price))
print("predict price 7:", predict_price)
mae = mean_absolute_error(data_true.values,predict_price)
rmse = np.sqrt(mean_squared_error(data_true.values,predict_price))
print(f"MAE:{mae}")
print(f"RMSE:{rmse}")
