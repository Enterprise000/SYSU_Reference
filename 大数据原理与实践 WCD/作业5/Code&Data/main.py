import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error,mean_squared_error

# 获取数据
# data = yf.download("MSFT", period="1y", interval="1d")
# data.to_csv(f"MSFT_data.csv")
# data.head()

# 数据处理
columns = ['Date','Close','High','Low','Open','Volume']
data = pd.read_csv(f"MSFT_data.csv", skiprows=2, names=columns,index_col="Date",parse_dates=True)
data = data["Close"]
data.ffill(inplace=True)
data_test = data[:-7]
data_true = data[-7:]

# 使用ARIMA模型，预测
model = ARIMA(data_test, order=(30,2,10))
model_fit = model.fit()
forward_step = 7
predict = model_fit.forecast(steps=forward_step)

# 打印预测结果
last_date = data_test.index[-1]
print(f"last date in train set:{last_date}")
print(f"actual stock price:{data_true}")
print(f"predicted stock price:{predict}")

# 绘图
index = pd.date_range(start=last_date ,periods=forward_step + 1,freq='D')[1:]
value = pd.Series(predict.values,index=index)
y_true = data_true.values
y_predict = predict.values
mae = mean_absolute_error(y_true,y_predict)
rmse = np.sqrt(mean_squared_error(y_true,y_predict))
print(f"MAE:{mae}")
print(f"RMSE:{rmse}")
data.plot(title="Price")
plt.show()
value.plot(title="Predicted Price")
plt.show()


