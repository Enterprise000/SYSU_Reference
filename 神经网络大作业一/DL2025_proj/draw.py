import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train_metrics.csv")

plt.plot(data["epoch"], data["train_acc"], label="Train Accuracy")
plt.plot(data["epoch"], data["train_loss"], label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("ResNet50 Overfitting Curve")
plt.legend()
plt.grid()
plt.show()