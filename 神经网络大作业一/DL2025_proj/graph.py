import matplotlib.pyplot as plt
import pandas as pd

csv = pd.read_csv("ckpt/v1.0_task2-1_NONE/fold1/train_metrics.csv")

plt.plot(csv["epoch"], csv["train_acc"], label = "Train Accuracy")
plt.plot(csv["epoch"], csv["val_acc"], label = "Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Performance")
plt.title("ResNet50 Overfit Perfermance")
plt.legend()
plt.grid()
plt.savefig("log/stastic/ResNet50 Overfit Perfermance.png")
plt.show()