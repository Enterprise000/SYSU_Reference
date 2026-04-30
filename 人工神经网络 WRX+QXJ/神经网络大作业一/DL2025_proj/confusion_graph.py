import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from analysis.analysis_tools import plot_confusion_matrix

csv = 'analysis/result/v1.0_final/fold1.csv'
df = pd.read_csv(csv)
y_true = df['true']
y_pred = df['pred']

class_list = 'datasets/CUB_200_2011/CUB_200_2011/classes.txt'
label = {}
with open(class_list, 'r') as f:
    for line in f:
        parts = line.strip().split()
        index = int(parts[0])
        name = ' '.join(parts[1:])
        label[index] = name
label_name = [label[i] for i in sorted(label)]

matrix = confusion_matrix(y_true, y_pred, labels=sorted(label.keys()))

save = 'analysis/result/v1.0_final'
os.makedirs(save, exist_ok=True)
plot_confusion_matrix(matrix, label_name, title="Confusion Matrix Final", save_path=save)