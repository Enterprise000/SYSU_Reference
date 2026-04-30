import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from analysis.analysis_tools import plot_roc_curve

csv = 'analysis/result/v1.0_final/fold1.csv'
df = pd.read_csv(csv)

y_true = df['true'].values
y_prob = np.stack(df['prob'].apply(eval).values)

label_map = {}
with open('datasets/CUB_200_2011/CUB_200_2011/classes.txt', 'r') as f:
    for line in f:
        idx, name = line.strip().split()
        label_map[int(idx)] = name
labels_name = [label_map[i] for i in sorted(label_map)]

save_path = 'analysis/result/v1.0_final'
os.makedirs(save_path, exist_ok=True)

plot_roc_curve(true=y_true, prob=y_prob, labels_name=labels_name, title='ROC final', save_path=save_path)