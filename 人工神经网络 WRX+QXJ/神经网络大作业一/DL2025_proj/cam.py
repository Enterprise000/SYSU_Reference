import os
import numpy as np
import pandas as pd
import torch
import cv2
import h5py

from analysis.analysis_tools import calculate_CAMs, save_heatmap

csv_path = 'analysis/result/v1.0_final/fold1.csv'
mid_feature_dir = 'analysis/mid_feature/v1.0_final/fold1'
fc_weight_path = 'analysis/result/v1.0_final/fold1_fc_weight.npy'
save_path = 'analysis/result/v1.0_final'
cam_save_dir = os.path.join(save_path, "cam")

os.makedirs(cam_save_dir, exist_ok=True)

df = pd.read_csv(csv_path)
fc_weight = np.load(fc_weight_path)

def load_hdf5_feature(path, key='feature_out'):
    with h5py.File(path, 'r') as f:
        return f[key][:]

for i in range(15,20):
    row = df.iloc[i]
    img_path = row['path']
    pred_label = int(row['pred'])

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    feature_path = os.path.join(mid_feature_dir, img_name)

    if not os.path.exists(feature_path):
        print(f"missing feature file: {feature_path}")
        continue
    features = load_hdf5_feature(feature_path, key='feature_out')
    if features.shape[0] != fc_weight.shape[1]:
        print("transform")
        features = np.transpose(features, (2, 0, 1))
    cams = calculate_CAMs(features, fc_weight, [pred_label])
    print("save camp")
    save_heatmap(cams, img_path, class_idx=0, cam_path=cam_save_dir)