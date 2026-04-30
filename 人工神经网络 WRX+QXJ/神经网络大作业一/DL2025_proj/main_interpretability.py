import os
import numpy as np
import data_utils.transform as tr
from config import INIT_TRAINER
from torchvision import transforms
from converter.common_utils import hdf5_reader
from analysis.analysis_tools import calculate_CAMs, save_heatmap

# 核心作用是通过加载图像的特征、权重和图像数据，
# 利用类激活映射（CAM）技术可视化深度学习模型（特别是 CNN）在做出分类决策时，关注图像的哪些区域。
# 通过这种方式，可以帮助研究者理解模型的决策过程，特别是当模型做出某个类别预测时，模型“看到”了图像的哪些部分。
# 最终生成的热力图将被保存到指定的路径。

features = hdf5_reader('./analysis/mid_feature/v1.0/fold1/Black_Footed_Albatross_0036_796127','feature_in')
# 默认的hook获取池化层的输入和输出，池化层的输入'feature_in'即为最后一层卷积层的输出
weight = np.load('./analysis/result/v1.0/fold1_fc_weight.npy')
# 线性层的权重
img_path = './datasets/CUB_200_2011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0036_796127.jpg'
# 对应的原始图像路径

transformer = transforms.Compose([
    tr.ToCVImage(),
    tr.RandomResizedCrop(size=INIT_TRAINER['image_size'], scale=(1.0, 1.0)),
    tr.ToTensor(),
    tr.Normalize(INIT_TRAINER['train_mean'], INIT_TRAINER['train_std']),
    tr.ToArray(),
])

classes = 200 # 总类别数
class_idx = 0 # 模型预测类别，也可以从最终结果的csv里面批量读取
cam_path = './analysis/result/v1.0/'
cams = calculate_CAMs(features, weight, range(classes))
save_heatmap(cams, img_path, class_idx, cam_path, transform=transformer)
