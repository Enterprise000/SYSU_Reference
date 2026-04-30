import sys
import cv2
from torch.utils.data import Dataset
import torch


# 从一个图像路径列表中读取图像文件。
# 通过对应的标签字典提供每个图像的标签。
# 可选地对图像进行数据增强或预处理。
# 返回一个包含图像和标签的字典。

class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - label_dict: dict, file path as key, label as value
    - transform: the data augmentation methods
    '''

    def __init__(self, path_list, label_dict, transform=None):
        self.path_list = path_list
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        # Get image and label
        # image: C,H,W
        # label: integer, 0,1,..
        image = cv2.imread(self.path_list[index])

        # Transform
        if self.transform is not None:
            image = self.transform(image)

        label = self.label_dict[self.path_list[index]]
        sample = {'image': image, 'label': int(label)}

        return sample

