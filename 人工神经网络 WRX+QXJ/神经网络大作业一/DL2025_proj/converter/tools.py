import os
import glob
import pandas as pd
import random
from tqdm import tqdm

from common_utils import hdf5_reader

RULE = {"non-r0": 0, "r0": 1}


# 为数据集中的每个文件分配标签并生成包含路径和标签的CSV文件。
def make_label_csv(input_path, csv_path):
    '''
    Make label csv file.
    label rule: non-r0->0, r0->1
    '''
    info = []
    for subdir in os.scandir(input_path):
        index = RULE[subdir.name]
        path_list = glob.glob(os.path.join(subdir.path, "*.hdf5"))
        sub_info = [[item, index] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'label']
    random.shuffle(info)
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(csv_path, index=False)


# 统计每个HDF5文件中的切片数量并生成CSV文件记录，切片数指的就是每个HDF5文件中的图像切片的数量。
def statistic_slice_num(input_path, csv_path):
    '''
    Count the slice number for per sample.
    '''
    info = []
    for subdir in os.scandir(input_path):
        path_list = glob.glob(os.path.join(subdir.path, "*.hdf5"))
        sub_info = [[item, hdf5_reader(item, 'image').shape[0]] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'slice_num']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(csv_path, index=False)


# 生成标签文件和统计切片数文件，以便进一步分析或模型训练
# HDF5文件：HDF5是一种常用的文件格式，通常用于存储大规模的数据。
# 在医学影像中，HDF5文件常常用来存储从CT、MRI等扫描仪中获得的三维图像数据。
# 这些图像数据通常是由一系列的二维切片（即图像的“层”）构成的。
#
# 图像切片：当进行CT或MRI扫描时，仪器将身体的一个部位按一定厚度进行切分，得到多个平面图像，
# 这些图像就是切片。
# 每个切片是一个二维图像，代表身体某一层的横截面。
#
# 切片数：在一个HDF5文件中，通常会存储某个特定部位的多张切片，
# 这些切片组成了一个完整的三维影像。
# 切片数就是这个文件中包含的切片的数量，即二维图像的层数。
# 例如，如果一个文件中包含100张切片，那么这个文件的切片数就是100。
if __name__ == "__main__":
    # Part-1: make label csv file
    # os.makedirs('./csv_file')

    input_path = os.path.abspath('../dataset/npy_data/full_data')
    csv_path = './csv_file/full_index.csv'
    make_label_csv(input_path, csv_path)

    # Part-2: Count the slice number
    # input_path = os.path.abspath('../dataset/npy_data/')
    # csv_path = './csv_file/slice_number.csv'
    # statistic_slice_num(input_path,csv_path)

