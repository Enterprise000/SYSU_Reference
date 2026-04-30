import os
import glob
import pandas as pd


# 该函数的作用是从 CUB 200 2011 数据集的图像文件夹中生成一个 CSV 文件，记录每个图像的文件路径和其对应的标签。
def make_csv_cub(input_path, csv_path, index_path=None):
    '''
    Make CUB 200 2011 csv file.
    '''

    info = []
    for subdir in os.scandir(input_path):
        label = int(subdir.name.split('.')[0])
        path_list = glob.glob(os.path.join(subdir.path, "*.jpg"))
        sub_info = [[item, label] for item in path_list]
        info.extend(sub_info)

    col = ['id', 'label']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data['label'] = info_data['label'] - 1

    # if index_path is not None:
    #     index = pd.read_csv(index_path, header=None, sep=' ').loc[:,1]
    #     index = input_path + index
    #     info_data.index = info_data['id']
    #     info_data = info_data.loc[index]

    info_data.to_csv(csv_path, index=False)


# 该函数的作用是根据一个预定义的分割文件将 CUB 200 2011 数据集划分为训练集和测试集，并保存为两个独立的 CSV 文件。
def split_csv_cub(split_path, csv_path):
    '''
    Split CUB 200 2011 csv file.
    '''

    split = pd.read_csv(split_path, header=None, sep=' ').loc[:, 1]
    info_data = pd.read_csv(csv_path)

    train_data = info_data.loc[split == 1]
    test_data = info_data.loc[split == 0]

    train_data.to_csv(csv_path + '_train.csv', index=False)
    test_data.to_csv(csv_path + '_test.csv', index=False)


# 检查是否存在一个目录 ./csv_file，如果没有则创建它。
# 设置数据集的路径：input_path 是图像文件的路径，csv_path 是生成的 CSV 文件路径，index_path 是可选的图像索引路径。
# 调用 make_csv_cub 函数生成 CSV 文件。
# 设置分割文件路径：split_path 用于定义训练集和测试集的划分。
# 调用 split_csv_cub 函数生成训练集和测试集的 CSV 文件
if __name__ == "__main__":

    # make csv file
    if not os.path.exists('./csv_file'):
        os.makedirs('./csv_file')

    input_path = './datasets/CUB_200_2011/CUB_200_2011/images/'
    csv_path = './csv_file/cub_200_2011.csv'
    index_path = './datasets/CUB_200_2011/CUB_200_2011/images.txt'
    make_csv_cub(input_path, csv_path, index_path)

    # split csv file
    split_path = './datasets/CUB_200_2011/CUB_200_2011/train_test_split.txt'
    split_csv_cub(split_path, csv_path)



