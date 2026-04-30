import shutil
import os


# 这个代码文件用于管理机器学习模型训练过程中产生的检查点文件。
# 它帮助用户整理并删除旧的权重文件，以确保只保留一定数量的最新权重文件，避免磁盘空间被过多的旧权重文件占用。
# 通过对权重文件名的排序（根据文件名中的数字），它能够精确地识别和保留最新的检查点文件。
def get_weight_path(ckpt_path):
    if os.path.isdir(ckpt_path):
        # pth_list = os.listdir(ckpt_path)
        pth_list = [f for f in os.listdir(ckpt_path) if f.endswith('.pth')]
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x: int(x.split('-')[0].split('=')[-1]))
            return os.path.join(ckpt_path, pth_list[-1])
        else:
            return None
    else:
        return None


def get_weight_list(ckpt_path, choice=None):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if choice is not None and eval(str(fold.name)[-1]) not in choice:
            continue
        if fold.is_dir():
            # weight_path = os.listdir(fold.path)
            weight_path = [f for f in os.listdir(fold.path) if f.endswith('.pth')]
            # print(weight_path)
            if len(weight_path) > 0:
                weight_path.sort(key=lambda x: int(x.split('-')[0].split('=')[-1]))
                path_list.append(os.path.join(fold.path, weight_path[-1]))
    path_list.sort(key=lambda x: x.split('/')[-2])
    return path_list


def remove_weight_path(ckpt_path, retain=5):
    if os.path.isdir(ckpt_path):
        # pth_list = os.listdir(ckpt_path)
        pth_list = [f for f in os.listdir(ckpt_path) if f.endswith('.pth')]
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x: int(x.split('-')[0].split('=')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path, pth_item))


def dfs_remove_weight(ckpt_path, retain=5):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path, retain=retain)
        else:
            pth_files = [f for f in os.listdir(ckpt_path) if f.endswith('.pth')]
            if len(pth_files) > retain:
                remove_weight_path(ckpt_path, retain=retain)
            break


if __name__ == '__main__':
    ckpt_path = './ckpt/'
    dfs_remove_weight(ckpt_path)
