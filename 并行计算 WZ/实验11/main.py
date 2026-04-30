from PIL import Image
import numpy as np


# 读取 .ppm 文件
def read_ppm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().decode()  # 读取 PPM 文件的头部
        if header.strip() != 'P6':
            raise ValueError("This is not a valid PPM P6 file.")

        # 跳过注释行
        while True:
            line = f.readline().decode()
            if line[0] != '#':  # 直到遇到非注释行
                width, height = map(int, line.split())
                break

        max_color_value = int(f.readline().decode())  # 读取最大颜色值（通常为255）

        # 读取图像数据
        img_data = np.frombuffer(f.read(), dtype=np.uint8)

        # 将数据重塑为图像形状 (height, width, 3) RGB
        img_data = img_data.reshape((height, width, 3))

        return img_data


image = Image.open('img.jpg')
image.save('img.ppm', 'PPM')
# 读取并转换为 NumPy 数组
image_data = read_ppm('image.ppm')
print(image_data)


