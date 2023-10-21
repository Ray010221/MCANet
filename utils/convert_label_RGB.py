import cv2
import numpy
import numpy as np
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

color_map = {
    0: [0, 0, 0],  # 类别0对应黑色 backgroud
    10: [0, 102, 204],  # 类别1对应棕色 farmland
    20: [0, 0, 255],  # 类别2对应红色  city
    30: [0, 255, 255],  # 类别3对应黄色 village
    40: [255, 0, 0],  # 类别4对应蓝色  water
    50: [0, 167, 85],  # 类别5对应绿色   forest
    60: [255, 255, 0],  # 类别6对应靛蓝色  road
    70: [153, 102, 153]  # 类别7对应紫色  others
}

def convert_label_toRGB(label, new_lbl_root, file_name):
    # print(label)
    # 将单通道掩码图像转换为RGB图像
    h, w = label.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            key = label[i, j]
            rgb_image[i, j] = color_map[key]
            # print(key, rgb_image[i, j])
    # print(rgb_image)
    path = os.path.join(new_lbl_root, file_name)
    cv2.imwrite(path, rgb_image)
    print('=====> save', path)

if __name__ == '__main__':
    lbl_root = r'../dataset/lbl'
    new_lbl_root = r'../dataset/lblRGB'
    #
    if not os.path.exists(new_lbl_root):
        os.makedirs(new_lbl_root)

    label_files = sorted(os.listdir(lbl_root))
    print(len(label_files))

    for file_name in label_files:
        data = np.array(Image.open(os.path.join(lbl_root, file_name)))
        # print(data.shape)
        convert_label_toRGB(data, new_lbl_root, file_name)
    # array = np.full((100, 200), 20, dtype=np.uint8)
    # print(array)
    # convert_label_toRGB(array, new_lbl_root, 'test.png')

