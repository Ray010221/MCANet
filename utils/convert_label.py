import numpy as np
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def old_label2new_label(lbl, root_path, name):
    lbl[np.where(lbl == 0)] = 0
    lbl[np.where(lbl == 10)] = 1 # farmland  (10,10,10)
    lbl[np.where(lbl == 20)] = 2 # city  (20,20,20)
    lbl[np.where(lbl == 30)] = 3 # village  (30,30,30)
    lbl[np.where(lbl == 40)] = 4 # water  (40,40,40)
    lbl[np.where(lbl == 50)] = 5 # forest  (50,50,50)
    lbl[np.where(lbl == 60)] = 6 # road  (60,60,60)
    lbl[np.where(lbl == 70)] = 7 # others (70,70,70)

    path = os.path.join(root_path, name)
    print('=====> save', path)
    Image.fromarray(np.uint8(lbl[:, :])).convert('L').save(path)

    return

if __name__ == '__main__':


    lbl_root = '../dataset/lbl'
    new_lbl_root = '../dataset/orignlbl'

    if not os.path.exists(new_lbl_root):
        os.makedirs(new_lbl_root)

    label_files = sorted(os.listdir(lbl_root))
    print(len(label_files))

    for file_name in label_files:
        data = np.array(Image.open(os.path.join(lbl_root, file_name)))
        old_label2new_label(data, new_lbl_root, file_name)



