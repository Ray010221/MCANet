# !/usr/bin/env python
# coding=utf-8

#### crop size 256 x 256, stride 256
import os
import cv2

class SARImage_to_patch:
    def __init__(self, patch_size, sar_path, crop_sar_image_path):
        self.stride = patch_size
        self.sar_path = sar_path
        self.crop_sar_image_path = crop_sar_image_path

        if not os.path.exists(crop_sar_image_path):
            os.makedirs(crop_sar_image_path)
    def to_patch(self):
        # sar:  NH49E001013.tif
        # optical: NH49E001013.tif

        sar_files = sorted(os.listdir(self.sar_path))

        print(len(sar_files))

        for file_name in sar_files:
            prefix = file_name.split('.')[0]
            sar_path = os.path.join(self.sar_path, file_name)

            # read SAR image
            img_sar = cv2.imread(sar_path, cv2.IMREAD_UNCHANGED)
            h, w = img_sar.shape

            n_sar = 1
            # SAR image
            for x in range(0, h - self.stride, self.stride):
                for y in range(0, w - self.stride, self.stride):
                    sub_img_label = img_sar[x:x+self.stride, y:y+self.stride]
                    print('====> sar save', os.path.join(self.crop_sar_image_path,  prefix + '_' + str(n_sar) + '.tif'))
                    cv2.imwrite(os.path.join(self.crop_sar_image_path,  prefix + '_' + str(n_sar) + '.tif'), sub_img_label)
                    n_sar = n_sar + 1

if __name__ == '__main__':
    image_size = 256

    # sar_path = r'E:\Segmentation\whu-opt-sar\sar'
    # crop_sar_image_path = r'E:\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch-master\whu-opt-sar\sar'

    sar_path = '../dataset/sar'
    crop_sar_image_path = '../dataset/sars'

    # image to patch
    task = SARImage_to_patch(image_size, sar_path, crop_sar_image_path)
    task.to_patch()