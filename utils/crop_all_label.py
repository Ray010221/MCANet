# !/usr/bin/env python
# coding=utf-8

#### crop size 256 x 256, stride 256
import os
import cv2

class LblImage_to_patch:
    def __init__(self, patch_size, lbl_path, crop_lbl_image_path):
        self.stride = patch_size

        self.lbl_path = lbl_path
        self.crop_lbl_image_path = crop_lbl_image_path

        if not os.path.exists(crop_lbl_image_path):
            os.makedirs(crop_lbl_image_path)
    def to_patch(self):
        lbl_files = sorted(os.listdir(self.lbl_path))
        print(len(lbl_files))

        for file_name in lbl_files:
            prefix = file_name.split('.')[0]
            lbl_path = os.path.join(self.lbl_path, file_name)

            # read lbl image
            img_lbl = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
            # h, w = img_lbl.shape
            h, w, c =img_lbl.shape
            n_lbl = 1
            # SAR image
            for x in range(0, h - self.stride, self.stride):
                for y in range(0, w - self.stride, self.stride):
                    sub_img_label = img_lbl[x : x + self.stride, y : y + self.stride]
                    print('====> sar save', os.path.join(self.crop_lbl_image_path,  prefix + '_' + str(n_lbl) + '.tif'))
                    cv2.imwrite(os.path.join(self.crop_lbl_image_path,  prefix + '_' + str(n_lbl) + '.tif'), sub_img_label)
                    n_lbl = n_lbl + 1

if __name__ == '__main__':
    image_size = 256

    # lbl_path = 'dataset/orignlbl'
    # crop_lbl_image_path = 'dataset/lbls'
    # lbl_path = '../dataset/orignlbl'
    # crop_lbl_image_path = '../dataset/lbls'
    lbl_path = '../dataset/lblRGB'
    crop_lbl_image_path = '../dataset/lblRGBs'

    # image to patch
    task = LblImage_to_patch(image_size, lbl_path, crop_lbl_image_path) # top 10 labels
    task.to_patch()