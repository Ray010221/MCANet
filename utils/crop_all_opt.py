#### crop size 256 x 256, stride 256
import os
import cv2

class OptImage_to_patch:
    def __init__(self, patch_size, sar_path, crop_sar_image_path, optical_path, crop_optical_image_path):
        self.stride = patch_size
        self.sar_path = sar_path
        self.optical_path = optical_path
        self.crop_sar_image_path = crop_sar_image_path
        self.crop_optical_image_path = crop_optical_image_path

        if not os.path.exists(crop_sar_image_path):
            os.makedirs(crop_sar_image_path)

        if not os.path.exists(crop_optical_image_path):
            os.makedirs(crop_optical_image_path)

    def to_patch(self):
        sar_files = sorted(os.listdir(self.sar_path))
        optical_files = sorted(os.listdir(self.optical_path))

        print(len(sar_files), len(optical_files))

        for file_name in sar_files:
            prefix = file_name.split('.')[0]
            optical_path = os.path.join(self.optical_path, file_name)

            # read Optical image
            img_optical = cv2.imread(optical_path, cv2.IMREAD_UNCHANGED)
            h, w, c = img_optical.shape
            # print(img_optical.shape)

            n_optical = 1
            # OPTICAL image
            for x in range(0, h - self.stride, self.stride):
                for y in range(0, w - self.stride, self.stride):
                    sub_img_label = img_optical[x : x + self.stride, y : y + self.stride]
                    print('====> optical save', os.path.join(self.crop_optical_image_path, prefix + '_' + str(n_optical) + '.tif'))
                    cv2.imwrite(os.path.join(self.crop_optical_image_path, prefix + '_' + str(n_optical) + '.tif'), sub_img_label)
                    n_optical = n_optical + 1

if __name__ == '__main__':
    image_size = 256

    sar_path = '../dataset/sar'
    crop_sar_image_path = '../dataset/sars'

    optical_path = '../dataset/optical'
    crop_optical_image_path = '../dataset/opticals'

    # image to patch
    task = OptImage_to_patch(image_size, sar_path, crop_sar_image_path, optical_path, crop_optical_image_path) # top 10 images
    task.to_patch()