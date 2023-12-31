import cv2

from model.MCANet import MACANet
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
class Args:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()   # 将图像转换为张量
])
def pred(img_sar_path, img_opt_path):
    model = MACANet(pretrained=False)
    model.load_state_dict(torch.load('weight/50-16-SGD-model.pth', map_location=args.device))
    model.eval()

    img_sar = Image.open(img_sar_path)
    img_opt = Image.open(img_opt_path)
    img_sar = transform(img_sar).unsqueeze(0)
    img_opt = transform(img_opt).unsqueeze(0)
    print(img_sar.shape, img_opt.shape)
    res = model(img_sar, img_opt)

    res = torch.argmax(res, dim=1)
    # print(res.shape)
    # pred = res.cpu().numpy().squeeze().astype(np.uint8)
    # print(res)
    color_label(res.numpy().squeeze())

    # print("predicted label is {}, {} {} 8".format(res, val.item(), ('>' if res == 1 else '<')))

color_map = {
    0: [0, 0, 0],  # 类别0对应黑色 backgroud
    1: [0, 102, 204],  # 类别1对应棕色 farmland
    2: [0, 0, 255],  # 类别2对应红色  city
    3: [0, 255, 255],  # 类别3对应黄色 village
    4: [255, 0, 0],  # 类别4对应蓝色  water
    5: [0, 167, 85],  # 类别5对应绿色   forest
    6: [255, 255, 0],  # 类别6对应靛蓝色  road
    7: [153, 102, 153]  # 类别7对应紫色  others
}
def color_label(label):
    print(label)
    # 将单通道掩码图像转换为RGB图像
    h, w = label.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            key = label[i, j]
            rgb_image[i, j] = color_map[key]
    cv2.imwrite('predict/predict12_1.jpg', rgb_image)
    # 使用OpenCV显示RGB图像
    cv2.imshow("Semantic Segmentation Result", rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = Args()
    img_sar_path = r'dataset\test\sar\NH49E001013_12.tif'
    img_opt_path = r'dataset\test\opt\NH49E001013_12.tif'
    pred(img_sar_path, img_opt_path)