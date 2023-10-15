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
    model.load_state_dict(torch.load('model.pth'))
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
    print(res)
    color_label(res.numpy().squeeze())

    # print("predicted label is {}, {} {} 8".format(res, val.item(), ('>' if res == 1 else '<')))

color_map = {
    0: [0, 0, 0],       # 类别0对应黑色
    1: [255, 0, 0],     # 类别1对应红色
    2: [0, 255, 0],     # 类别2对应绿色
    3: [0, 0, 255],     # 类别3对应蓝色
    4: [255, 255, 0],   # 类别4对应黄色
    5: [255, 0, 255],   # 类别5对应品红
    6: [0, 255, 255],   # 类别6对应青色
    7: [128, 128, 128]  # 类别7对应灰色
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
    cv2.imwrite('predict.jpg',rgb_image)
    # 使用OpenCV显示RGB图像
    cv2.imshow("Semantic Segmentation Result", rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = Args()
    img_sar_path = r'D:\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch-master\whu-opt-sar\test\sar\NH49E001013_12.tif'
    img_opt_path = r'D:\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch-master\whu-opt-sar\test\opt\NH49E001013_12.tif'
    pred(img_sar_path, img_opt_path)