import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataSet import WHU_OPT_SARDataset
from model.MCANet import MACANet
import torch
import numpy as np
class Args:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8

def test():
    test_dataset = WHU_OPT_SARDataset(class_name='whu-sar-opt',
                                      root='dataset/test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    model = MACANet(pretrained=False)
    model.load_state_dict(torch.load('weight/50-16-Adam-model.pth', map_location=args.device))
    model = model.to(args.device)
    model.eval()

    acc = 0
    nums = 0
    for idx, (sar, opt, label) in enumerate(tqdm(test_dataloader)):
        sar = sar.to(args.device)  # .to(torch.float)
        opt = opt.to(args.device)
        label = label.to(args.device)
        outputs = model(sar, opt)
        final_class = torch.argmax(outputs, dim=1)
        final_class.to(args.device)
        label = label.long()

        acc += torch.sum(final_class == label).item()
        nums += label.size()[0] * label.size()[1] * label.size()[2]

    print("test OA = {:.3f}%".format(100 * acc / nums))


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
    test()