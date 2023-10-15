import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataSet import WHU_OPT_SARDataset
from model.MCANet import MACANet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 设置随机数种子保证论文可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# 以类的方式定义参数，还有很多方法，config文件等等
class Args:
    def __init__(self) -> None:
        self.batch_size = 4
        self.lr = 0.001
        self.epochs = 20
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:3")



def train():
    # train_dataset = WHU_OPT_SARDataset(class_name='whu-sar-opt', root=r'D:\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch-master\whu-opt-sar\train')
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_dataset = WHU_OPT_SARDataset(class_name='whu-sar-opt', root=r'D:\High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch-master\whu-opt-sar\val')
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    train_dataset = WHU_OPT_SARDataset(class_name='whu-sar-opt',
                                       root='dataset/train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = WHU_OPT_SARDataset(class_name='whu-sar-opt',
                                     root='dataset/val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    model = MACANet().to(args.device)
    # print(model)
    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        # =========================train=======================
        for idx, (sar, opt, label) in enumerate(tqdm(train_dataloader)):
            sar = sar.to(args.device)
            opt = opt.to(args.device)
            label = label.to(args.device)
            outputs = model(sar, opt)
            final_class = torch.argmax(outputs, dim=1)
            final_class.to(args.device)
            # output_max, preds = torch.max(outputs, 1)
            # print(outputs.shape, label.shape)
            # print(output_max.shape, preds.shape)
            # output_max = output_max.data.cpu().numpy().squeeze().astype(np.uint8)

            # label = label.long()
            # outputs = outputs.to(args.device)
            # print(output_max.shape)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            # print(loss)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) #用来梯度裁剪
            optimizer.step()
            train_epoch_loss.append(loss.item())
            # acc += sum(outputs.max(axis=1)[1] == label).cpu()
            # acc += torch.sum(outputs.max(axis=1)[1] == label).item()
            acc += torch.sum(final_class == label).item()
            nums += label.size()[0] * label.size()[1] * label.size()[2]
            # print(label.size())
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}%, loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))
        # =========================val=========================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0

            for idx, (sar, opt, label) in enumerate(tqdm(val_dataloader)):
                sar = sar.to(args.device)  # .to(torch.float)
                opt = opt.to(args.device)
                label = label.to(args.device)
                outputs = model(sar, opt)
                final_class = torch.argmax(outputs, dim=1)
                final_class.to(args.device)
                # output_max, preds = torch.max(outputs, 1)
                # preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                label = label.long()

                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())

                # acc += sum(outputs.max(axis=1)[1] == label).cpu()
                acc += torch.sum(final_class == label).item()
                nums += label.size()[0] * label.size()[1] * label.size()[2]
                # print(acc, nums)

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)

            print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(epoch, 100 * acc / nums, np.average(val_epoch_loss)))

    # =========================plot==========================
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_epochs_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss, '-o', label="train_loss")
    plt.plot(valid_epochs_loss, '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.show()
    # =========================save model=====================
    torch.save(model.state_dict(), 'model.pth')



if __name__ == '__main__':
    args = Args()
    train()