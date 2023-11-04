import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MACAM import MCAM
from model.ASPP import ASPP
from model.ResNet import resnet101


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MACANet(nn.Module):
    def __init__(self, num_classes = 1000, pretrained = True, backbone = 'ResNet101', att_type=None):
        super(MACANet, self).__init__()
        # print(num_classes)
        self.encoder = EncoderBlock(pretrained, backbone, att_type=att_type)
        self.decoder = DecoderBlock(num_classes)


    def forward(self, sar_img, opt_img):
        opt_sar_low_high_features = self.encoder.forward(sar_img, opt_img)
        classification = self.decoder(opt_sar_low_high_features)

        return classification
class EncoderBlock(nn.Module):
    def __init__(self, pretrained = True, backbone = 'ResNet101', num_classes=1000, att_type=None):
        super(EncoderBlock, self).__init__()
        if backbone == 'ResNet101':
            self.SAR_resnet = resnet101(pretrained, type='sar', num_classes=num_classes, att_type=att_type)
            self.OPT_resnet = resnet101(pretrained, type='opt', num_classes=num_classes, att_type=att_type)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use ResNet101.'.format(backbone))

        self.MCAM_low = MCAM(in_channels=256)
        self.MCAM_high = MCAM(in_channels=2048)
        self.ASPP = ASPP(in_channels=2560, atrous_rates=[6, 12, 18])
        self.conv1 = conv1x1(2048, 256)
        self.conv2 = conv1x1(768, 48)

    def forward(self, sar_img, opt_img):
        sar_feats = self.SAR_resnet.forward(sar_img)
        opt_feats = self.OPT_resnet.forward(opt_img)

        sar_low_feat = sar_feats[1]
        sar_high_feat = sar_feats[4]
        sar_final_feat = self.conv1(sar_feats[4])
        opt_low_feat = opt_feats[1]
        opt_high_feat = opt_feats[4]
        opt_final_feat = self.conv1(opt_feats[4])

        low_level_features = self.MCAM_low(sar_low_feat, opt_low_feat)
        high_level_features = self.MCAM_high(sar_high_feat, opt_high_feat)

        low_level_sar_opt = torch.cat([sar_low_feat, opt_low_feat], 1)
        high_level_sar_opt = torch.cat([sar_final_feat, opt_final_feat], 1)

        low_sar_opt_features = torch.cat([low_level_sar_opt, low_level_features], 1)
        high_sar_opt_features = torch.cat([high_level_sar_opt, high_level_features], 1)

        low_sar_opt_features = self.conv2(low_sar_opt_features)
        high_sar_opt_features = self.ASPP(high_sar_opt_features)
        high_sar_opt_features = F.interpolate(high_sar_opt_features, size=(64, 64), mode='bilinear', align_corners=False)

        opt_sar_low_high_features = torch.cat([high_sar_opt_features, low_sar_opt_features], 1)

        return opt_sar_low_high_features
class DecoderBlock(nn.Module):
    def __init__(self, num_class):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, num_class, kernel_size=1)
        )
    def forward(self, opt_sar_low_high_features):
        final_class = self.conv(opt_sar_low_high_features)
        final_img = F.interpolate(final_class, size=(256, 256), mode='bilinear', align_corners=False)

        return final_img