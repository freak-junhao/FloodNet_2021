import models.resnet as resnet_
from models.senet.se_count import Senet_Mcan
from models.mcan.net import Net
from models.vit_models.vit import ViT
from models.efficientnet_pytorch.model import EfficientNet

import torch
import torch.nn as nn


class vit_extracter(nn.Module):
    def __init__(self, config):
        super(vit_extracter, self).__init__()
        self.backbone = ViT(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            num_classes=config['classes'],
            dim=config['dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            dropout=config['dropout'],
            emb_dropout=config['emb_dropout']
        )

        in_channels = (config['image_size'] // config['patch_size']) ** 2
        self.up_conv1d = nn.Sequential(
            nn.Conv1d(in_channels, 100, 1),
            nn.LayerNorm(config['dim']),
        )

    def forward(self, x):
        _, x = self.backbone(x)
        x = self.up_conv1d(x)
        return x


class ef_extracter(nn.Module):
    def __init__(self, model_name):
        super(ef_extracter, self).__init__()

        self.ef_net = EfficientNet.from_pretrained(model_name)

        # self define
        self.up_sample_2048 = nn.Sequential(
            nn.Conv2d(1792, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.up_sample_100 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(100),
        )

    def forward(self, x):
        x = self.ef_net(x)
        x = self.up_sample_2048(x)
        x = torch.transpose(x, 1, 3)
        x = self.up_sample_100(x).squeeze(2)

        return x


class VqaNet(nn.Module):
    def __init__(self, config, pretrained_emb, token_size, answer_size):
        super(VqaNet, self).__init__()
        self.config = config

        # image feature extract
        if config['feature_model'] == 'resnet101':
            self.image_net = resnet_.resnet101(pretrained=True)
        elif config['feature_model'] == 'se_resnet152':
            self.image_net = Senet_Mcan(config)
        elif config['feature_model'] == 'vit':
            self.image_net = vit_extracter(config)
        elif 'efficientnet' in config['feature_model']:
            self.image_net = ef_extracter(config['feature_model'])
        else:
            self.image_net = resnet_.resnext101_32x8d(pretrained=True)

        # backbone
        self.backbone = Net(config, pretrained_emb, token_size, answer_size)

    def forward(self, x, ques_ix):
        if not self.config['use_npz']:
            image_feat = self.image_net(x)
        else:
            image_feat = x

        out = self.backbone(image_feat, ques_ix)
        return out


if __name__ == '__main__':
    data = torch.randn((3, 224, 224))
    # net = VqaNet()
