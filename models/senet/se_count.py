import torch
from torch import nn

from models.senet.se_resnet import se_resnet152, se_resnet101


class Senet_Count(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = se_resnet101()
        self.fc = nn.Linear(2048, config['num_classes'])

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Senet_Mcan(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = se_resnet152()

        self.up_sample = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(100),
        )

    def forward(self, x):
        x = self.backbone(x)

        # up sample
        x = torch.transpose(x, 1, 3)
        x = self.up_sample(x).squeeze(2)
        return x
