import torch
import torch.nn as nn


class CountLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.CrossEntropyLoss()
        self.l2 = nn.BCELoss(reduction='sum')

    def forward(self, x1, x2, y1, y2):
        loss1 = self.l1(x1, y1)
        loss2 = self.l2(x2, y2)

        all_loss = self.alpha * loss1 + self.beta * loss2
        return all_loss, loss1, loss2
