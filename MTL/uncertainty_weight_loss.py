# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class UncertaintyWeightLoss(nn.Module):

    def __init__(self, num=2):
        super(UncertaintyWeightLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


if __name__ == '__main__':

    awl = UncertaintyWeightLoss(7)
    print(awl.parameters())
