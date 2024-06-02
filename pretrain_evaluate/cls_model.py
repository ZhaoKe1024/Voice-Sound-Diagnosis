#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/5/23 12:14
# @Author: ZhaoKe
# @File : cls_model.py
# @Software: PyCharm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CLSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=256, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.cls1 = nn.Sequential()
        self.cls1.append(nn.Flatten(start_dim=1))
        self.cls1.append(nn.Linear(in_features=256, out_features=200))
        self.cls1.append(nn.ReLU())
        self.cls1.append(nn.Dropout(p=0.2))
        self.cls2 = nn.Sequential()
        self.cls2.append(nn.Linear(in_features=200, out_features=200))
        self.cls2.append(nn.ReLU())
        self.cls2.append(nn.Linear(in_features=200, out_features=4))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_wav):
        fm = self.conv1(inp_wav)
        # print(fm.shape)
        fm = self.maxpool1(self.relu1(fm))
        fm = self.conv2(fm)
        fm = self.maxpool2(self.relu2(fm))
        fm = self.conv3(fm)
        fm = self.maxpool3(self.relu3(fm))
        feat = self.cls1(fm)
        feat = self.cls2(feat)
        pred = self.softmax(feat)
        # print(pred.shape)
        return pred
        # print(fm)


if __name__ == '__main__':
    clm = CLSModel()
    clm(torch.rand(16, 8, 26))
