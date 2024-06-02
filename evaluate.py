#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/2 20:48
# @Author: ZhaoKe
# @File : evaluate.py
# @Software: PyCharm
import torch
from cls_model import CLSModel




def heatmap_eval():
    model = CLSModel()
    model.load_state_dict(torch.load(f"./ckpt/cls_model_40.pt"))

    X_test, y_test = torch.load("./xtest.pt"), torch.load("./ytest.pt")




