#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/2 20:48
# @Author: ZhaoKe
# @File : evaluate.py
# @Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import torch
from torch.utils.data import Dataset, DataLoader
from cls_model import CLSModel


class BaseDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, ind):
        return torch.from_numpy(self.datas[ind]), self.labels[ind]

    def __len__(self):
        return len(self.labels)


def heatmap_eval():
    cl_model = CLSModel()
    cl_model.load_state_dict(torch.load(f"./ckpt/cls_model_26.pt"))
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cl_model = cl_model.to(device)
    X_test, y_test = torch.load("./xtest.pt"), torch.load("./ytest.pt")
    testset = BaseDataset(X_test, y_test)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    ypred_eval = None
    ytrue_eval = None
    for jdx, (x_img, y_label) in enumerate(test_loader):
        X_mel = x_img.transpose(1, 2).to(device)
        # y_mel = y_label.to(device)
        # print(X_mel.shape)
        pred = cl_model(X_mel)
        # print(y_label.shape, pred.shape)
        if jdx == 0:
            ytrue_eval = y_label
            ypred_eval = pred
        else:
            ytrue_eval = torch.concat((ytrue_eval, y_label), dim=0)
            ypred_eval = torch.concat((ypred_eval, pred), dim=0)
        print(ytrue_eval.shape, ypred_eval.shape)
    ytrue_eval = ytrue_eval.argmax(-1).data.cpu().numpy()
    ypred_eval = ypred_eval.argmax(-1).data.cpu().numpy()
    print(ytrue_eval.shape, ypred_eval.shape)

    # def get_heat_map(pred_matrix, label_vec, savepath):
    savepath = "./result_hm.png"
    max_arg = list(ypred_eval)
    conf_mat = metrics.confusion_matrix(max_arg, ytrue_eval)
    conf_mat = conf_mat / conf_mat.sum(axis=1)
    df_cm = pd.DataFrame(conf_mat, index=["Healthy", "Hyperkinetic\nDysphonia", "Hyperkinetic\nDysphonia", "Reflux\nLaryngitis"], columns=["Healthy", "Hyperkinetic\nDysphonia", "Hyperkinetic\nDysphonia", "Reflux\nLaryngitis"])
    heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='YlGnBu')  # , cbar_kws={'format': '%.2f%'})
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    heatmap_eval()
