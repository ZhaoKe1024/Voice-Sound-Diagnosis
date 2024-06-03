#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/2 20:48
# @Author: ZhaoKe
# @File : evaluate.py
# @Software: PyCharm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sklearn.metrics as metrics
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from cls_model import CLSModel

WAV_LENGTH = 3872


def count_num():
    cnts = [0, 0, 0, 0]
    with open("./dataset.txt", 'r') as fin:
        lines = fin.readlines()
        print(len(lines))
        for line in tqdm(lines):
            parts = line.strip().split(',')
            path, lab = parts[0], int(parts[1])
            # mfccs.append(torch.from_numpy(path2mfcc(path)))
            # labels.append(int(parts[1]))
            cnts[lab] += 1
    print(cnts)


def signal2mfcc(sig):
    """
    0 x2, 1 x1, 2 x5, 3 x30
    :return:
    """
    if len(sig) > WAV_LENGTH:
        st = random.randint(0, len(sig) - WAV_LENGTH)
        sig = sig[st:st + WAV_LENGTH]
    else:
        new_sig = np.zeros(WAV_LENGTH)
        st = (WAV_LENGTH - len(sig)) // 2
        new_sig[st:st + len(sig)] = sig
        sig = new_sig
    # print(sig.shape)
    mfcc = librosa.feature.mfcc(y=sig, sr=8000, n_mfcc=26)
    # print(mfcc.shape)
    return mfcc


def create_dataset():
    mfccs = []
    labels = []
    device = torch.device("cpu")
    lab2num = {0: 2, 1: 1, 2: 5, 3: 30}
    # [687, 1278, 224, 42]
    # 687*2+1278*1+224*5+42*30=5032
    with open("./dataset.txt", 'r') as fin:
        lines = fin.readlines()
        print(len(lines))
        for line in tqdm(lines):
            parts = line.strip().split(',')
            lab = int(parts[1])
            sig, y = librosa.load(path=parts[0])  # (22871,)
            j = 0
            while j < lab2num[lab]:
                mfccs.append(torch.from_numpy(signal2mfcc(sig=sig)))
                labels.append(int(parts[1]))
                j += 1
            # line = fin.readline()
    clean_labels = torch.from_numpy(np.array(labels))
    output_matrix = torch.zeros((clean_labels.shape[0], 4), device=device)
    output_matrix = output_matrix.scatter_(1, clean_labels.unsqueeze(1).long().to(device), 1)
    mfcc_tensor = torch.stack(mfccs).to(device)
    print(mfcc_tensor.shape, output_matrix.shape)
    # torch.Size([5032, 26, 8])
    # torch.Size([5032, 4]) ke yi ting dao
    torch.save(mfcc_tensor, "./svdX.pt")
    torch.save(output_matrix, "./svdy.pt")


class BaseDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, ind):
        return self.datas[ind], self.labels[ind]

    def __len__(self):
        return len(self.labels)


def heatmap_eval():
    cl_model = CLSModel()
    cl_model.load_state_dict(torch.load(f"./ckpt/VOICEDIISVD/cls_model_24.pt"))
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cl_model = cl_model.to(device)

    X_test, y_test = torch.from_numpy(torch.load("./xtest.pt")).to(device), torch.load("./ytest.pt").to(device)
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
    savepath = "./result_hm_svd.png"
    max_arg = list(ypred_eval)
    conf_mat = metrics.confusion_matrix(max_arg, ytrue_eval)
    conf_mat = conf_mat / conf_mat.sum(axis=1)
    df_cm = pd.DataFrame(conf_mat,
                         index=["Healthy", "Hyperkinetic\nDysphonia", "Hyperkinetic\nDysphonia", "Reflux\nLaryngitis"],
                         columns=["Healthy", "Hyperkinetic\nDysphonia", "Hyperkinetic\nDysphonia",
                                  "Reflux\nLaryngitis"])
    heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='YlGnBu')  # , cbar_kws={'format': '%.2f%'})
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    # count_num()
    # create_dataset()
    # test_path = "F:/DataBase/Voice/svd1/pathological/male/714/2360/2360-a_h.wav"
    # path2mfcc(test_path)
    heatmap_eval()
