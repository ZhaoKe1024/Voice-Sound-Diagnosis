# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-06-02 21:26
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cls_model import CLSModel


class BaseDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, ind):
        return self.datas[ind], self.labels[ind]

    def __len__(self):
        return len(self.labels)


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    X_train1, y_train1 = torch.from_numpy(torch.load("./xtrain.pt")).to(device), torch.load("./ytrain.pt").to(device)
    X_test1, y_test1 = torch.from_numpy(torch.load("./xtest.pt")).to(device), torch.load("./ytest.pt").to(device)
    print(X_train1.shape, y_train1.shape)
    print(X_test1.shape, y_test1.shape)
    X_train, y_train = torch.load("./svdX.pt").to(device), torch.load("./svdy.pt").to(device)
    print(X_train.shape, y_train.shape)
    X_train = torch.concatenate((X_train, X_train1), dim=0)
    y_train = torch.concatenate((y_train, y_train1), dim=0)
    trainset = BaseDataset(X_train, y_train)
    testset = BaseDataset(X_test1, y_test1)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    cl_model = CLSModel().to(device)

    class_loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(cl_model.parameters(), lr=0.001)

    old = 0
    STD_acc = []
    STD_loss = []
    loss_line = []

    for epoch_id in tqdm(range(50), desc="Train"):
        cl_model.train()
        loss_list = []
        for idx, (X_mel, y_mel) in enumerate(train_loader):
            optimizer.zero_grad()
            X_mel = X_mel.transpose(1, 2).to(device)
            y_mel = y_mel.to(device)
            # print(X_mel.shape)
            pred = cl_model(X_mel)
            loss_v = class_loss(pred, y_mel)
            loss_v.backward()
            loss_list.append(loss_v.item())
            optimizer.step()
        loss_line.append(np.array(loss_list).mean())
        cl_model.eval()
        with torch.no_grad():
            acc_list = []
            loss_list = []
            for idx, (X_mel, y_mel) in enumerate(test_loader):
                X_mel = X_mel.transpose(1, 2).to(device)
                y_mel = y_mel.to(device)
                # print(X_mel.shape)
                pred = cl_model(X_mel)
                loss_eval = class_loss(pred, y_mel)
                # print(y_mel.argmax(-1))
                # print(pred.argmax(-1))
                acc_batch = metrics.accuracy_score(y_mel.argmax(-1).data.cpu().numpy(),
                                                   pred.argmax(-1).data.cpu().numpy())
                acc_list.append(acc_batch)
                loss_list.append(loss_eval.item())
            acc_per = np.array(acc_list).mean()
            print("new acc:", acc_per)
            STD_acc.append(acc_per)
            STD_loss.append(np.array(loss_list).mean())
            if acc_per > old:
                old = acc_per
                print("new acc:", acc_per)
                if acc_per > 0.80:
                    print(f"Epoch[{epoch_id}]: {acc_per}")
                    torch.save(cl_model.state_dict(), f"./ckpt/VOICEDIISVD/cls_model_{epoch_id}.pt")
                    torch.save(optimizer.state_dict(), f"./ckpt/VOICEDIISVD/optimizer_{epoch_id}.pt")
    plt.figure(0)
    plt.plot(range(len(loss_line)), loss_line, c="red", label="train_loss")
    plt.plot(range(len(STD_loss)), STD_loss, c="blue", label="valid_loss")
    plt.plot(range(len(STD_acc)), STD_acc, c="green", label="valid_accuracy")
    plt.xlabel("iteration")
    plt.ylabel("metrics")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
