# -*- coding: utf-8 -*-
# %matplotlib inline
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Adam
import torch.autograd as autograd
from torch.autograd import gradcheck
from torch.utils.data import DataLoader

import random
import math
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, log_loss
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold


# !pip install torchquantum


import torchquantum as tq
from torchquantum.measurement import expval_joint_analytical
import warnings

seed = 1001
#random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float32)
warnings.simplefilter('ignore', UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NZ_INPUT_ANGLE = 2
# device = "cpu"


import torchvision
from torchvision.transforms import v2

semeion_data = torchvision.datasets.SEMEION(root='./data', download=True)
dataset_name = 'semeion'
data, label = semeion_data.data, semeion_data.labels
data = data/255*math.pi/8 # pi/2
n_qubits = 8
n_class = len(np.unique(label))

data = torch.nn.AvgPool2d( (2,2), stride=(2,2) )( torch.from_numpy(data) ) # # (1593, 16, 16) => (1593, 8, 8)
data = data.reshape(-1,data.shape[1]*data.shape[2]) # (1593, 8, 8) => (1593, 64)
label = torch.from_numpy(label)
n_data, n_features = data.shape


class CoeffLayer(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = torch.nn.Parameter(coeff)
    def forward(self, x):
        ret = x * self.coeff
        return ret

class ConstCoeffLayer(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff
    def forward(self, x):
        ret = x * self.coeff
        return ret


class QNNsubModel(nn.Module):
    def __init__(self):
        # params is numpy array
        super().__init__()
        self.n_wires = n_qubits
        self.encoder_gates_x = ([tq.functional.rx] * self.n_wires + [tq.functional.ry] * self.n_wires)*4
        self.n_block = 5
        self.n_depth_per_block = 20
        params = np.random.rand( self.n_wires*self.n_depth_per_block*self.n_block*2)*math.pi
        self.u_layers = tq.QuantumModuleList()
        for j in range(self.n_depth_per_block*self.n_block):
            for i in range(self.n_wires):
                self.u_layers.append( tq.RX(has_params=True, trainable=True, init_params=params[i+(2*j)*self.n_wires]) )
            for i in range(self.n_wires):
                self.u_layers.append( tq.RY(has_params=True, trainable=True, init_params=params[i+(2*j+1)*self.n_wires]) )

    def forward(self, x):
        bsz, nx_features = x.shape
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz = bsz, device=x.device, record_op=False
        )
        n_depth_per_block = self.n_depth_per_block
        for d in range(self.n_block-1): # (2,4)
            for k in range(n_depth_per_block):
                for j in range(2*d*n_depth_per_block+2*k,2*d*n_depth_per_block+2*k+2):
                    for i in range(self.n_wires):
                        self.u_layers[i+j*self.n_wires](qdev, wires=i)
                for i in range(self.n_wires):
                    qdev.cz(wires=[i,(i+1)%self.n_wires])
            # data encoding
            for j in range(2*d,2*d+2): # (0,2) (2,4)
                for k in range(self.n_wires):
                    self.encoder_gates_x[k+j*self.n_wires](qdev, wires=k, params=x[:, (k+j*self.n_wires)])
            for i in range(self.n_wires):
                qdev.cz(wires=[i,(i+1)%self.n_wires])
        for d in range(self.n_block-1,self.n_block): # (4,5)
            for k in range(n_depth_per_block):
                for j in range(2*d*n_depth_per_block+2*k,2*d*n_depth_per_block+2*k+2):
                    for i in range(self.n_wires):
                        self.u_layers[i+j*self.n_wires](qdev, wires=i)
                if k==n_depth_per_block-1:
                    break
                for i in range(self.n_wires):
                    qdev.cz(wires=[i,(i+1)%self.n_wires])

        obs_list = [ expval_joint_analytical(qdev, "I"*i+Pauli+"I"*(self.n_wires-1-i)) for Pauli in ["X","Z"] for i in range(n_class//2)]
        ret = torch.stack(obs_list, dim=1)
        return ret


class QNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn1 = QNNsubModel()
        self.qnn2 = QNNsubModel()
    def forward(self, x):
        ret = self.qnn1(x)
        return ret


def train(data, label, model, optimizer):
    model.train(mode=True)
    optimizer.zero_grad()
    pred = model(data)
    loss = torch.nn.CrossEntropyLoss()(pred, label)
    acc = (pred.argmax(axis=1) == label).sum().item() / len(label)
    # acc = accuracy_score(y_tr, pred.argmax(axis=1).cpu().detach().numpy() )
    print(f"train loss: {loss.item():.5f}, train acc: {acc:.3f}", end=' ')
    loss.backward()
    optimizer.step()
    return loss.item(), acc

def valid(data, label, model):
    model.train(mode=False)
    with torch.no_grad():
        pred = model(data)
        loss = torch.nn.CrossEntropyLoss()(pred, label)
    acc = (pred.argmax(axis=1) == label).sum().item() / len(label)
    # acc = accuracy_score(y_te, pred.argmax(axis=1).cpu().detach().numpy() )
    print(f"valid loss: {loss.item():.5f} valid acc: {acc:.3f}")
    return loss.item(), acc


all_all_tr_loss = []
all_all_valid_loss = []
all_all_tr_acc = []
all_all_valid_acc = []

max_epochs = 300
lr = 0.005
n_fold = 5
skf = StratifiedKFold(n_splits=n_fold)
for i_fold, (tr, te) in enumerate(skf.split(data, label)):
    data_tr, data_te, label_tr, label_te = data[tr].to(device), data[te].to(device), label[tr].to(device), label[te].to(device)
    model = torch.nn.Sequential(
        QNNModel(),
        ConstCoeffLayer(35),
        nn.Softmax(dim=1)
    )
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    all_tr_loss = []
    all_valid_loss = []
    all_tr_acc = []
    all_valid_acc = []
    for i_epoch in range(max_epochs):
        print(f"{i_fold}-{i_epoch}th:", end=" ")
        loss_tr, acc_tr = train(data_tr, label_tr, model, optimizer)
        loss_valid, acc_valid = valid(data_te, label_te, model)
        all_tr_loss.append(loss_tr)
        all_valid_loss.append(loss_valid)
        all_tr_acc.append(acc_tr)
        all_valid_acc.append(acc_valid)
        ###
    all_all_tr_loss.append(all_tr_loss)
    all_all_valid_loss.append(all_valid_loss)
    all_all_tr_acc.append(all_tr_acc)
    all_all_valid_acc.append(all_valid_acc)


train_acc = []
valid_acc = []
train_loss = []
valid_loss = []
for i in range(len(all_all_tr_acc)):
    train_acc.append(all_all_tr_acc[i][-1])
    valid_acc.append(all_all_valid_acc[i][-1])
    train_loss.append(all_all_tr_loss[i][-1])
    valid_loss.append(all_all_valid_loss[i][-1])
    print(f"train acc: {train_acc[-1]}, test acc: {valid_acc[-1]}, train loss: {train_loss[-1]}, valid loss: {valid_loss[-1]}")

print( np.mean(train_acc), np.mean(valid_acc) )
print( np.std(train_acc), np.std(valid_acc) )
print( np.mean(train_loss), np.mean(valid_loss))
print( np.std(train_loss), np.std(valid_loss))


nu0 = model[0].qnn1.n_depth_per_block
c0 = int(model[1].coeff)
prefix_name = dataset_name+"_"+"qnn"+str(nu0)+"_c"+str(c0)+"_"+str(n_qubits)+"qubits_"
if False:
    pd.DataFrame(all_all_tr_acc).to_csv(prefix_name+"_tr_acc.csv", index=False)
    pd.DataFrame(all_all_valid_acc).to_csv(prefix_name+"_valid_acc.csv", index=False)
    pd.DataFrame(all_all_tr_loss).to_csv(prefix_name+"_tr_loss.csv", index=False)
    pd.DataFrame(all_all_valid_loss).to_csv(prefix_name+"_valid_loss.csv", index=False)



