"""Distributed QNN experiments from the paper, one entry point.

    python dqnn.py                                  # run every experiment
    python dqnn.py semeion16x16_4qnn --n-qubits 5   # one experiment on 5 qubits

Each QNN encodes 2*n_qubits*(n_block-1) features, so the number of QNNs is
derived from --n-qubits; the feature vector is zero-padded (RX/RY(0) = identity)
when it does not split evenly. Preset names give the default-qubit layout.
"""
import argparse
import math
import warnings

import numpy as np
import torch
import torchquantum as tq
import torchvision
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import Adam
from torchquantum.measurement import expval_joint_analytical

# angle: pixel 255 -> pi/angle, c: constant output scale, block: n_block, qubits: default n_qubits
PRESETS = {
    'semeion8x8_1qnn':   dict(dataset='semeion', pool=True,  angle=8, c=35,   block=5, qubits=8, epochs=300),
    'semeion8x8_2qnn':   dict(dataset='semeion', pool=True,  angle=4, c=20,   block=3, qubits=8, epochs=300),
    'semeion16x16_4qnn': dict(dataset='semeion', pool=False, angle=8, c=10,   block=5, qubits=8, epochs=300),
    'semeion16x16_8qnn': dict(dataset='semeion', pool=False, angle=4, c=5,    block=3, qubits=8, epochs=300),
    'mnist28x28_14qnn':  dict(dataset='mnist',   pool=False, angle=4, c=4.75, block=5, qubits=7, epochs=50),
}
N_CLASS = 10
SEED = 1001
LR = 5e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConstCoeffLayer(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff
    def forward(self, x):
        return x * self.coeff


class QNNsubModel(nn.Module):
    def __init__(self, n_wires, n_block, n_depth_per_block=20):
        super().__init__()
        self.n_wires = n_wires
        self.encoder_gates_x = ([tq.functional.rx] * n_wires + [tq.functional.ry] * n_wires)*(n_block-1)
        self.n_block = n_block
        self.n_depth_per_block = n_depth_per_block
        params = np.random.rand(n_wires*n_depth_per_block*n_block*2)*math.pi
        self.u_layers = tq.QuantumModuleList()
        for j in range(n_depth_per_block*n_block):
            for i in range(n_wires):
                self.u_layers.append( tq.RX(has_params=True, trainable=True, init_params=params[i+(2*j)*n_wires]) )
            for i in range(n_wires):
                self.u_layers.append( tq.RY(has_params=True, trainable=True, init_params=params[i+(2*j+1)*n_wires]) )

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=False)
        n_depth_per_block = self.n_depth_per_block
        for d in range(self.n_block-1):
            for k in range(n_depth_per_block):
                for j in range(2*d*n_depth_per_block+2*k,2*d*n_depth_per_block+2*k+2):
                    for i in range(self.n_wires):
                        self.u_layers[i+j*self.n_wires](qdev, wires=i)
                for i in range(self.n_wires):
                    qdev.cz(wires=[i,(i+1)%self.n_wires])
            # data encoding
            for j in range(2*d,2*d+2):
                for k in range(self.n_wires):
                    self.encoder_gates_x[k+j*self.n_wires](qdev, wires=k, params=x[:, (k+j*self.n_wires)])
            for i in range(self.n_wires):
                qdev.cz(wires=[i,(i+1)%self.n_wires])
        d = self.n_block-1
        for k in range(n_depth_per_block):
            for j in range(2*d*n_depth_per_block+2*k,2*d*n_depth_per_block+2*k+2):
                for i in range(self.n_wires):
                    self.u_layers[i+j*self.n_wires](qdev, wires=i)
            if k==n_depth_per_block-1:
                break
            for i in range(self.n_wires):
                qdev.cz(wires=[i,(i+1)%self.n_wires])

        obs_list = [ expval_joint_analytical(qdev, "I"*i+Pauli+"I"*(self.n_wires-1-i)) for Pauli in ["X","Z"] for i in range(N_CLASS//2)]
        return torch.stack(obs_list, dim=1)


class QNNModel(nn.Module):
    """Splits the features into equal chunks, one QNN per chunk, and sums their outputs."""
    def __init__(self, n_qnn, n_wires, n_block):
        super().__init__()
        self.qnns = nn.ModuleList(QNNsubModel(n_wires, n_block) for _ in range(n_qnn))
    def forward(self, x):
        return sum(qnn(chunk) for qnn, chunk in zip(self.qnns, x.chunk(len(self.qnns), dim=1)))


def load(cfg):
    """Returns ((x, y), test-or-None) with x as flattened rotation angles."""
    def prep(x):
        x = torch.from_numpy(x/255*math.pi/cfg['angle'])
        if cfg['pool']:
            x = nn.AvgPool2d((2,2), stride=(2,2))(x) # 16x16 => 8x8
        return x.reshape(len(x), -1)
    if cfg['dataset'] == 'mnist':
        tr = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        te = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        return (prep(tr.data.numpy()), tr.targets), (prep(te.data.numpy()), te.targets)
    ds = torchvision.datasets.SEMEION(root='./data', download=True)
    return (prep(ds.data), torch.from_numpy(ds.labels)), None


def train(data, label, model, optimizer):
    model.train(mode=True)
    optimizer.zero_grad()
    pred = model(data)
    loss = torch.nn.CrossEntropyLoss()(pred, label)
    acc = (pred.argmax(axis=1) == label).sum().item() / len(label)
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
    print(f"valid loss: {loss.item():.5f} valid acc: {acc:.3f}")
    return loss.item(), acc


def run(name, cfg, n_qubits, epochs):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    (data, label), test = load(cfg)
    per_qnn = 2*n_qubits*(cfg['block']-1)
    n_qnn = math.ceil(data.shape[1] / per_qnn)
    pad = n_qnn*per_qnn - data.shape[1]
    print(f"== {name}: {data.shape[1]} features -> {n_qnn} QNNs x {n_qubits} qubits ({per_qnn} features each, {pad} zero-padded)")
    pad_fn = lambda x: nn.functional.pad(x, (0, pad)).to(device)

    def new_model():
        model = nn.Sequential(QNNModel(n_qnn, n_qubits, cfg['block']), ConstCoeffLayer(cfg['c']), nn.Softmax(dim=1)).to(device)
        return model, Adam(model.parameters(), lr=LR, weight_decay=5e-5)

    if test is not None:  # fixed train/test split, 50 stratified minibatches per epoch
        model, optimizer = new_model()
        data_te, label_te = pad_fn(test[0]), test[1].to(device)
        for i_epoch in range(epochs):
            for i_batch, (_, idx) in enumerate(StratifiedKFold(n_splits=50).split(data, label)):
                print(f"{i_epoch}th epoch-{i_batch}th iteration:", end=" ")
                loss_tr, acc_tr = train(pad_fn(data[idx]), label[idx].to(device), model, optimizer)
                print()
            loss_valid, acc_valid = valid(data_te, label_te, model)
        print(f"train acc: {acc_tr}, test acc: {acc_valid}, train loss: {loss_tr}, valid loss: {loss_valid}")
        return

    results = []  # final (train acc, valid acc, train loss, valid loss) per fold
    for i_fold, (tr, te) in enumerate(StratifiedKFold(n_splits=5).split(data, label)):
        model, optimizer = new_model()
        data_tr, data_te, label_tr, label_te = pad_fn(data[tr]), pad_fn(data[te]), label[tr].to(device), label[te].to(device)
        for i_epoch in range(epochs):
            print(f"{i_fold}-{i_epoch}th:", end=" ")
            loss_tr, acc_tr = train(data_tr, label_tr, model, optimizer)
            loss_valid, acc_valid = valid(data_te, label_te, model)
        results.append((acc_tr, acc_valid, loss_tr, loss_valid))
        print(f"train acc: {acc_tr}, test acc: {acc_valid}, train loss: {loss_tr}, valid loss: {loss_valid}")
    results = np.array(results)
    print("mean train/valid acc:", results[:, 0].mean(), results[:, 1].mean())
    print("std  train/valid acc:", results[:, 0].std(), results[:, 1].std())
    print("mean train/valid loss:", results[:, 2].mean(), results[:, 3].mean())
    print("std  train/valid loss:", results[:, 2].std(), results[:, 3].std())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ponytail: no argparse choices=, Python 3.10 rejects nargs='*' + choices when none given
    p.add_argument('experiments', nargs='*', help=f"run in order, default all: {', '.join(PRESETS)}")
    p.add_argument('--n-qubits', type=int, help='qubits per QNN (default: 8, or 7 for MNIST)')
    p.add_argument('--epochs', type=int, help='override the preset epoch count')
    args = p.parse_args()
    args.experiments = args.experiments or list(PRESETS)
    if unknown := [e for e in args.experiments if e not in PRESETS]:
        p.error(f"unknown experiment(s) {unknown}; choose from {list(PRESETS)}")
    if args.n_qubits is not None and args.n_qubits < N_CLASS//2:
        p.error(f"--n-qubits must be >= {N_CLASS//2}: X and Z are read from {N_CLASS//2} qubits for {N_CLASS} classes")
    warnings.simplefilter('ignore', UserWarning)
    torch.set_default_dtype(torch.float32)
    for name in args.experiments:
        cfg = PRESETS[name]
        run(name, cfg, args.n_qubits or cfg['qubits'], args.epochs or cfg['epochs'])
