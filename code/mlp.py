
""" MLP """ 

import sys
sys.path.insert(0, '/media/tx-deepocean/data_1/yzhen/fun/myenvs/torch')
sys.path.append('/media/tx-deepocean/data_1/yzhen/fun/myenvs/pkgs')

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class MLP(nn.Module):
    """ 多层感知器 """
    def __init__(self, in_dim, hid_dims, out_dim, mode):
        super(MLP, self).__init__()
        
        clf = [Rearrange('b c h w -> b (c h w)'), nn.Linear(in_dim, hid_dims[0])]
        if len(hid_dims) > 1:
            [clf.append(nn.Linear(hid_dims[i], hid_dims[i+1])) for i in range(len(hid_dims)-1)]
        clf.append(nn.Linear(hid_dims[-1], out_dim))
        self.clf = nn.Sequential(*clf)
        if mode == 'clf': self.nonlin = nn.Softmax(dim=-1)
        if mode == 'rec': self.nonlin = nn.Sigmoid()
        
    def forward(self, x):
        logits = self.clf(x)
        probs = self.nonlin(logits)
        return probs

def set_model(seed, device, in_dim, hid_dims, out_dim, lr, mode):
    """ 构建模型和优化器 """
    torch.manual_seed(seed)
    net = MLP(in_dim=in_dim, hid_dims=hid_dims, out_dim=out_dim, mode=mode)
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    return net, optimizer

if __name__ == "__main__":
    net, _ = set_model(1234, 'cpu', 28*28, [128, 256], 10, 1e-1, 'clf')
    x = torch.randn(2, 1, 28, 28)
    y_pred = net(x)
    print(y_pred.shape)