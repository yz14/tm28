
import sys
sys.path.insert(0, '/media/tx-deepocean/data_1/yzhen/fun/myenvs/torch')
sys.path.append('/media/tx-deepocean/data_1/yzhen/fun/myenvs/pkgs')

import time
import argparse
import torch
import torch.nn.functional as F
from data_loader import get_mnist

## models 
from mlp import set_model
# from vit import set_model
# from simple_vit import set_model

def get_args():
    """ parameters for training mlp """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='device')
    parser.add_argument('--device', type=str, default='cuda:3', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='device')
    parser.add_argument('--lr', type=float, default=1e-1, help='device')
    parser.add_argument('--n_epochs', type=int, default=10, help='device')
    parser.add_argument('--batch_size', type=int, default=64, help='device')
    parser.add_argument('--in_dim', type=int, default=28*28, help='device')
    parser.add_argument('--hid_dims', type=list, default=[128, 256], help='device')
    parser.add_argument('--out_dim', type=int, default=10, help='device')
    parser.add_argument('--mode', type=str, default='clf', help='classification or reconstruction')
    args = parser.parse_args()
    return args

def compute_accuracy(net, device, data_loader):
    net.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += y.size(0)
            correct_pred += (predicted_labels == y).sum()
        return correct_pred.float()/num_examples * 100
    
def fit(net, optimizer, device, n_epochs, train_loader, valid_loader):
    t0 = time.time()
    for epoch in range(n_epochs):
        net.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            
            x, y = x.to(device), y.to(device)
                
            probs = net(x)
            cost = F.cross_entropy(probs, y)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                    %(epoch+1, n_epochs, batch_idx, len(train_loader), cost))

        with torch.set_grad_enabled(False):
            print('Epoch: %03d/%03d training accuracy: %.2f%%  valid accuracy: %.2f%%' % (
                epoch+1, n_epochs, 
                compute_accuracy(net, device, train_loader),
                compute_accuracy(net, device, valid_loader)))
            
        print('Time elapsed: %.2f min' % ((time.time() - t0)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - t0)/60))

if __name__ == "__main__":
    args = get_args()
    train_loader, valid_loader = get_mnist(args.data_path, args.batch_size)
    net, optimizer = set_model(args.seed, args.device, args.in_dim, args.hid_dims, args.out_dim, args.lr, args.mode)
    # net, optimizer = set_model(args.device, args.lr)
    # net, optimizer = set_model(28, 7, 10, 256, 4, 16, 256, 1, 64, args.device, args.lr)
    fit(net, optimizer, args.device, args.n_epochs, train_loader, valid_loader)