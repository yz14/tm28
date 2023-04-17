
""" data loader """

import sys
sys.path.insert(0, '/media/tx-deepocean/data_1/yzhen/fun/myenvs/torch')
sys.path.append('/media/tx-deepocean/data_1/yzhen/fun/myenvs/pkgs')

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def get_mnist(data_path, batch_size):
    """ load train and test data """
    tsf = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=data_path, transform=tsf, train=True)
    valid_dataset = datasets.MNIST(root=data_path, transform=tsf, train=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader