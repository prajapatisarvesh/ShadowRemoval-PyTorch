from model.rs_net import RSNet, RefinementNet
import torch
import torch.nn as nn
from data_loader.data_loader import ISTDLoader
import os

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image = torch.rand(1, 3, 224, 224).to(device=device)
    model = RSNet()
    train = ISTDLoader(csv_file='train.csv', root_dir=os.getcwd())
    print(model)
    model.to(device=device)
    x = model.forward(image)
    print(x.shape)