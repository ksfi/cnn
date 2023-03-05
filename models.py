#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
# from pytorch_lightning.callbacks import TQDMProgressBar

OUTPUT_SIZE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = n_epochs = 2

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 28
N_CLASSES = 10

# [(W−K+2P)/S]+1

# http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
class LeNet(nn.Module):
    def __init__(self, n_out):
        super(LeNet, self).__init__()
        self.feature_extract = nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.Sigmoid(),
                nn.AvgPool2d(2),
                nn.Conv2d(6, 16, 5),
                nn.Sigmoid(),
                nn.AvgPool2d(2),
                nn.Conv2d(16, 120, 5),
                nn.Sigmoid()
                )
        self.classifier = nn.Sequential(
                nn.Linear(120, 84),
                nn.Sigmoid(),
                nn.Linear(84, n_out)
                )

    def forward(self, x):
        x = self.feature_extract(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, n_out):
        super(AlexNet, self).__init__()
        self.feature_extract = nn.Sequential(
                nn.Conv2d(3, 96, 11, stride=4), nn.ReLu(),
                nn.MaxPool(3, stride=2),
                nn.Conv2d(96, 256, 5, pad=2), nn.ReLu(),
                nn.MaxPool(3, stride=2),
                nn.Conv2d(256, 384, pad=1), nn.ReLu(),
                nn.Conv2d(384, 384, pad=1), nn.ReLu(),
                nn.Conv2d(256, 384, pad=1), nn.ReLu(),
                nn.MaxPool(3, stride=2))
        self.Classifier = nn.Sequential(
                nn.Linear(5*5*256, 4096), nn.Relu(), nn.Dropout(0.5),
                nn.Linear(4096, 1000), nn.Relu(), nn.Dropout(0.5),
                nn.Linear(5*5*256, 1000))
    def forward(self, x):
        x = self.feature_extract(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# https://arxiv.org/pdf/1409.1556.pdf
class ConvNetA(nn.Module):
    def __init__(self, n_out):
        super(VGG16, self).__init__()
        self.feature_extract = nn.Sequential(
                nn.Conv2d(3, 64), nn.ReLu(),
                nn.Conv2d(64, 128), nn.ReLu()
                )


transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)


if __name__ == "__main__":
    m = LeNet(10).to(DEVICE)
    optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(m.parameters(), lr=0.1)
    m.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_pred = m(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    m.eval()
    eval_losses=[]
    eval_accu=[]
    running_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(valid_loader):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_pred = m(X_batch)
            loss = loss_fn(y_pred, y_batch)
            running_loss += loss.item()
            _, predicted = y_pred.max(1)
            total += y_batch.size(0)
            correct += y_pred.eq(y_batch.resize_(y_pred.size())).sum().item()

    test_loss=running_loss/len(valid_loader)
    accu=100.*correct/total
    print(f"correct {correct}, total {total}")

    eval_losses.append(test_loss)
    eval_accu.append(accu)

    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 










