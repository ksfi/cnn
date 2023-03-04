#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUTPUT_SIZE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 2

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 28
N_CLASSES = 10

# [(W−K+2P)/S]+1

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
        probs = F.softmax(x, dim=1)
        return x, probs

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
#     optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE)
#     criterion = nn.CrossEntropyLoss()

n_epochs = 200
loss_fn = nn.BCELoss()
optimizer = optim.SGD(m.parameters(), lr=0.1)
m.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        y_pred = torch.stack(m(X_batch))
        print(type(y_pred), type(y_batch))
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

m.eval()
y_pred = m(X_test)
acc = (y_pred.round() == y_test).float().mean()
acc = float(acc)
print("Model accuracy: %.2f%%" % (acc*100))









