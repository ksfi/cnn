import models
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary

OUTPUT_SIZE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = n_epochs = 1

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 28
N_CLASSES = 10

def load(m):
    global transforms
    if type(m).__name__ == "LeNet":
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

    else:
        transforms = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='cifar_data', 
                                       train=True, 
                                       transform=transforms,
                                       download=True)
        valid_dataset = datasets.CIFAR10(root='cifar_data', 
                                       train=False, 
                                       transform=transforms)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=False)

    return train_dataset, valid_dataset, train_loader, valid_loader

def train(m, train_dataset, valid_dataset, train_loader, valid_loader, loss_fn):
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

def evaluation(m, train_dataset, valid_dataset, train_loader, valid_loader, loss_fn):
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
    return total, correct, running_loss

def run(m):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    train_dataset, valid_dataset, train_loader, valid_loader = load(m)
    train(m, train_dataset, valid_dataset, train_loader, valid_loader, loss_fn)
    total, correct, running_loss = evaluation(m, train_dataset, valid_dataset, train_loader, valid_loader, loss_fn)
    return total, correct, running_loss, len(valid_loader)

if __name__ == "__main__":
    #     m = models.LeNet(10).to(DEVICE)
    m = models.AlexNet(10).to(DEVICE)
    total, correct, running_loss, valid_loader_len = run(m)

    test_loss=running_loss/valid_loader_len
    accu=100.*correct/total
    print(f"correct {correct}, total {total}")

    eval_losses=[]
    eval_accu=[]
    eval_losses.append(test_loss)
    eval_accu.append(accu)

    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 
    print(utils.outH([1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], 224))





