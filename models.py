#!/usr/bin/env python3

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
                nn.Conv2d(3, 96, 11, stride=4), nn.ReLU(),
                nn.MaxPool2d(3, stride=2),
                nn.Conv2d(96, 256, 5, pad=2), nn.ReLU(),
                nn.MaxPool2d(3, stride=2),
                nn.Conv2d(256, 384, pad=1), nn.ReLU(),
                nn.Conv2d(384, 384, pad=1), nn.ReLU(),
                nn.Conv2d(256, 384, pad=1), nn.ReLU(),
                nn.MaxPool2s(3, stride=2))
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
        super(ConvNetA, self).__init__()
        self.feature_extract = nn.Sequential(
                nn.Conv2d(3, 64, 3), nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3), nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 256, 3), nn.ReLU(),
                nn.Conv2d(256, 256, 3), nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(256, 512, 3), nn.ReLU(),
                nn.Conv2d(512, 512, 3), nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(256, 512, 3), nn.ReLU(),
                nn.Conv2d(512, 512, 3), nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
                )
        self.Classifier = nn.Sequential(
                nn.Linear(8*8*512, 4096), nn.Relu(),
                nn.Linear(4096, 4096), nn.Relu(),
                nn.Linear(4096, 1000), nn.Relu()
                )
    def forward(self, x):
        x = nn.feature_extract(x),
        x = torch.flatten(x, 1),
        x = self.classifier(x)
        x = F.softmax(x)
        return x


# transforms1 = transforms.Compose([transforms.Resize((32, 32)),
#                                  transforms.ToTensor()])
# 
# train_dataset1 = datasets.MNIST(root='mnist_data', 
#                                train=True, 
#                                transform=transforms,
#                                download=True)
# 
# valid_dataset1 = datasets.MNIST(root='mnist_data', 
#                                train=False, 
#                                transform=transforms)
# 
# train_loader1 = DataLoader(dataset=train_dataset, 
#                           batch_size=BATCH_SIZE, 
#                           shuffle=True)
# 
# valid_loader1 = DataLoader(dataset=valid_dataset, 
#                           batch_size=BATCH_SIZE, 
#                           shuffle=False)
# 
# 
# 
# transforms2 = transforms.Compose([transforms.Resize((224, 224)),
#                                  transforms.ToTensor()])
# 
# train_dataset2 = datasets.ImageNet(root='imnet_data', 
#                                train=True, 
#                                transform=transforms2,
#                                download=True)
# 
# valid_dataset2 = datasets.MNIST(root='imnet_data', 
#                                train=False, 
#                                transform=transforms2)
# 
# train_loader2 = DataLoader(dataset=train_dataset2,
#                           batch_size=BATCH_SIZE, 
#                           shuffle=True)
# 
# valid_loader1 = DataLoader(dataset=valid_dataset2, 
#                           batch_size=BATCH_SIZE, 
#                           shuffle=False)
