import torch
import torch.nn as nn

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
                nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(),
                nn.MaxPool2d(3, stride=2),
                nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(),
                nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(),
                nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(3, stride=2))
        self.Classifier = nn.Sequential(
                nn.Linear(5*5*256, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(4096, 1000))
    def forward(self, x):
        x = self.feature_extract(x)
        x = torch.flatten(x, 1)
        x = self.Classifier(x)
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
