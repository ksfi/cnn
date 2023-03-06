from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 32

transforms1 = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

train_dataset1 = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset1 = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

train_loader1 = DataLoader(dataset=train_dataset1, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader1 = DataLoader(datasetvalid_dataset1,
                          batch_size=BATCH_SIZE, 
                          shuffle=False)




transforms2 = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor()])

train_dataset2 = datasets.CIFAR10(root='cifar_data', 
                               train=True, 
                               transform=transforms2,
                               download=True)

valid_dataset2 = datasets.CIFAR10(root='cifar_data', 
                               train=False, 
                               transform=transforms2)

train_loader2 = DataLoader(dataset=train_dataset2,
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader1 = DataLoader(dataset=valid_dataset2, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)
