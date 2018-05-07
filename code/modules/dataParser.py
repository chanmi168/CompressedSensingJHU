from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os


def prepareDataset(args):
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # if args.dataset == 'mnist':
    #     DatasetClass = datasets.MNIST
    # if args.dataset == 'CIFAR10':
    #     DatasetClass = datasets.CIFAR10

    # train_dataset = DatasetClass(
    #     dataset_dir, train=True, download=True,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ]))
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # test_dataset = DatasetClass(
    #         dataset_dir, train=False, 
    #         transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.1307,), (0.3081,))
    #     ]))
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader
    # return train_loader, test_loader

class ImageDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    # Override to give PyTorch access to any data
    def __getitem__(self, index):
        data = self.data[:,index]
        label = self.labels[index]
        return data, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.data.shape[1]
