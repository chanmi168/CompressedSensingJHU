# Michae's tryout on torchvision ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import logging
import numpy as np

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from dataParser import *
from modelFunctions import *

# Training settings
parser = argparse.ArgumentParser(description='ResNet Tryout -- ')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='TB',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='sgd', metavar='O',
                    help='Optimizer options are sgd, p3sgd, adam, rms_prop')
parser.add_argument('--momentum', type=float, default=0.5, metavar='MO',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='I',
                    help="""how many batches to wait before logging detailed
                            training status, 0 means never log """)
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='D',
                    help='Options are mnist, fashion_mnist and fashion_mnist_handbag')
parser.add_argument('--data_dir', type=str, default='../../data/', metavar='F',
                    help='Where to put data')
parser.add_argument('--name', type=str, default='', metavar='N',
                    help="""A name for this training run, this
                            affects the directory so use underscores and not spaces.""")
parser.add_argument('--model', type=str, default='default', metavar='M',
                    help="""Options are default, P2Q7DefaultChannelsNet,
                    P2Q7HalfChannelsNet, P2Q7DoubleChannelsNet,
                    P2Q8BatchNormNet, P2Q9DropoutNet, P2Q10DropoutBatchnormNet,
                    P2Q11ExtraConvNet, P2Q12RemoveLayerNet, and P2Q13UltimateNet.""")
parser.add_argument('--print_log', action='store_true', default=False,
                    help='prints the csv log when training is complete')

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

def saveResults(train_losses, train_accs, val_losses, val_acc, best_val_acc):
    # TODO implement me
    raise NotImplementedError


def run_experiment(args, train_loader, test_loader):
    total_minibatch_count = 0

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = models.resnet18(pretrained=False)
    epochs_to_run = args.epochs
    optimizer = optim.Adam(model.parameters())
    # Run the primary training loop, starting with validation accuracy of 0
    val_acc = 0
    best_val_acc = 0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(1, epochs_to_run + 1):
        # train for 1 epoch (forward+backward)
        (train_acc, train_loss) = train(args, model, optimizer, train_loader, epoch)
        train_losses.append(np.squeeze(train_loss.data.numpy()))
        train_accs.append(train_acc.data.numpy())

        # validate progress on test dataset
        (val_acc, val_loss) = test(args, model, test_loader, epoch)
        val_losses.append(np.squeeze(val_loss.data.numpy()))
        val_accs.append(val_acc)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../data/.pt')

    saveResults(train_losses, train_accs, val_losses, val_acc, best_val_acc)


def resnetMain():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Prepare Dataset
    # train_dataset = ImageDataset(data = train_data, labels = train_label)
    # train_dataloader = ImageDataset(train_dataset, batch_size=args.batch_size,
    #                                 shuffle=True)

    (train_loader, test_loader) = prepareDataset(args)
    
    # Start training, evaluate loss and acc
    run_experiment(args, train_loader, test_loader)

    print('done loading')

    # TODO implement me
    # raise NotImplementedError

if __name__ == '__main__':
    resnetMain()