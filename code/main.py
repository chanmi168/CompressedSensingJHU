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
# TODO get these scripts imported right
from modules import *

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
parser.add_argument('--depthLayers', action='store_true', default=False,
                    help='modify layers for depth completion')

args = parser.parse_args()

if args.depthLayers:
    print('ResNet18 modified for depth completion')
else:
    print('ResNet18')

def saveResults(train_losses, train_accs, val_losses, val_acc, best_val_acc):
    # TODO implement me
    raise NotImplementedError

def train(args, model, optimizer, train_loader, epoch):
    # Training
    model.train()
    correct_count = np.array(0)
    train_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(progress_bar):
        if args.cuda:
            data, target = data.cuda(0), target.cuda(0)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        # Forward prediction step
        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss += loss

        # Backpropagation step
        loss.backward()
        optimizer.step()

        # The batch has ended, determine the
        # accuracy of the predicted outputs
        _, argmax = torch.max(output, 1)

        # target labels and predictions are
        # categorical values
        accuracy = (target == argmax.squeeze()).float().mean()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_count += pred.eq(target.data.view_as(pred)).cpu().sum()


    train_size = Variable(torch.Tensor([len(train_loader.dataset)]).double())
    train_loss /= train_size

    progress_bar.write(
    'Epoch: {} - train results - Average train_loss: {:.4f}, train_acc: {}/{} ({:.2f}%)'.format(
        epoch, np.squeeze(train_loss.data.numpy()), correct_count, len(train_loader.dataset),
        100. * correct_count / len(train_loader.dataset)))

    return accuracy, train_loss

def test(args, model, test_loader, epoch):
    # Validation Testing
    model.eval()
    test_loss = 0
    correct = 0
    progress_bar = tqdm(test_loader, desc='Validation')
    for data, target in progress_bar:
        if args.cuda:
            data, target = data.cuda(0), target.cuda(0)
            print(torch.cuda.get_device_name(0))
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target)  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_size = Variable(torch.Tensor([len(test_loader.dataset)]).double())
    test_loss /= test_size

    acc = np.array(correct, np.float32) / test_size.data.numpy()


    progress_bar.write(
        'Epoch: {} - validation test results - Average val_loss: {:.4f}, val_acc: {}/{} ({:.2f}%)'.format(
            epoch, np.squeeze(test_loss.data.numpy()), correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return acc, test_loss

def run_experiment(args, train_loader, test_loader):
    total_minibatch_count = 0

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_classes': 10}
    model = ResNet18(**kwargs)
    # model = depthnet(**kwargs)
    if args.cuda:
        model.cuda()
    # model = models.resnet18(pretrained=False, **kwargs)
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
            torch.save(model.state_dict(), '../../data/.pt')

    saveResults(train_losses, train_accs, val_losses, val_acc, best_val_acc)


def resnetMain():
    global args
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Prepare Dataset
    # train_dataset = ImageDataset(data = train_data, labels = train_label)
    # train_dataloader = ImageDataset(train_dataset, batch_size=args.batch_size,
    #                                 shuffle=True)

    print('==> Preparing data..')
    (train_loader, test_loader) = prepareDataset(args)
    
    # Start training, evaluate loss and acc
    run_experiment(args, train_loader, test_loader)


    # TODO implement me
    # raise NotImplementedError

if __name__ == '__main__':
    resnetMain()