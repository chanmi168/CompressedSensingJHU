import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import argparse
import logging
import numpy as np


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