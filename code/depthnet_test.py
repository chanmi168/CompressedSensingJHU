''' Use this script to do preliminary tests for layer implementation and unit test
    for future maintenance
'''
import math
from test_common import TestCase, run_tests
import unittest
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.legacy.optim as old_optim
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable
from torch import sparse
from torch.autograd import gradcheck
import numpy as np
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from modules import *


class TestLayers(TestCase):
    def check_net(self, model, test_input, test_target):
        r"""Do forward and backward passes, make sure there's no error
        Args:
            - model: the model to be tested (can be single or mulit-layer)
            - test_input: torch Tensor
            - test_target: torch Tensor 
        Shape:
            - test_input: :math:`(N, C, H, W)`
            - test_target: :math:`(N, C_t, H_t, W_t)` 
            (usually have same height and width as input)
        Examples::
            >>> m = EncodingLayer()
            >>> input = torch.randn(20, 4, 32, 32)
            >>> output = m(input)
            >>> output.size()
            (20, 2, 32, 32)
        """
        model.train()
        # output from network
        data = torch.autograd.Variable(test_input)
        # output from network
        target = torch.autograd.Variable(test_target).type_as(data)
        if next(model.parameters()).is_cuda:
            data = data.cuda()
            target = target.cuda()
        print(data.size())
        optimizer = SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        # Forward prediction step
        output = model(data)
        print(output.size())
        loss = F.mse_loss(output, target)
        # Backpropagation step
        loss.backward()
        optimizer.step()

    # Major tests: must make sure model can do forward and backward passes
    def test_resnet18(self):
        path_to_file = os.getcwd()
        im_frame = Image.open(path_to_file + '/modules/sample.png').convert("RGB")
        im_np = np.array(im_frame)
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.imshow(im_np)
        fig.savefig(path_to_file + '/modules/testImg.png')   # save the figure for inspection
        plt.close(fig)

        im_np = np.transpose(im_np, (2, 0, 1))
        self.assertEqual(im_np.shape, (3, 375, 1242))
        batch_size = 16
        out_channel = 512
        test_input = torch.Tensor(im_np).expand((batch_size,) + im_np.shape)
        test_target = (torch.rand(batch_size, out_channel, 12, 39) * 2).double()
        # self.check_net(ResNet(BasicBlock, [2, 2, 2, 2]), test_input, test_target)

    def test_encoding(self):
        batch_size = 5
        input_channel = 512
        out_channel = 256
        test_input = (torch.rand(batch_size, input_channel, 8, 29) * 2).double()
        test_target = (torch.rand(batch_size, out_channel, 8, 29) * 2).double()
        # self.check_net(EncodingLayer(), test_input, test_target)


    def test_decoding(self):
        batch_size = 5
        input_channel = 256
        out_channel = 1
        test_input = (torch.rand(batch_size, input_channel, 8, 29) * 2).double()
        test_target = (torch.rand(batch_size, out_channel, 228, 912) * 2).double()
        # self.check_net(DecodingLayer(), test_input, test_target)


    def test_depthLoss(self):
        # TODO implement me
        pass


    def test_depthnet(self):
        path_to_file = os.getcwd()
        im_frame = Image.open(path_to_file + '/modules/sample.png').convert("RGB")
        im_np = np.array(im_frame)
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.imshow(im_np)
        fig.savefig(path_to_file + '/modules/testImg.png')   # save the figure for inspection
        plt.close(fig)

        im_np = np.transpose(im_np, (2, 0, 1))
        self.assertEqual(im_np.shape, (3, 375, 1242))
        batch_size = 16
        out_channel = 512
        test_input = torch.Tensor(im_np).expand((batch_size,) + im_np.shape)
        test_target = (torch.rand(batch_size, out_channel, 228, 912) * 2).double()
        self.check_net(depthnet(), test_input, test_target)

    # Minor tests: make sure sub-layer/function works as expected
    def test_vis_restnet(self):
        # visualize resnet output
        # TODO implement me
        pass

    def test_Unpool(self):
        # output for input [1] should be [[1,0],[0,0]]
        # TODO implement me
        pass
        
    def test_UpProj(self):
        # compute output using UpProj and numpy
        # TODO implement me
        pass       

if __name__ == '__main__':
    # Automatically call every function
    # in this file which begins with test.
    # see unittest library docs for details.

    run_tests()