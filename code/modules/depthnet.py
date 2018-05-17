'''depthnet in PyTorch.

Powerful net that's capable of perform depth completion through a encoding and 
decoding scheme using RGB and incomplete depth input

Reference:
[1] Ma, F., Karaman, S.: Sparse-to-dense: Depth prediction from sparse depth samples and a single image.
    In: International Conference on Robotics and Automation (ICRA). (2018)
[2] Zhang, Y., Funkhouser, T.: Deep Depth Completion of a Single RGB-D Image
[3] I. Laina, C. Rupprecht, V. Belagiannis, F. Tombari, and
    N. Navab. Deeper depth prediction with fully convolutional
    residual networks. In 3D Vision (3DV), 2016 Fourth International
    Conference on, pages 239–248. IEEE, 2016.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    # TODO minor, maybe we can write description?
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # TODO minor, maybe we can write description?
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(tensor=m.weight, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

class EncodingLayer(nn.Module):
    r"""EncodingLayer adpats the architecture described in [1]. It 
    includes one 1x1 convolutional layer followed by a bath 
    normalization layer. It expects a 4D input tensor of a 
    dimension (N, C, H, W). The output will be a 4D tensor of a
    dimension (N, C//2, H, W).

    Args:
        - None. The variables used for each sub-layer are hard-coded
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C//2, H, W)` (same height and width as input)
    Examples::
        >>> m = EncodingLayer()
        >>> input = torch.randn(20, 4, 32, 32)
        >>> output = m(input)
        >>> output.size()
        (20, 2, 32, 32)
    """
    def __init__(self):
        super(EncodingLayer, self).__init__()
        self.conv = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0,
                               bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x

class Unpool(nn.Module):
    # TODO minor, maybe we can write description?

    # Unpool: 2*2 unpooling with zero padding 
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        # self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU 
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride)) # currently not compatible with running on CPU 
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)


class UpProj(nn.Module):
    # TODO minor, maybe we can write description?    
    def __init__(self, input_ch):
        super(UpProj, self).__init__()
        self.input_ch = input_ch
        self.output_ch = input_ch//2
        # self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=2)
        # TODO stole it from https://github.com/fangchangma/sparse-to-dense.pytorch/blob/master/models.py
        # need to verify it's right
        self.unpool = Unpool(input_ch)

        # upper branch
        self.conv1 = nn.Conv2d(self.input_ch, self.output_ch, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.output_ch, self.output_ch, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_ch)

        # lower branch
        self.conv3 = nn.Conv2d(self.input_ch, self.output_ch, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_ch)

        # final relu
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.unpool(x)
        upper_x = x
        lower_x = x

        # upper branch
        upper_x = self.conv1(upper_x)
        upper_x = self.bn1(upper_x)
        upper_x = self.relu1(upper_x)
        upper_x = self.conv2(upper_x)
        upper_x = self.bn2(upper_x)

        # lower branch
        lower_x = self.conv3(lower_x)
        lower_x = self.bn3(lower_x)

        # sum them
        out = self.relu2(upper_x + lower_x)
        
        return out



class DecodingLayer(nn.Module):
    r"""DecodingLayer adpats the architecture described in [1]. It includes
    four up-projection (UpProj) layers [3], one convolutional layer, and
    one bilinear upsampling layer. It expects a 4D input tensor of a 
    dimension (N, C, H, W). The output will be a 4D input tensor of a 
    dimension (N, 1, H, W) since it's the last layer of a autoencoder.

    Args:
        - None. The variables used for each sub-layer are hard-coded
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, 1, H, W)` (same height and width as input)
    Examples::
        >>> m = DecodingLayer()
        >>> input = torch.randn(20, 4, 32, 32)
        >>> output = m(input)
        >>> output.size()
        (20, 1, 32, 32)
    """
    def __init__(self):
        super(DecodingLayer, self).__init__()
        self.UpProj1 = UpProj(input_ch=256)
        self.UpProj2 = UpProj(input_ch=128)
        self.UpProj3 = UpProj(input_ch=64)
        self.UpProj4 = UpProj(input_ch=32)
        self.conv = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bilinear = nn.Upsample(size=(228, 912), mode='bilinear')

    def forward(self, x):
        x = self.UpProj1(x)
        x = self.UpProj2(x)
        x = self.UpProj3(x)
        x = self.UpProj4(x)
        x = self.conv(x)
        x = self.bilinear(x)
        return x

class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)

class depthLoss(_WeightedLoss):
    # TODO implement and write class description (check P3BCELoss)
    def __init__(self):
        super(depthLoss, self).__init__()

    def forward(self, input, target):
        raise NotImplementedError

class depthnet(nn.Module):
    r"""depthnet adpats a similar architecture described in [1]. It includes
    a ResNet18, a encoding layer, and a decoding layer. It expects an input 
    tensor with four channels (RGB+depth). The loss is defined the same as
    in [2]. 

    ResNet18: similar to that in torchvision model, but the avg_pool layer
    and the linear layer are removed
    Encoding layer: 1x1 conv layer + bath normalization layer
    Decoding layer: four upsampling layer (same as upProj, described in 
    Figure 2c) + 3x3 conv layer + bilinear upsampling

    Args:
        - input_ch: number of the channels in input (for just RGB images, 
        input_ch = 3)
    Shape:
        - input_ch: C
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples::
        >>> m = depthnet(input_ch=4)
        >>> input = torch.randn(20, 4, 32, 32)
        >>> output = m(input)
    """
    def __init__(self, input_ch=3):
        super(depthnet, self).__init__()
        self.input_ch = input_ch
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        self.encoding = EncodingLayer()
        self.decoding = DecodingLayer()

    def forward(self, input):
        output = self.resnet18(input)
        output = self.encoding(output)
        output = self.decoding(output)
        return output

    def __repr__(self):
        inputCh_str = ', input channel' if self.input_ch else ''
        return self.__class__.__name__ + '(' + 'input_ch=' + str(self.input_ch) + inputCh_str + ')'

