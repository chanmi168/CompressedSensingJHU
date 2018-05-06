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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        # TODO check whether bias or not
        self.layer_convEnc = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.layer_batchEnc = nn.BatchNorm2d(256)
        self.layer_upsampDec1 = nn.Upsample(size=(256, 128), scale_factor=2, mode='nearest')
        self.layer_upsampDec2 = nn.Upsample(size=(128, 64), scale_factor=2, mode='nearest')
        self.layer_upsampDec3 = nn.Upsample(size=(64, 32), scale_factor=2, mode='nearest')
        self.layer_upsampDec4 = nn.Upsample(size=(32, 16), scale_factor=2, mode='nearest')
        self.layer_convDec = nn.Conv2d(16, 1, kernel_size=3, bias=False)
        self.layer_upsampDec5 = nn.Upsample(size=(1, 1), scale_factor=2, mode='bilinear')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        


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
    .. _Efficient Object Localization Using Convolutional Networks:
       http://arxiv.org/abs/1411.4280
    """
    def __init__(self, input_ch=3):
        super(depthnet, self).__init__()
        self.input_ch = input_ch

    def forward(self, input):
        layers = []
        # ResNet18
        layers.append(resnet18(self.input_ch))
        # ResNet(BasicBlock, [2,2,2,2], **kwargs)
        # Encoding
        # layers.append(depthEnc())
        # Decoding
        # layers.append(depthDec())
        return nn.Sequential(*layers)

    def __repr__(self):
        inputCh_str = ', input channel' if self.input_ch else ''
        return self.__class__.__name__ + '(' + 'input_ch=' + str(self.input_ch) + inputCh_str + ')'

