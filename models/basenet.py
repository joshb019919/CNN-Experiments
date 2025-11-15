import torch
import torch.nn as nn

from utils import he_init


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BaseModule(nn.Module):
    """A module with n_convs conv-bn-relu layers.

    The first conv in the module can have stride > 1 for downsampling.
    """
    def __init__(self, in_ch, out_ch, n_convs=2, first_stride=1):
        super().__init__()
        layers = []
        # first conv may downsample
        layers.append(ConvBNReLU(in_ch, out_ch, stride=first_stride))
        for _ in range(n_convs - 1):
            layers.append(ConvBNReLU(out_ch, out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class BaseNet(nn.Module):
    """BaseNet configurable for 2 convs per module (BaseNet-10) or 4 
    convs per module (BaseNet-16). Naming corresponds to conv counts:

    - 3 modules * (2 convs) + 1 initial? 
    
    The assignment defines structure as 3 modules.
    """
    def __init__(self, convs_per_module=2, num_classes=10):
        super().__init__()
        # Assuming input 1-channel (Fashion-MNIST)
        self.module1 = BaseModule(1, 8, n_convs=convs_per_module, first_stride=1)
        self.module2 = BaseModule(8, 16, n_convs=convs_per_module, first_stride=2)
        self.module3 = BaseModule(16, 32, n_convs=convs_per_module, first_stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, num_classes)

        # initialization
        self.apply(he_init)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    