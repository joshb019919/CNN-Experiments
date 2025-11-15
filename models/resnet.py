import torch
import torch.nn as nn
from utils import he_init


class ResidualBlock(nn.Module):
    """Basic residual block: two conv layers with BatchNorm and ReLU.

    If stride != 1 or channels change, a projection (1x1) is used on 
    the shortcut.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.need_proj = (in_ch != out_ch) or (stride != 1)
        if self.need_proj:
            # 1x1 projection to match channels/spatial dims
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            self.bn_proj = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.need_proj:
            identity = self.bn_proj(self.proj(x))
        out += identity
        out = self.relu(out)
        return out

class ResNetCustom(nn.Module):
    """Start conv (8 filters) then 4 residual modules with filter counts:

    8, 16, 32, 64. 
    
    Each module contains N residual blocks; for ResNet-18 style we will
    use 2 blocks per module (total 8 blocks -> call it ResNet-18 analog).
    """
    def __init__(self, blocks_per_module=2, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)

        # modules
        self.layer1 = self._make_layer(8, 8, blocks_per_module, first_stride=1)
        self.layer2 = self._make_layer(8, 16, blocks_per_module, first_stride=2)
        self.layer3 = self._make_layer(16, 32, blocks_per_module, first_stride=2)
        self.layer4 = self._make_layer(32, 64, blocks_per_module, first_stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

        self.apply(he_init)

    def _make_layer(self, in_ch, out_ch, blocks, first_stride):
        layers = []
        # first block may downsample
        layers.append(ResidualBlock(in_ch, out_ch, stride=first_stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn_stem(self.stem(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    