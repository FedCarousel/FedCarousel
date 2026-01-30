"""
ResNet architectures for different datasets.

Contains:
- ResNet8 for CIFAR-10
- ResNet18 for CIFAR-100 and Tiny ImageNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic ResNet block with two 3x3 convolutions."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet8(nn.Module):
    """
    ResNet-8 for CIFAR-10 (32x32 images, 10 classes).
    
    Architecture:
    - Pre-conv: 3→16 channels
    - Layer 1: 16 channels, 1 block
    - Layer 2: 32 channels, 1 block, stride 2
    - Layer 3: 64 channels, 1 block, stride 2
    - FC: 64→num_classes
    
    Total: 10 layer groups for layer-wise training
    """
    def __init__(self, num_classes: int = 10):
        super(ResNet8, self).__init__()
        self.in_planes = 16
        
        # Layer 0: pre_conv, pre_bn
        self.pre_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, 
                                  padding=1, bias=False)
        self.pre_bn = nn.BatchNorm2d(16)
        
        # Layers 1-8: Residual blocks
        self.layers = nn.ModuleList([
            self._make_layer(16, 1, stride=1),   # layers.0
            self._make_layer(32, 1, stride=2),   # layers.1
            self._make_layer(64, 1, stride=2),   # layers.2
        ])
        
        # Layer 9: Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.pre_bn(self.pre_conv(x)))
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    """
    ResNet-18 for CIFAR-100 and Tiny ImageNet.
    
    Architecture:
    - Pre-conv: 3→64 channels (with optional maxpool for larger images)
    - Layer 1: 64 channels, 2 blocks
    - Layer 2: 128 channels, 2 blocks, stride 2
    - Layer 3: 256 channels, 2 blocks, stride 2
    - Layer 4: 512 channels, 2 blocks, stride 2
    - FC: 512→num_classes
    
    Total: 21 layer groups for layer-wise training
    """
    def __init__(self, num_classes: int = 100, use_maxpool: bool = False):
        """
        Args:
            num_classes: Number of output classes (100 for CIFAR-100, 200 for TinyImageNet)
            use_maxpool: Whether to use maxpool after pre_conv (True for TinyImageNet)
        """
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.use_maxpool = use_maxpool
        
        # Layer 0: pre_conv, pre_bn
        kernel_size = 7 if use_maxpool else 3
        stride = 2 if use_maxpool else 1
        padding = 3 if use_maxpool else 1
        
        self.pre_conv = nn.Conv2d(3, 64, kernel_size=kernel_size, 
                                  stride=stride, padding=padding, bias=False)
        self.pre_bn = nn.BatchNorm2d(64)
        
        if use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layers 1-20: Residual blocks
        self.layers = nn.ModuleList([
            self._make_layer(64, 2, stride=1),    # layers.0 (blocks 0-1)
            self._make_layer(128, 2, stride=2),   # layers.1 (blocks 0-1)
            self._make_layer(256, 2, stride=2),   # layers.2 (blocks 0-1)
            self._make_layer(512, 2, stride=2),   # layers.3 (blocks 0-1)
        ])
        
        # Layer 21: Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.pre_bn(self.pre_conv(x)))
        if self.use_maxpool:
            x = self.maxpool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_resnet8(num_classes: int = 10) -> ResNet8:
    """Factory function for ResNet-8."""
    return ResNet8(num_classes=num_classes)


def create_resnet18(num_classes: int = 100, use_maxpool: bool = False) -> ResNet18:
    """
    Factory function for ResNet-18.
    
    Args:
        num_classes: Number of output classes
        use_maxpool: Use maxpool for larger images (TinyImageNet)
    """
    return ResNet18(num_classes=num_classes, use_maxpool=use_maxpool)
