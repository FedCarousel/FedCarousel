# models/resnet34.py — ResNet-34 with a CIFAR stem (3x3 conv, no maxpool)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


BN_BUFFERS  = ("running_mean", "running_var", "num_batches_tracked")
NUM_BLOCKS_5  = 5
NUM_BLOCKS_10 = 10


# ── Architecture ──────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                                stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3,
                                stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = F.relu(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))
        out += self.downsample(x)
        return F.relu(out)


class ResNet34Cifar(nn.Module):
    """ResNet-34 with a CIFAR stem (3x3 conv, no maxpool).
    Stages: [3, 4, 6, 3] BasicBlocks.
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.in_planes = 64

        # Stem (CIFAR-style: 3x3, stride 1, no maxpool)
        self.pre_conv = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.pre_bn   = nn.BatchNorm2d(64)

        # 4 stages of BasicBlocks: [3, 4, 6, 3]
        self.layers = nn.ModuleList([
            self._make_layer(64,  3, stride=1),   # layers.0: 3 blocks, 64 channels
            self._make_layer(128, 4, stride=2),   # layers.1: 4 blocks, 128 channels
            self._make_layer(256, 6, stride=2),   # layers.2: 6 blocks, 256 channels
            self._make_layer(512, 3, stride=2),   # layers.3: 3 blocks, 512 channels
        ])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks  = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.pre_bn(self.pre_conv(x)))
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def create_model(num_classes: int = 100) -> nn.Module:
    return ResNet34Cifar(num_classes=num_classes)


# ── Block mappings ────────────────────────────────────────────────────────────

def _key_to_block_id_5(k: str) -> int:
    """5-block map: one residual stage per block."""
    if k.startswith("pre_conv") or k.startswith("pre_bn"):
        return 0
    if k.startswith("layers.0."):
        return 1
    if k.startswith("layers.1."):
        return 2
    if k.startswith("layers.2."):
        return 3
    if k.startswith("layers.3.") or k.startswith("fc."):
        return 4
    raise KeyError(f"[ResNet34 5-block] Unmapped: {k}")


def _key_to_block_id_10(k: str) -> int:
    """
    10-block map: residual stages split into sub-groups.
      Block 0 : stem
      Block 1 : layers.0.*   (the three 64-channel blocks, grouped)
      Block 2 : layers.1.[01].*
      Block 3 : layers.1.[23].*
      Block 4 : layers.2.[01].*
      Block 5 : layers.2.[23].*
      Block 6 : layers.2.[45].*
      Block 7 : layers.3.0.*
      Block 8 : layers.3.1.*
      Block 9 : layers.3.2.* + fc.*
    """
    if k.startswith("pre_conv") or k.startswith("pre_bn"):
        return 0

    # layers.0 -> block 1 (all three BasicBlocks grouped)
    if k.startswith("layers.0."):
        return 1

    # layers.1 -> blocks 2 and 3 (in pairs)
    if k.startswith("layers.1.0.") or k.startswith("layers.1.1."):
        return 2
    if k.startswith("layers.1.2.") or k.startswith("layers.1.3."):
        return 3

    # layers.2 -> blocks 4, 5, 6 (in pairs)
    if k.startswith("layers.2.0.") or k.startswith("layers.2.1."):
        return 4
    if k.startswith("layers.2.2.") or k.startswith("layers.2.3."):
        return 5
    if k.startswith("layers.2.4.") or k.startswith("layers.2.5."):
        return 6

    # layers.3 -> blocks 7, 8, 9 (individually)
    if k.startswith("layers.3.0."):
        return 7
    if k.startswith("layers.3.1."):
        return 8
    if k.startswith("layers.3.2.") or k.startswith("fc."):
        return 9

    raise KeyError(f"[ResNet34 10-block] Unmapped: {k}")


# Precompute
_tmp = ResNet34Cifar(num_classes=100)
_keys = list(_tmp.state_dict().keys())

RESNET34_BLOCK_MAP_5  = [_key_to_block_id_5(k)  for k in _keys]
RESNET34_BLOCK_MAP_10 = [_key_to_block_id_10(k) for k in _keys]

del _tmp, _keys