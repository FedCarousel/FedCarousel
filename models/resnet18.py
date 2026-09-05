# models/resnet18.py
# ResNet-18 with a CIFAR stem (3x3 conv), plus 8-, 10- and 21-block mappings.
# The 21-block map is the finest admissible partition used in the paper.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        return F.relu(out)

class ResNet18CifarStem(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.pre_conv = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.pre_bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layers = nn.ModuleList([
            self._make_layer(64, 2, stride=1),   # layers.0
            self._make_layer(128, 2, stride=2),  # layers.1
            self._make_layer(256, 2, stride=2),  # layers.2
            self._make_layer(512, 2, stride=2),  # layers.3
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = F.relu(self.pre_bn(self.pre_conv(x)))
        x = self.maxpool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def create_model(num_classes: int):
    return ResNet18CifarStem(num_classes=num_classes)

# ── 10-block decomposition ────────────────────────────────────────────────────
NUM_BLOCKS_10_R18 = 10

def _key_to_block_id_10_r18(k: str) -> int:
    if k.startswith("pre_conv") or k.startswith("pre_bn"):
        return 0
    if k.startswith("layers.0.0."):
        return 1
    if k.startswith("layers.0.1."):
        return 2
    if k.startswith("layers.1.0."):
        return 3
    if k.startswith("layers.1.1."):
        return 4
    if k.startswith("layers.2.0."):
        return 5
    if k.startswith("layers.2.1."):
        return 6
    if k.startswith("layers.3.0."):
        return 7
    if k.startswith("layers.3.1."):
        return 8
    if k.startswith("fc."):
        return 9
    raise KeyError(f"[ResNet18 10-block] Unmapped key: {k}")

# Precompute
_tmp_r18_10 = ResNet18CifarStem(num_classes=10)
RESNET18_BLOCK_MAP_10 = [_key_to_block_id_10_r18(k)
                          for k in _tmp_r18_10.state_dict().keys()]


# ── 8-block decomposition (more balanced parameter counts) ───────────────────
NUM_BLOCKS_8_R18 = 8

def _key_to_block_id_8_r18(k: str) -> int:
    # Block 0 : stem + layers.0.0
    if k.startswith("pre_conv") or k.startswith("pre_bn"):
        return 0
    if k.startswith("layers.0.0."):
        return 0   # grouped with the stem
    if k.startswith("layers.0.1."):
        return 1
    if k.startswith("layers.1.0."):
        return 2
    if k.startswith("layers.1.1."):
        return 3
    if k.startswith("layers.2.0."):
        return 4
    if k.startswith("layers.2.1."):
        return 5
    if k.startswith("layers.3.0."):
        return 6
    if k.startswith("layers.3.1.") or k.startswith("fc."):
        return 7
    raise KeyError(f"[ResNet18 8-block] Unmapped key: {k}")

_tmp_r18_8 = ResNet18CifarStem(num_classes=10)
RESNET18_BLOCK_MAP_8 = [_key_to_block_id_8_r18(k)
                         for k in _tmp_r18_8.state_dict().keys()]

NUM_BLOCKS_21_R18 = 21

def _key_to_block_id_21_r18(k: str) -> int:
    # ── Stem ──────────────────────────────────────────────────────────────────
    if k.startswith("pre_conv") or k.startswith("pre_bn"):
        return 0
    # ── layers.0.0 ────────────────────────────────────────────────────────────
    if k.startswith("layers.0.0.conv1") or k.startswith("layers.0.0.bn1"):
        return 1
    if k.startswith("layers.0.0.conv2") or k.startswith("layers.0.0.bn2"):
        return 2
    # ── layers.0.1 ────────────────────────────────────────────────────────────
    if k.startswith("layers.0.1.conv1") or k.startswith("layers.0.1.bn1"):
        return 3
    if k.startswith("layers.0.1.conv2") or k.startswith("layers.0.1.bn2"):
        return 4
    # ── layers.1.0 (downsample tested LAST to avoid prefix collisions) ────────
    if k.startswith("layers.1.0.conv1") or k.startswith("layers.1.0.bn1"):
        return 5
    if k.startswith("layers.1.0.conv2") or k.startswith("layers.1.0.bn2"):
        return 6
    if k.startswith("layers.1.0.downsample"):
        return 7
    # ── layers.1.1 ────────────────────────────────────────────────────────────
    if k.startswith("layers.1.1.conv1") or k.startswith("layers.1.1.bn1"):
        return 8
    if k.startswith("layers.1.1.conv2") or k.startswith("layers.1.1.bn2"):
        return 9
    # ── layers.2.0 ────────────────────────────────────────────────────────────
    if k.startswith("layers.2.0.conv1") or k.startswith("layers.2.0.bn1"):
        return 10
    if k.startswith("layers.2.0.conv2") or k.startswith("layers.2.0.bn2"):
        return 11
    if k.startswith("layers.2.0.downsample"):
        return 12
    # ── layers.2.1 ────────────────────────────────────────────────────────────
    if k.startswith("layers.2.1.conv1") or k.startswith("layers.2.1.bn1"):
        return 13
    if k.startswith("layers.2.1.conv2") or k.startswith("layers.2.1.bn2"):
        return 14
    # ── layers.3.0 ────────────────────────────────────────────────────────────
    if k.startswith("layers.3.0.conv1") or k.startswith("layers.3.0.bn1"):
        return 15
    if k.startswith("layers.3.0.conv2") or k.startswith("layers.3.0.bn2"):
        return 16
    if k.startswith("layers.3.0.downsample"):
        return 17
    # ── layers.3.1 ────────────────────────────────────────────────────────────
    if k.startswith("layers.3.1.conv1") or k.startswith("layers.3.1.bn1"):
        return 18
    if k.startswith("layers.3.1.conv2") or k.startswith("layers.3.1.bn2"):
        return 19
    # ── Classifier ────────────────────────────────────────────────────────────
    if k.startswith("fc."):
        return 20
    raise KeyError(f"[ResNet18 21-block] Unmapped key: {k}")

# Precompute. num_classes does not change key names, only the fc weight shape.
_tmp_r18_21 = ResNet18CifarStem(num_classes=10)
RESNET18_BLOCK_MAP_21 = [_key_to_block_id_21_r18(k)
                          for k in _tmp_r18_21.state_dict().keys()]
del _tmp_r18_21
