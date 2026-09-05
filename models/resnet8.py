# models/resnet8.py
# ResNet-8 for CIFAR-10 / CIFAR-100 with explicit layer-block mappings.
#
# 5-block decomposition:
#   block 0: pre_conv + pre_bn (+ BN buffers)
#   block 1: layers.0.0 (conv1/bn1/conv2/bn2 + optional downsample)
#   block 2: layers.1.0 (...)
#   block 3: layers.2.0 (...)
#   block 4: fc
#
# Both mappings are aligned with the order of state_dict().keys().

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")
NUM_BLOCKS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        return F.relu(out)


class ResNet8(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16

        # block 0
        self.pre_conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.pre_bn = nn.BatchNorm2d(16)

        # blocks 1-3
        self.layers = nn.ModuleList([
            self._make_layer(16, 1, stride=1),  # layers.0.0  -> block 1
            self._make_layer(32, 1, stride=2),  # layers.1.0  -> block 2
            self._make_layer(64, 1, stride=2),  # layers.2.0  -> block 3
        ])

        # block 4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.pre_bn(self.pre_conv(x)))
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def create_model(num_classes: int) -> nn.Module:
    return ResNet8(num_classes=num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Block mapping
# ─────────────────────────────────────────────────────────────────────────────
def _key_to_block_id(k: str) -> int:
    # block 0
    if k.startswith("pre_conv") or k.startswith("pre_bn"):
        return 0
    # block 1..3 (whole BasicBlock)
    if k.startswith("layers.0.0."):
        return 1
    if k.startswith("layers.1.0."):
        return 2
    if k.startswith("layers.2.0."):
        return 3
    # block 4
    if k.startswith("fc."):
        return 4
    raise KeyError(f"[ResNet8 block map] Unmapped state_dict key: {k}")


def block_map_from_state_dict_keys(sd_keys: List[str]) -> List[int]:
    """Return a list aligned with state_dict().keys(): block_id per tensor."""
    return [_key_to_block_id(k) for k in sd_keys]


# Optional: full dict mapping "state_dict key -> block id"
def param_block_dict(num_classes: int = 10) -> Dict[str, int]:
    m = ResNet8(num_classes=num_classes)
    return {k: _key_to_block_id(k) for k in m.state_dict().keys()}


# Convenience: precomputed mapping for default num_classes=10
_tmp = ResNet8(num_classes=10)
RESNET8_BLOCK_MAP = block_map_from_state_dict_keys(list(_tmp.state_dict().keys()))

# ── 10-block decomposition ────────────────────────────────────────────────────
NUM_BLOCKS_10 = 10

def _key_to_block_id_10(k: str) -> int:
    if k.startswith("pre_conv") or k.startswith("pre_bn"):
        return 0
    if k.startswith("layers.0.0.conv1") or k.startswith("layers.0.0.bn1"):
        return 1
    if k.startswith("layers.0.0.conv2") or k.startswith("layers.0.0.bn2"):
        return 2
    if k.startswith("layers.1.0.conv1") or k.startswith("layers.1.0.bn1"):
        return 3
    if k.startswith("layers.1.0.conv2") or k.startswith("layers.1.0.bn2"):
        return 4
    if k.startswith("layers.1.0.downsample"):
        return 5
    if k.startswith("layers.2.0.conv1") or k.startswith("layers.2.0.bn1"):
        return 6
    if k.startswith("layers.2.0.conv2") or k.startswith("layers.2.0.bn2"):
        return 7
    if k.startswith("layers.2.0.downsample"):
        return 8
    if k.startswith("fc."):
        return 9
    raise KeyError(f"[ResNet8 10-block map] Unmapped key: {k}")

# Precompute
_tmp10 = ResNet8(num_classes=10)
RESNET8_BLOCK_MAP_10 = [_key_to_block_id_10(k) for k in _tmp10.state_dict().keys()]