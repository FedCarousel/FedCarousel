# models/cnn_femnist.py
# 4-block CNN for FEMNIST/EMNIST (28x28, single channel)
#
# Block 0: conv1 + bn1  (low-level features)
# Block 1: conv2 + bn2  (high-level features)
# Block 2: fc1          (dense representation)
# Block 3: fc2          (classifier)

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

BN_BUFFERS  = ("running_mean", "running_var", "num_batches_tracked")
NUM_BLOCKS_4 = 4


class CNNFemnist(nn.Module):
    """Standard CNN for FEMNIST/EMNIST.

    Input : (B, 1, 28, 28)
    Params: ~1.2M
    """
    def __init__(self, num_classes: int = 62):
        super().__init__()
        # ── Block 0 ─────────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        # ── Block 1 ──────────────────────────────────────────────────────
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        # ── Block 2 ──────────────────────────────────────────────────────
        # After two MaxPool(2,2) on 28x28 -> 7x7 -> 64*7*7 = 3136
        self.fc1   = nn.Linear(64 * 7 * 7, 512)
        # ── Block 3 ──────────────────────────────────────────────────────
        self.fc2   = nn.Linear(512, num_classes)

        self.pool  = nn.MaxPool2d(2, 2)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 28 -> 14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 14 -> 7
        x = x.view(x.size(0), -1)                         # 3136
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_model(num_classes: int = 62) -> nn.Module:
    return CNNFemnist(num_classes=num_classes)


# ── Block mapping ─────────────────────────────────────────────────────────────

def _key_to_block_id(k: str) -> int:
    if k.startswith("conv1") or k.startswith("bn1"):
        return 0
    if k.startswith("conv2") or k.startswith("bn2"):
        return 1
    if k.startswith("fc1"):
        return 2
    if k.startswith("fc2"):
        return 3
    raise KeyError(f"[CNNFemnist] Unmapped key: {k}")


# Precompute
_tmp = CNNFemnist(num_classes=62)
CNN_BLOCK_MAP_4 = [_key_to_block_id(k) for k in _tmp.state_dict().keys()]
del _tmp