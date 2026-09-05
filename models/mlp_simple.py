# models/mlp_simple.py
# Simple 5-block MLP baseline (4 hidden layers + classifier).

from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")

# 5 blocks = 4 hidden + fc
_HIDDEN = [512, 256, 128, 64]
_N_HID = len(_HIDDEN)
NUM_BLOCKS = _N_HID + 1  # = 5


class SimpleMLP(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        in_dims = [3 * 32 * 32] + _HIDDEN[:-1]  # [3072, 512, 256, 128]

        self.layers = nn.ModuleList([nn.Linear(in_dims[i], _HIDDEN[i]) for i in range(_N_HID)])
        self.bns    = nn.ModuleList([nn.BatchNorm1d(_HIDDEN[i]) for i in range(_N_HID)])
        self.relu   = nn.ReLU(inplace=True)
        self.fc     = nn.Linear(_HIDDEN[-1], num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer, bn in zip(self.layers, self.bns):
            x = self.relu(bn(layer(x)))
        return self.fc(x)


def _build_param_block_map() -> Dict[str, int]:
    """
    Mapping (state_dict key) -> block id
      block i: layers.i + bns.i  (incl BN buffers)
      block 4: fc
    """
    m: Dict[str, int] = {}
    for i in range(_N_HID):
        m[f"layers.{i}.weight"] = i
        m[f"layers.{i}.bias"]   = i

        m[f"bns.{i}.weight"] = i
        m[f"bns.{i}.bias"]   = i
        m[f"bns.{i}.running_mean"] = i
        m[f"bns.{i}.running_var"]  = i
        m[f"bns.{i}.num_batches_tracked"] = i

    m["fc.weight"] = NUM_BLOCKS - 1
    m["fc.bias"]   = NUM_BLOCKS - 1
    return m


PARAM_BLOCK_MAP: Dict[str, int] = _build_param_block_map()


def build_block_map_from_state_dict_keys(sd_keys: List[str]) -> List[int]:
    """Return block_map aligned with state_dict().keys() order."""
    out: List[int] = []
    for k in sd_keys:
        bid = PARAM_BLOCK_MAP.get(k)
        if bid is None:
            raise KeyError(f"[SimpleMLP_5blocks] Unknown state_dict key '{k}' (missing in PARAM_BLOCK_MAP)")
        out.append(int(bid))
    return out


def create_model(num_classes: int) -> nn.Module:
    return SimpleMLP(num_classes=num_classes)


# Precompute default block_map with a dummy num_classes (keys don't depend on num_classes except fc shape)
_tmp = SimpleMLP(num_classes=10)
MLP_BLOCK_MAP = build_block_map_from_state_dict_keys(list(_tmp.state_dict().keys()))