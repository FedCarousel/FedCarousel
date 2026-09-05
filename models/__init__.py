# models/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
# Model registry.
#
# `load_model_and_blockmap` returns (model, block_map, num_blocks) where
# `block_map` is a list aligned with state_dict().keys(): for each tensor of the
# state dict it gives the id of the layer-block it belongs to. This mapping is
# the only thing FedCarousel needs to know about an architecture, which is why
# adding a new model only requires writing its block map.
#
# Each block fuses a convolution with its BatchNorm layer into an inseparable
# (Conv, BN) unit, as stated in Section VI-A of the paper.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Tuple, List

import torch.nn as nn

from .resnet18 import (
    create_model as create_r18,
    RESNET18_BLOCK_MAP_10, NUM_BLOCKS_10_R18,
    RESNET18_BLOCK_MAP_8, NUM_BLOCKS_8_R18,
    RESNET18_BLOCK_MAP_21, NUM_BLOCKS_21_R18,
)
from .resnet8 import (
    create_model as create_r8,
    RESNET8_BLOCK_MAP, NUM_BLOCKS as RESNET8_NUM_BLOCKS,
    RESNET8_BLOCK_MAP_10, NUM_BLOCKS_10 as RESNET8_NUM_BLOCKS_10,
)
from .resnet34 import (
    create_model as create_r34,
    RESNET34_BLOCK_MAP_5, NUM_BLOCKS_5 as RESNET34_NUM_BLOCKS_5,
    RESNET34_BLOCK_MAP_10, NUM_BLOCKS_10 as RESNET34_NUM_BLOCKS_10,
)
from .cnn_femnist import (
    create_model as create_cnn,
    CNN_BLOCK_MAP_4, NUM_BLOCKS_4 as CNN_NUM_BLOCKS_4,
)
from .mlp_simple import (
    create_model as create_mlp,
    MLP_BLOCK_MAP, NUM_BLOCKS as MLP_NUM_BLOCKS,
)

__all__ = ["load_model_and_blockmap"]


def load_model_and_blockmap(model_name: str, num_classes: int,
                            num_blocks: int) -> Tuple[nn.Module, List[int], int]:
    """Instantiate a model and return (model, block_map, num_blocks).

    Available decompositions
    ------------------------
      ResNet-8  : L = 5 (default) or 10
      ResNet-18 : L = 5 (default), 8, 10 or 21 (finest admissible partition)
      ResNet-34 : L = 5 (default) or 10
      CNNFemnist: L = 4
      SimpleMLP : L = 5
    """
    mn = model_name.lower()

    if mn == "resnet34":
        m = create_r34(num_classes=num_classes)
        if num_blocks == 10:
            return m, list(RESNET34_BLOCK_MAP_10), RESNET34_NUM_BLOCKS_10
        return m, list(RESNET34_BLOCK_MAP_5), RESNET34_NUM_BLOCKS_5

    if mn in ("cnnfemnist", "cnn"):
        m = create_cnn(num_classes=num_classes)
        return m, list(CNN_BLOCK_MAP_4), CNN_NUM_BLOCKS_4

    if mn == "resnet8":
        m = create_r8(num_classes=num_classes)
        if num_blocks == 10:
            return m, list(RESNET8_BLOCK_MAP_10), RESNET8_NUM_BLOCKS_10
        return m, list(RESNET8_BLOCK_MAP), RESNET8_NUM_BLOCKS   # default: 5

    if mn == "resnet18":
        m = create_r18(num_classes=num_classes)

        if num_blocks == 21:
            return m, list(RESNET18_BLOCK_MAP_21), NUM_BLOCKS_21_R18
        if num_blocks == 10:
            return m, list(RESNET18_BLOCK_MAP_10), NUM_BLOCKS_10_R18
        if num_blocks == 8:
            return m, list(RESNET18_BLOCK_MAP_8), NUM_BLOCKS_8_R18

        # 5 blocks (default): one residual stage per block.
        sd_keys = list(m.state_dict().keys())

        def _key_to_block_id(k: str) -> int:
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
            raise KeyError(k)

        block_map = [_key_to_block_id(k) for k in sd_keys]
        return m, block_map, 5

    if mn in ("simplemlp", "mlp"):
        m = create_mlp(num_classes=num_classes)
        return m, list(MLP_BLOCK_MAP), int(MLP_NUM_BLOCKS)

    raise ValueError(f"Unknown model name: {model_name}")
