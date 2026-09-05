# client.py
# ─────────────────────────────────────────────────────────────────────────────
# Flower client for FedCarousel.
#
# The client is the place where the *scheduling* of FedCarousel is realised:
# given the round number and its group (cluster) id, it decides which
# layer-block it is responsible for and trains only that block. Everything else
# (local SGD/Adam, FedProx proximal term, SCAFFOLD control variates) is a
# standard local solver, unchanged by FedCarousel.
#
# Block selection, see `_block_from_index`:
#   carousel : block = (cluster_id + position) % L   -> the L blocks advance in
#              parallel, each aggregated over ~N/L clients.
#   fedpart  : block = position % L                  -> one block advances at a
#              time, aggregated over all N clients (FedPart baseline).
# ─────────────────────────────────────────────────────────────────────────────

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import flwr as fl

from config import RUN_CFG
from models import load_model_and_blockmap


def get_lr(phase: str, block_id: int = None) -> float:
    """Return the learning rate for a given phase.

    phase: "global" | "partial" | "partial_head"
    """
    opt = RUN_CFG["optimizer"].lower()
    if opt == "adam":
        if phase == "partial" and block_id is not None:
            lr_map = RUN_CFG.get("lr_partial_by_block", {})
            if lr_map and block_id in lr_map:
                return float(lr_map[block_id])
        return {
            "global":       RUN_CFG["lr_global_adam"],
            "partial":      RUN_CFG["lr_partial_adam"],
            "partial_head": RUN_CFG["lr_partial_block4_adam"],
        }[phase]
    return {
        "global":       RUN_CFG["lr_global_sgd"],
        "partial":      RUN_CFG["lr_partial_sgd"],
        "partial_head": RUN_CFG["lr_partial_block4_sgd"],
    }[phase]


def make_optimizer(params, lr: float):
    opt = RUN_CFG["optimizer"].lower()
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=RUN_CFG["weight_decay"])
    return torch.optim.SGD(
        params, lr=lr,
        momentum=RUN_CFG["momentum"],
        weight_decay=RUN_CFG["weight_decay"],
        nesterov=True,
    )


class LayerWiseClient(fl.client.NumPyClient):
    """Client that trains either the full model or a single layer-block."""

    def __init__(self, trainloader, valloader, cluster_id: int, client_id: int):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.cluster_id = int(cluster_id)
        self.client_id = int(client_id)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.block_map, self.num_blocks = load_model_and_blockmap(
            RUN_CFG["MODEL_NAME"], RUN_CFG["num_classes"], RUN_CFG["num_blocks"]
        )
        self.model.to(self.device)

        # Map a module prefix ("layers.0.0.bn1") to its block id, so that BN
        # modules can be switched to train()/eval() per block.
        self._prefix_to_block: dict = {}
        for i, k in enumerate(self.model.state_dict().keys()):
            prefix = k.rsplit(".", 1)[0]
            self._prefix_to_block.setdefault(prefix, self.block_map[i])

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def _accumulate_block_grad_norms(self, key_to_bid: dict,
                                     accum: dict, count: dict) -> None:
        """Accumulate ||dL/dw||^2 per block after each backward pass.

        This is what feeds the Magnitude Gradient curves of Figure 2.
        """
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            bid = key_to_bid.get(name, -1)
            if bid < 0:
                continue
            accum[bid] = accum.get(bid, 0.0) + float((p.grad.data ** 2).sum())
            count[bid] = count.get(bid, 0) + 1

    def _module_name_to_block_id(self, module_name: str) -> int:
        # named_modules() yields exactly the state_dict prefixes.
        return self._prefix_to_block.get(module_name, -1)

    # ── BatchNorm handling ───────────────────────────────────────────────────

    def _set_bn_train_eval_by_target_block(self, target_block: int) -> None:
        """Decide which BN layers update their running statistics.

        bn_train_all_blocks=True : ALL BN layers in train(). Every client
            refreshes the statistics of the whole network each round; they are
            then aggregated server-side (bn_agg_all_blocks=True), so the global
            model stays consistent. This is the setting used in the paper.
        bn_train_all_blocks=False : only the active block. Use it only together
            with bn_agg_all_blocks=False AND frozen gamma/beta.
        """
        if RUN_CFG.get("bn_train_all_blocks", False):
            for _, m in self.model.named_modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.train()
            return

        for _, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
        for name, m in self.model.named_modules():
            if (isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
                    and self._module_name_to_block_id(name) == target_block):
                m.train()

    # ── Parameter I/O ────────────────────────────────────────────────────────

    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        sd = OrderedDict({k: torch.tensor(v).to(self.device)
                          for k, v in zip(keys, parameters)})
        self.model.load_state_dict(sd, strict=True)

    # ── Scheduling ───────────────────────────────────────────────────────────

    def get_training_mode(self, round_num: int) -> str:
        """Return "global" (full model) or "partial" (single block)."""
        scenario = RUN_CFG.get("training_scenario", 1)

        if scenario == 1:
            return "partial"

        if scenario == 2:
            warmup = RUN_CFG.get("warmup_rounds")
            return "global" if round_num <= warmup else "partial"

        if scenario == 4:
            T = int(RUN_CFG.get("switch_round", 300))
            if round_num <= T:
                pos = (round_num - 1) % RUN_CFG["cycle_length"]
                return "global" if pos < RUN_CFG["global_rounds_per_cycle"] else "partial"
            return RUN_CFG.get("post_switch_mode", "partial")

        # scenario 3: cyclic alternation
        pos = (round_num - 1) % RUN_CFG["cycle_length"]
        G = RUN_CFG["global_rounds_per_cycle"]
        if RUN_CFG.get("sync_at_end_of_cycle", False):
            return "global" if pos >= RUN_CFG["cycle_length"] - G else "partial"
        return "global" if pos < G else "partial"

    def _block_from_index(self, block_idx: int) -> int:
        """Single decision point between carousel and fedpart.

        carousel : the block depends on the client's group, so the L blocks
                   advance in parallel (Jacobi), each aggregated over ~N/L
                   clients.
        fedpart  : the same block for every client, so one block advances at a
                   time (Gauss-Seidel), aggregated over the N clients. This is
                   the FedPart baseline *inside this framework*, hence directly
                   comparable.
        """
        if RUN_CFG.get("variant", "carousel") == "fedpart":
            return block_idx % self.num_blocks
        return (self.cluster_id + block_idx) % self.num_blocks

    def get_block_to_train(self, round_num: int) -> int:
        """Return the id of the block to train, or -1 for a full-model round."""
        scenario = RUN_CFG.get("training_scenario", 1)
        block_repeat = RUN_CFG.get("block_repeat", 1)

        if scenario == 1:
            # G = 0: every round is partial.
            cycle_len = RUN_CFG["cycle_length"]          # = num_blocks * block_repeat
            partial_pos = (round_num - 1) % cycle_len
            block_idx = partial_pos // block_repeat
            return self._block_from_index(block_idx)

        if scenario == 2:
            warmup = RUN_CFG.get("warmup_rounds", 5)
            if round_num <= warmup:
                return -1
            partial_idx = round_num - warmup - 1         # 0-indexed after warmup
            block_idx = (partial_idx // block_repeat) % self.num_blocks
            return self._block_from_index(block_idx)

        if scenario == 4:
            T = int(RUN_CFG.get("switch_round", 300))
            G = RUN_CFG["global_rounds_per_cycle"]
            if round_num <= T:                           # cyclic phase (= scenario 3)
                pos = (round_num - 1) % RUN_CFG["cycle_length"]
                if pos < G:
                    return -1
                partial_pos = pos - G
                block_idx = partial_pos // block_repeat
                return self._block_from_index(block_idx)
            # after the switch
            if RUN_CFG.get("post_switch_mode", "partial") == "global":
                return -1                                # full training: no block
            partial_idx = round_num - T - 1              # carousel re-indexed at T+1
            block_idx = (partial_idx // block_repeat) % self.num_blocks
            return self._block_from_index(block_idx)

        # scenario 3
        G = RUN_CFG["global_rounds_per_cycle"]
        cyc = RUN_CFG["cycle_length"]
        pos = (round_num - 1) % cyc
        if RUN_CFG.get("sync_at_end_of_cycle", False):
            if pos >= cyc - G:
                return -1
            partial_pos = pos                            # partial rounds come first
        else:
            if pos < G:
                return -1
            partial_pos = pos - G
        block_idx = partial_pos // block_repeat
        return self._block_from_index(block_idx)

    # ── Local training ───────────────────────────────────────────────────────

    def _maybe_clip(self, params):
        if RUN_CFG["grad_clip_norm"] is None:
            return
        torch.nn.utils.clip_grad_norm_(params, RUN_CFG["grad_clip_norm"])

    def train_global(self, epochs: int, algo: str = "fedavg",
                     global_params=None, fedprox_mu: float = 0.0) -> dict:
        """Full-model local training (FedAvg / FedProx / synchronization round)."""
        if self.trainloader is None:
            return {}

        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        sd_keys = list(self.model.state_dict().keys())
        key_to_bid = {k: self.block_map[i] for i, k in enumerate(sd_keys)}
        opt = make_optimizer(self.model.parameters(), lr=get_lr("global"))
        crit = nn.CrossEntropyLoss()

        # Optionally limit the number of batches (global_data_fraction ablation).
        fraction = float(RUN_CFG.get("global_data_fraction", 1.0))
        n_batches = max(1, int(fraction * len(self.trainloader)))

        grad_accum, grad_count = {}, {}

        for _ in range(epochs):
            for batch_idx, (x, y) in enumerate(self.trainloader):
                if batch_idx >= n_batches:
                    break

                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = crit(self.model(x), y)

                if algo == "fedprox" and global_params is not None and fedprox_mu > 0.0:
                    prox = sum(torch.sum((p - gp) ** 2)
                               for p, gp in zip(self.model.parameters(), global_params))
                    loss = loss + 0.5 * fedprox_mu * prox

                if not torch.isfinite(loss):
                    return {}

                loss.backward()
                self._accumulate_block_grad_norms(key_to_bid, grad_accum, grad_count)
                self._maybe_clip(self.model.parameters())
                opt.step()

        return {b: float((grad_accum[b] / grad_count[b]) ** 0.5)
                for b in grad_accum if grad_count[b] > 0}

    def train_partial_block(self, block_id: int, epochs: int):
        """Local training restricted to a single layer-block (FedCarousel)."""
        if self.trainloader is None:
            return
        self.model.train()
        self._set_bn_train_eval_by_target_block(block_id)

        # Model-agnostic lookup through the state_dict keys.
        sd_keys = list(self.model.state_dict().keys())
        key_to_bid = {k: self.block_map[i] for i, k in enumerate(sd_keys)}

        trainable = []
        for name, p in self.model.named_parameters():
            bid = key_to_bid.get(name, -1)

            is_bn_param = any(
                isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
                for n, m in self.model.named_modules() if n == name.rsplit(".", 1)[0]
            )
            if RUN_CFG["freeze_bn_gamma_beta_in_partial"] and is_bn_param:
                p.requires_grad = False
                continue

            p.requires_grad = (bid == block_id)
            if p.requires_grad:
                trainable.append(p)

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train()
            return

        phase = "partial_head" if block_id == (self.num_blocks - 1) else "partial"
        lr = get_lr(phase, block_id=block_id)
        opt = make_optimizer(trainable, lr=lr)
        crit = nn.CrossEntropyLoss()

        grad_accum, grad_count = {}, {}

        for _ in range(epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = crit(self.model(x), y)
                if not torch.isfinite(loss):
                    return {}
                loss.backward()
                # Capture the gradient BEFORE the optimizer step.
                self._accumulate_block_grad_norms(key_to_bid, grad_accum, grad_count)
                self._maybe_clip(trainable)
                opt.step()

        for p in self.model.parameters():
            p.requires_grad = True

        return {b: float((grad_accum[b] / grad_count[b]) ** 0.5)
                for b in grad_accum if grad_count[b] > 0}

    def train_scaffold(self, all_params: List[np.ndarray],
                       P_all: int, epochs: int) -> Tuple[List, List]:
        """SCAFFOLD local training with control-variate correction.

        Input  : [model_params (P_all) | c_global (P_all) | c_i (P_all)]
        Output : [model_params (P_all) | delta_c_i (P_all)]

        Update    : g_corrected = g - c_i + c
        delta_c_i = -c + (x_global - y_local) / (K * lr)
        """
        if self.trainloader is None:
            zeros = [np.zeros_like(p) for p in all_params[:P_all]]
            return all_params[:P_all], zeros

        # ── Unpack ───────────────────────────────────────────────────────────
        model_params_np = all_params[:P_all]
        c_global_np = all_params[P_all:2 * P_all]
        c_i_np = all_params[2 * P_all:3 * P_all]

        self.set_parameters(model_params_np)

        sd_keys = list(self.model.state_dict().keys())
        x_global = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        param_dict = dict(self.model.named_parameters())

        c_global_t = {k: torch.tensor(c_global_np[i], dtype=torch.float32).to(self.device)
                      for i, k in enumerate(sd_keys)}
        c_i_t = {k: torch.tensor(c_i_np[i], dtype=torch.float32).to(self.device)
                 for i, k in enumerate(sd_keys)}

        # ── Training ─────────────────────────────────────────────────────────
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True

        lr = get_lr("global")
        opt = make_optimizer(self.model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        K = 0   # number of local steps

        for _ in range(epochs):
            for x_batch, y in self.trainloader:
                x_batch, y = x_batch.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = crit(self.model(x_batch), y)

                if not torch.isfinite(loss):
                    break

                loss.backward()

                # SCAFFOLD correction: grad <- grad - c_i + c
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        p.grad.data.add_(-c_i_t[name] + c_global_t[name])

                self._maybe_clip(self.model.parameters())
                opt.step()
                K += 1

        # ── delta_c_i ────────────────────────────────────────────────────────
        denom = max(K * lr, 1e-10)
        delta_c_i = []
        y_local_sd = self.model.state_dict()

        for k in sd_keys:
            if k in param_dict and K > 0:
                x_g = x_global[k].to(self.device)
                y_l = y_local_sd[k].detach()
                dci = -c_global_t[k] + (x_g - y_l) / denom
                delta_c_i.append(dci.cpu().numpy().astype(np.float32))
            else:
                # BN buffers, or K = 0: no correction.
                delta_c_i.append(np.zeros_like(c_i_np[sd_keys.index(k)], dtype=np.float32))

        updated_params = [v.detach().cpu().numpy()
                          for v in self.model.state_dict().values()]

        return updated_params, delta_c_i

    # ── Flower entry point ───────────────────────────────────────────────────

    def fit(self, parameters, config):
        algo = config.get("algo", "fedcarousel")
        round_num = int(config.get("round_num", 1))

        # Determinism: each Ray worker starts with its own unseeded torch RNG.
        # Without this, data augmentation differs between runs and the
        # fedpart / carousel arms are not comparable.
        _s = (RUN_CFG["seed"] * 100003 + round_num * 1009 + self.client_id) % (2 ** 31)
        torch.manual_seed(_s)
        torch.cuda.manual_seed_all(_s)

        # Group assignment read by client_id (= partition-id). The server sends
        # the whole assignment table, each client reads its own entry; this is
        # robust for static, k-means and dynamic grouping alike.
        assign_str = config.get("cluster_assignment")
        if assign_str:
            assignments = [int(x) for x in assign_str.split(",")]
            self.cluster_id = int(assignments[self.client_id])
        elif "cluster_id" in config:        # backward compatibility
            self.cluster_id = int(config["cluster_id"])

        # ── SCAFFOLD ─────────────────────────────────────────────────────────
        if algo == "scaffold":
            P_all = int(config.get("num_model_params", len(list(self.model.state_dict()))))
            updated_params, delta_c_i = self.train_scaffold(
                parameters, P_all, RUN_CFG["epochs_global"]
            )
            n = len(self.trainloader.dataset) if self.trainloader is not None else 0
            return updated_params + delta_c_i, n, {
                "cid":           int(self.client_id),
                "training_mode": "global",
                "layer_trained": -1,
                "algo":          "scaffold",
            }

        self.set_parameters(parameters)

        global_params = None
        if algo == "fedprox":
            global_params = [p.detach().clone() for p in self.model.parameters()]

        grad_norms = {}

        if algo in ("fedavg", "fedprox"):
            grad_norms = self.train_global(
                epochs=RUN_CFG["epochs_global"], algo=algo,
                global_params=global_params,
                fedprox_mu=RUN_CFG.get("fedprox_mu", 0.01),
            )
            block_trained = -1
            mode = "global"

        elif algo == "fedcarousel":
            mode = self.get_training_mode(round_num)

            # Full-model round with server-side refinement: the server already
            # fine-tuned the model, clients simply return it unchanged.
            if mode == "global" and RUN_CFG.get("server_finetune", False):
                n = 1   # non-zero so the result is not filtered out server-side
                return self.get_parameters(config), n, {
                    "cluster_id":    int(self.cluster_id),
                    "training_mode": "global_skip",
                    "layer_trained": -1,
                    "algo":          "fedcarousel",
                }

            if mode == "global":
                grad_norms = self.train_global(
                    epochs=RUN_CFG["epochs_global"], algo="fedcarousel"
                )
                block_trained = -1
            else:
                block_trained = self.get_block_to_train(round_num)
                grad_norms = self.train_partial_block(
                    block_trained, RUN_CFG["epochs_partial"]
                )
        else:
            raise ValueError(algo)

        n = len(self.trainloader.dataset) if self.trainloader is not None else 0

        # Encode the per-block gradient norms in the metrics dict ("gn_b0", ...).
        gn_metrics = {f"gn_b{b}": float(v) for b, v in (grad_norms or {}).items()}

        return self.get_parameters(config), n, {
            "cluster_id":    int(self.cluster_id),
            "training_mode": str(mode),
            "layer_trained": int(block_trained),
            "algo":          str(algo),
            **gn_metrics,
        }
