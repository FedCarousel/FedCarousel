# strategies/fedcarousel_strategy.py
# ─────────────────────────────────────────────────────────────────────────────
# Server-side strategy of FedCarousel.
#
# Two aggregation regimes:
#   * full-model round  ("global")  -> plain weighted FedAvg over all clients.
#   * block round       ("partial") -> per-block weighted average: the clients
#     that trained block b are averaged together and written back only into the
#     coordinates of block b. Since the K groups train disjoint blocks in
#     parallel, the server reconstructs the complete model at every round,
#     which is what bounds cross-block staleness to a single round.
#
# The strategy also records the per-block diagnostic metrics used for Figures 2
# and 3 (Magnitude Gradient and Directional Consistency).
# ─────────────────────────────────────────────────────────────────────────────

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns

from models import load_model_and_blockmap
from config import RUN_CFG, BN_BUFFERS
from metrics import (aggregate_client_grad_norms, compute_grad_evolution,
                     update_grad_dc_history, compute_grad_stability,
                     compute_grad_norm_sq_and_ratio,
                     update_dc_history, compute_dc)


class LayerWiseFedAvg(fl.server.strategy.FedAvg):
    """FedCarousel server strategy (block-wise parallel aggregation)."""

    def __init__(self, client_to_cluster: dict = None, server_loader=None,
                 trainset=None, server_pool_idx=None, **kwargs):
        super().__init__(**kwargs)
        self.current_global_params: Optional[List[np.ndarray]] = None

        # ── Server-side refinement state ─────────────────────────────────────
        self.server_loader = server_loader
        self.trainset = trainset
        self.server_pool_idx = server_pool_idx
        self._ft_cycle_count = 0
        self._rng_server_ft = np.random.RandomState(RUN_CFG["seed"] + 7777)
        # Snapshot taken just BEFORE server fine-tuning, so that the per-block
        # displacement caused by the server can be measured (otherwise
        # aggregate_fit would compare the fine-tuned model with itself).
        self._params_before_ft: Optional[List[np.ndarray]] = None
        self._ft_round: int = -1

        # ── Diagnostic metrics ───────────────────────────────────────────────
        self.tau_dc = int(RUN_CFG["tau_dc"])
        self.eps_metric = float(RUN_CFG["eps_metric"])
        self.metrics_history: List[Dict] = []

        # ── Communication accounting ─────────────────────────────────────────
        self.cumulative_comm_mb = 0.0
        self.comm_history: List[Dict] = []
        self._bytes_per_param = 4        # float32
        self._block_param_counts = None
        self._full_param_count = None

        # ── Model and block map ──────────────────────────────────────────────
        model, block_map, num_blocks = load_model_and_blockmap(
            RUN_CFG["MODEL_NAME"], RUN_CFG["num_classes"], RUN_CFG["num_blocks"]
        )
        self.sd_keys = list(model.state_dict().keys())
        self.block_map = block_map
        self.num_blocks = int(num_blocks)

        self.dc_hist: Dict[int, List[np.ndarray]] = {b: [] for b in range(self.num_blocks)}
        self.grad_dc_hist: Dict[int, List[float]] = {b: [] for b in range(self.num_blocks)}
        self.prev_block_grad_norms: Dict[int, float] = {}

        # Indices of each block's tensors, excluding BN buffers.
        self.block_to_indices_no_bn = {b: [] for b in range(self.num_blocks)}
        for i, k in enumerate(self.sd_keys):
            b = self.block_map[i]
            if not k.endswith(BN_BUFFERS):
                self.block_to_indices_no_bn[b].append(i)

        # Indices INCLUDING the BN buffers (running_mean / running_var), used by
        # _aggregate_partial_blockwise when bn_partial_agg=True.
        # num_batches_tracked is an int64 counter: averaging it in float32 would
        # silently corrupt it, and it is meaningless once aggregated, so it is
        # excluded everywhere.
        self.block_to_indices_all = {b: [] for b in range(self.num_blocks)}
        for i, _k in enumerate(self.sd_keys):
            if _k.endswith("num_batches_tracked"):
                continue
            self.block_to_indices_all[self.block_map[i]].append(i)

        # BN buffer indices across all blocks, used when bn_agg_all_blocks=True
        # to refresh the statistics of the whole network at every round.
        self.bn_buffer_indices = [
            i for i, k in enumerate(self.sd_keys)
            if k.endswith(("running_mean", "running_var"))
        ]

        # ── Client grouping (static or dynamic) ──────────────────────────────
        self.client_to_cluster: dict = dict(client_to_cluster) if client_to_cluster else {}
        self._rng_rebuild = np.random.RandomState(RUN_CFG["seed"] + 999)

        if RUN_CFG["debug_print_mapping"]:
            print("\n[DEBUG] Server mapping per block (no BN buffers):")
            for b in range(self.num_blocks):
                idxs = self.block_to_indices_no_bn[b]
                sample = [self.sd_keys[i] for i in idxs[:15]]
                print(f"  block {b}: {len(idxs)} tensors sample={sample}")

    # ── Server-side refinement ───────────────────────────────────────────────

    def _resample_server_loader(self) -> None:
        """Redraw a server subset from the pool."""
        if self.trainset is None or self.server_pool_idx is None:
            return  # no resampling possible, keep the fixed loader

        from torch.utils.data import Subset, DataLoader

        frac = RUN_CFG.get("server_finetune_fraction", 0.05)
        n_total = len(self.trainset)
        n_server = max(32, int(n_total * frac))

        pool = list(self.server_pool_idx)
        chosen = self._rng_server_ft.choice(pool, size=min(n_server, len(pool)),
                                            replace=False).tolist()

        subset = Subset(self.trainset, chosen)
        self.server_loader = DataLoader(subset, batch_size=RUN_CFG["batch_size"],
                                        shuffle=True, num_workers=0, pin_memory=False)

        if RUN_CFG.get("debug", False):
            print(f"[ServerFT] resampled {len(chosen)} samples from pool of {len(pool)}")

    def _server_finetune(self, server_round: int) -> None:
        """Fine-tune the global model on the server's own subset."""
        if self.current_global_params is None:
            return

        if RUN_CFG.get("server_ft_resample", True):
            self._resample_server_loader()

        if self.server_loader is None:
            return

        import torch
        import torch.nn as nn
        from collections import OrderedDict

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, _, _ = load_model_and_blockmap(
            RUN_CFG["MODEL_NAME"], RUN_CFG["num_classes"], RUN_CFG["num_blocks"]
        )
        model.to(device)

        sd = OrderedDict({k: torch.tensor(v).to(device)
                          for k, v in zip(self.sd_keys, self.current_global_params)})
        model.load_state_dict(sd, strict=True)
        model.train()

        for p in model.parameters():
            p.requires_grad = True

        lr = float(RUN_CFG.get("server_finetune_lr", 1e-4))
        epochs = int(RUN_CFG.get("server_finetune_epochs", 5))
        wd = float(RUN_CFG.get("server_ft_weight_decay", RUN_CFG.get("weight_decay", 5e-4)))

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        n_samp = len(self.server_loader.dataset)

        for _ in range(epochs):
            for x, y in self.server_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                opt.step()

        # Full state dict (parameters + BN buffers).
        self.current_global_params = [v.detach().cpu().numpy()
                                      for v in model.state_dict().values()]
        self._ft_cycle_count += 1

        print(f"[ServerFT] round {server_round} | cycle {self._ft_cycle_count} | "
              f"{n_samp} samples | {epochs} ep | lr={lr} wd={wd}")

    # ── Dynamic re-clustering ────────────────────────────────────────────────

    def _should_rebuild(self, server_round: int) -> bool:
        if not RUN_CFG.get("dynamic_clustering", False):
            return False
        freq = int(RUN_CFG.get("cluster_rebuild_freq", 1))
        if freq <= 0:
            return False
        return (server_round - 1) % freq == 0

    def _rebuild_clusters(self, server_round: int) -> None:
        """Uniform random rebuild: each group gets n//k or n//k + 1 clients."""
        n = RUN_CFG["num_clients"]
        k = RUN_CFG["num_clusters"]
        assignments = np.array([i % k for i in range(n)], dtype=int)
        self._rng_rebuild.shuffle(assignments)
        self.client_to_cluster = {i: int(assignments[i]) for i in range(n)}

        if RUN_CFG.get("debug_counts", False):
            sizes = {c: int((assignments == c).sum()) for c in range(k)}
            print(f"[SERVER] Round {server_round} — groups rebuilt: {sizes}")

    # ── Round configuration ──────────────────────────────────────────────────

    def get_mode(self, server_round: int) -> str:
        """Return "global" (full model) or "partial" (block round)."""
        scenario = RUN_CFG.get("training_scenario", 1)

        if scenario == 1:
            return "partial"

        if scenario == 2:
            warmup = RUN_CFG.get("warmup_rounds", 5)
            return "global" if server_round <= warmup else "partial"

        if scenario == 4:                       # cyclic until T, then fixed
            T = int(RUN_CFG.get("switch_round", 300))
            if server_round <= T:
                pos = (server_round - 1) % RUN_CFG["cycle_length"]
                return "global" if pos < RUN_CFG["global_rounds_per_cycle"] else "partial"
            return RUN_CFG.get("post_switch_mode", "partial")

        # scenario 3
        pos = (server_round - 1) % RUN_CFG["cycle_length"]
        G = RUN_CFG["global_rounds_per_cycle"]
        if RUN_CFG.get("sync_at_end_of_cycle", False):
            return "global" if pos >= RUN_CFG["cycle_length"] - G else "partial"
        return "global" if pos < G else "partial"

    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(
            client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        )

        if self._should_rebuild(server_round):
            self._rebuild_clusters(server_round)

        mode = self.get_mode(server_round)

        # ── Server-side refinement ───────────────────────────────────────────
        # sync_at_end_of_cycle=False : refine on the FIRST round of the cycle
        # sync_at_end_of_cycle=True  : refine on the LAST round of the cycle
        if RUN_CFG.get("server_finetune", False):
            freq = int(RUN_CFG.get("server_finetune_freq", 1))
            period = RUN_CFG["cycle_length"] * freq

            if RUN_CFG.get("sync_at_end_of_cycle", False):
                do_ft = (server_round % period == 0)
            else:
                do_ft = ((server_round - 1) % period == 0)

            if do_ft:
                if self.current_global_params is None:
                    self.current_global_params = parameters_to_ndarrays(parameters)
                self._params_before_ft = [p.copy() for p in self.current_global_params]
                self._ft_round = int(server_round)
                self._server_finetune(server_round)
                parameters = ndarrays_to_parameters(self.current_global_params)
                print(f"[SERVER] Round {server_round}: server refinement (freq={freq} cycles)")
            elif mode == "global":
                # Full-model round with server_finetune enabled but no refinement
                # scheduled: clients skip and the server does nothing -> empty
                # round. With G=1 and freq=1 this never happens.
                print(f"[SERVER] Warning: round {server_round} is an EMPTY global round "
                      f"(server_finetune enabled but no refinement this round). "
                      f"Use G=1 with freq=1 to avoid it.")

        # ── Group assignment, robust to the ClientProxy id ───────────────────
        # client.cid (ClientProxy) is not the partition-id in every Flower
        # version, so the whole assignment table is serialized and each client
        # reads its own entry. Works for k-means, random and dynamic grouping.
        N = RUN_CFG["num_clients"]
        k = RUN_CFG["num_clusters"]
        assign_str = ",".join(str(self.client_to_cluster.get(i, i % k)) for i in range(N))

        if RUN_CFG.get("debug_counts", False):
            from collections import Counter
            sizes = Counter(self.client_to_cluster.get(i, i % k) for i in range(N))
            print(f"[SERVER] round={server_round} group sizes SENT = "
                  f"{dict(sorted(sizes.items()))}")

        instructions = []
        for client in clients:
            cfg = {
                "round_num":          server_round,
                "algo":               "fedcarousel",
                "training_mode":      mode,
                "cluster_assignment": assign_str,
            }
            instructions.append((client, FitIns(parameters, cfg)))

        return instructions

    # ── Aggregation ──────────────────────────────────────────────────────────

    def aggregate_fit(self, server_round, results, failures):
        active = [(c, r) for c, r in results if r.num_examples and r.num_examples > 0]
        if not active:
            if self.current_global_params is not None:
                return ndarrays_to_parameters(self.current_global_params), {}
            return None, {}

        mode = self.get_mode(server_round)
        prev_params = (None if self.current_global_params is None
                       else [p.copy() for p in self.current_global_params])

        # ── Full-model round with server refinement: already fine-tuned ──────
        if mode == "global" and RUN_CFG.get("server_finetune", False):
            # Clients returned the model unchanged; current_global_params already
            # holds the refined model, so there is nothing to re-aggregate.
            new_params = [p.copy() for p in self.current_global_params]

            # The refinement happened in configure_fit, so compare against the
            # snapshot taken BEFORE it, otherwise the delta is zero by design.
            ft_done = (self._ft_round == int(server_round)
                       and self._params_before_ft is not None)
            base = self._params_before_ft if ft_done else prev_params

            if base is not None:
                _, ratio, deltas = compute_grad_norm_sq_and_ratio(
                    base, new_params,
                    self.block_to_indices_no_bn, self.num_blocks, self.eps_metric
                )
                update_dc_history(self.dc_hist, deltas, self.tau_dc)
                dc_vals = compute_dc(self.dc_hist, self.tau_dc, self.eps_metric)
            else:
                ratio, dc_vals = {}, {}

            # One entry per round, with no gap: otherwise metrics_history is
            # misaligned with the round index in every curve.
            self.metrics_history.append({
                "round":           int(server_round),
                "mode":            "server_ft" if ft_done else "global_empty",
                "grad_norm":       {},
                "grad_norm_delta": {},
                "grad_norm_ratio": {},
                "grad_stability":  {},
                "dw_ratio":        ratio,
                "DC_dw":           dc_vals,
            })

            if ft_done and RUN_CFG.get("debug_metrics", False):
                per_block = " ".join(f"b{b}={ratio[b]:.4f}" for b in sorted(ratio))
                print(f"[ServerFT] round {server_round} relative per-block "
                      f"displacement: {per_block}")

            self._params_before_ft = None
            self.current_global_params = new_params
            return ndarrays_to_parameters(self.current_global_params), {}

        # ── Standard aggregation ─────────────────────────────────────────────
        if mode == "global":
            aggregated_params, metrics = self._aggregate_global(active)
        else:
            aggregated_params, metrics = self._aggregate_partial_blockwise(active, server_round)

        new_params = [p.copy() for p in parameters_to_ndarrays(aggregated_params)]

        # ── Diagnostic metrics ───────────────────────────────────────────────
        curr_grad_norms = aggregate_client_grad_norms(active, self.num_blocks)
        grad_evo = compute_grad_evolution(
            self.prev_block_grad_norms, curr_grad_norms, self.eps_metric
        )
        update_grad_dc_history(self.grad_dc_hist, curr_grad_norms, self.tau_dc)
        grad_stability = compute_grad_stability(
            self.grad_dc_hist, self.tau_dc, self.eps_metric
        )

        if prev_params is not None:
            _, ratio, deltas = compute_grad_norm_sq_and_ratio(
                prev_params, new_params,
                self.block_to_indices_no_bn, self.num_blocks, self.eps_metric
            )
            dc_accum = update_dc_history(self.dc_hist, deltas, self.tau_dc)
            dc_vals = compute_dc(self.dc_hist, self.tau_dc, self.eps_metric)
        else:
            ratio, dc_accum, dc_vals = {}, {}, {}

        self.metrics_history.append({
            "round":           int(server_round),
            "mode":            mode,
            # Actual gradient (measured client-side)
            "grad_norm":       grad_evo["grad_norm"],
            "grad_norm_delta": grad_evo["grad_norm_delta"],
            "grad_norm_ratio": grad_evo["grad_norm_ratio"],
            "grad_stability":  grad_stability,
            # Pseudo-gradient (server-side parameter delta)
            "dw_ratio":        ratio,
            "DC_dw":           dc_vals,
        })
        self.prev_block_grad_norms = dict(curr_grad_norms)

        if RUN_CFG["debug_metrics"]:
            print("\n" + "─" * 72)
            print(f"[SERVER] ROUND {server_round} METRICS mode={mode}")
            print("─" * 72)
            for b in range(self.num_blocks):
                gn = grad_evo["grad_norm"].get(b)
                gd = grad_evo["grad_norm_delta"].get(b)
                gs = grad_stability.get(b)
                dc = dc_vals.get(b)
                acc = dc_accum.get(b, 0)

                gn_s = f"{gn:.6f}" if gn is not None else "—"
                gd_s = f"{gd:.6f}" if gd is not None else "—"
                gs_s = (f"{gs:.4f}" if gs is not None
                        else f"wait({len(self.grad_dc_hist.get(b, []))}/{self.tau_dc})")
                dc_s = f"{dc:.6f}" if dc is not None else f"wait({acc}/{self.tau_dc})"

                print(f"  block {b}: ||g||={gn_s} | delta||g||={gd_s} | "
                      f"stability={gs_s} | DC(dw)={dc_s}")
            print("─" * 72)

        self.current_global_params = new_params
        return ndarrays_to_parameters(self.current_global_params), metrics

    def _aggregate_global(self, active_results):
        """Plain weighted FedAvg over the full model."""
        base = parameters_to_ndarrays(active_results[0][1].parameters)
        agg = [np.zeros_like(p, dtype=np.float32) for p in base]
        total = sum(fr.num_examples for _, fr in active_results)
        for i in range(len(agg)):
            s = np.zeros_like(agg[i], dtype=np.float32)
            for _, fr in active_results:
                nd = parameters_to_ndarrays(fr.parameters)
                s += nd[i].astype(np.float32) * (fr.num_examples / total)
            agg[i] = s
        return ndarrays_to_parameters(agg), {}

    def _aggregate_partial_blockwise(self, active_results, server_round: int):
        """Per-block weighted average (the core of FedCarousel).

        Clients are grouped by the block they trained; within a block, their
        updates are averaged and written back only into that block's
        coordinates. All L blocks are refreshed in the same round.
        """
        agg = ([p.copy() for p in self.current_global_params]
               if self.current_global_params
               else [p.copy() for p in parameters_to_ndarrays(active_results[0][1].parameters)])

        # bn_partial_agg=True -> include the active block's BN buffers.
        use_bn = RUN_CFG.get("bn_partial_agg", False)
        idx_map = self.block_to_indices_all if use_bn else self.block_to_indices_no_bn

        groups: Dict[int, List] = defaultdict(list)
        for _, fr in active_results:
            b = int(fr.metrics.get("layer_trained", -1))
            if b >= 0:
                groups[b].append(fr)

        if RUN_CFG["debug_counts"]:
            counts = {b: len(v) for b, v in sorted(groups.items())}
            print(f"[SERVER] round={server_round} PARTIAL clients-per-block = {counts}")

        # One deserialization per client instead of one per parameter index.
        nds_by_block = {
            b: [(parameters_to_ndarrays(fr.parameters), fr.num_examples) for fr in fr_list]
            for b, fr_list in groups.items()
        }

        for b, fr_list in groups.items():
            idxs = idx_map[b]
            if not idxs:
                continue
            total_n = sum(fr.num_examples for fr in fr_list)
            if total_n == 0:
                continue
            nds = nds_by_block[b]
            for pidx in idxs:
                s = np.zeros_like(agg[pidx], dtype=np.float32)
                for nd, n in nds:
                    s += nd[pidx].astype(np.float32) * (n / total_n)
                agg[pidx] = s

        # ── BN statistics of ALL blocks, over ALL active clients ─────────────
        # Keeps the global model's normalization consistent with what the
        # clients actually saw during training. Uplink cost: two floats per BN
        # channel, negligible next to a conv block's weights.
        if RUN_CFG.get("bn_agg_all_blocks", False) and self.bn_buffer_indices:
            total_all = sum(fr.num_examples for _, fr in active_results)
            if total_all > 0:
                nds_all = [(parameters_to_ndarrays(fr.parameters), fr.num_examples)
                           for _, fr in active_results]
                for pidx in self.bn_buffer_indices:
                    s = np.zeros_like(agg[pidx], dtype=np.float32)
                    for nd, n in nds_all:
                        s += nd[pidx].astype(np.float32) * (n / total_all)
                    agg[pidx] = s

        return ndarrays_to_parameters(agg), {}

    def get_metrics_history(self) -> List[Dict]:
        return self.metrics_history
