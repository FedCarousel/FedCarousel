# metrics.py
# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic metrics used in Section III-B of the paper.
#
#   Magnitude Gradient (MG)      : ||w_l^t - w_l^{t-1}||^2, the strength of the
#                                  update applied to block l at round t
#                                  (Figure 2).
#   Directional Consistency (DC) : ||sum_t d_t|| / sum_t ||d_t|| over a sliding
#                                  window of tau rounds. A value near 1 means
#                                  the successive updates of a block point in
#                                  the same direction; a value near 0 means
#                                  they cancel out (Figure 3).
#
# Two variants are recorded: one on the server-side parameter deltas
# ("pseudo-gradient" dw) and one on the true gradient norms measured on the
# client side and shipped back in the fit metrics.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List

import numpy as np


def extract_block_vector_no_bn(params: List[np.ndarray],
                               block_to_indices_no_bn: Dict[int, List[int]],
                               block_id: int) -> np.ndarray:
    """Flatten every tensor of a block into a single vector (BN buffers excluded)."""
    idxs = block_to_indices_no_bn[block_id]
    if not idxs:
        return np.array([], dtype=np.float32)
    parts = [params[i].reshape(-1).astype(np.float32) for i in idxs]
    return np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.float32)


def compute_grad_norm_sq_and_ratio(prev_params, new_params,
                                   block_to_indices_no_bn,
                                   num_blocks: int, eps: float):
    """Per-block update magnitude and relative displacement.

    Returns (grad_norm_sq, ratio, deltas) where
      grad_norm_sq[b] = ||dw_b||^2
      ratio[b]        = ||dw_b|| / ||w_b||
      deltas[b]       = dw_b (kept to feed the DC sliding window)
    """
    grad_norm_sq: Dict[int, float] = {}
    ratio: Dict[int, float] = {}
    deltas: Dict[int, np.ndarray] = {}

    for b in range(num_blocks):
        w_prev = extract_block_vector_no_bn(prev_params, block_to_indices_no_bn, b)
        w_new = extract_block_vector_no_bn(new_params, block_to_indices_no_bn, b)
        if w_prev.size == 0:
            continue

        delta = w_new - w_prev
        if np.allclose(delta, 0.0, atol=1e-12):
            continue

        grad_norm_sq[b] = float(np.sum(delta * delta))
        ratio[b] = float(np.linalg.norm(delta) / (np.linalg.norm(w_prev) + eps))
        deltas[b] = delta.astype(np.float32)

    return grad_norm_sq, ratio, deltas


def update_dc_history(dc_hist: Dict[int, List[np.ndarray]],
                      deltas: Dict[int, np.ndarray], tau: int) -> Dict[int, int]:
    """Push the new deltas into the per-block sliding window of length tau."""
    for b, d in deltas.items():
        dc_hist[b].append(d)
        if len(dc_hist[b]) > tau:
            dc_hist[b].pop(0)
    return {b: len(dc_hist[b]) for b in dc_hist}


def compute_dc(dc_hist: Dict[int, List[np.ndarray]],
               tau: int, eps: float) -> Dict[int, float]:
    """Directional consistency: ||sum d_t|| / sum ||d_t|| over the window."""
    dc_vals: Dict[int, float] = {}
    for b, window in dc_hist.items():
        if len(window) != tau:
            continue
        sum_d = np.sum(window, axis=0)
        numer = float(np.linalg.norm(sum_d))
        denom = float(sum(np.linalg.norm(d) for d in window))
        dc_vals[b] = float(numer / (denom + eps))
    return dc_vals


# ── Metrics computed on the true client-side gradients ───────────────────────

def aggregate_client_grad_norms(results, num_blocks: int) -> Dict[int, float]:
    """Aggregate the clients' per-block gradient norms (weighted by n_examples)."""
    total_n = sum(fr.num_examples for _, fr in results)
    agg_norms = {}
    for _, fr in results:
        w = fr.num_examples / total_n if total_n > 0 else 0.0
        for b in range(num_blocks):
            key = f"gn_b{b}"
            if key in fr.metrics:
                agg_norms[b] = agg_norms.get(b, 0.0) + float(fr.metrics[key]) * w
    return agg_norms


def compute_grad_evolution(prev_grad_norms: Dict[int, float],
                           curr_grad_norms: Dict[int, float],
                           eps: float) -> Dict[str, Dict[int, float]]:
    """Round-to-round evolution of the gradient norm.

    grad_norm       : ||g|| at the current round
    grad_norm_delta : |curr - prev|                  (absolute change)
    grad_norm_ratio : curr / (prev + eps)            (change ratio)
    grad_norm_rel   : |curr - prev| / (prev + eps)   (relative change)
    """
    result = {
        "grad_norm":       {},
        "grad_norm_delta": {},
        "grad_norm_ratio": {},
        "grad_norm_rel":   {},
    }
    for b, curr in curr_grad_norms.items():
        result["grad_norm"][b] = curr
        if b in prev_grad_norms:
            prev = prev_grad_norms[b]
            result["grad_norm_delta"][b] = abs(curr - prev)
            result["grad_norm_ratio"][b] = curr / (prev + eps)
            result["grad_norm_rel"][b] = abs(curr - prev) / (prev + eps)
    return result


def update_grad_dc_history(grad_dc_hist: Dict[int, List[float]],
                           curr_grad_norms: Dict[int, float], tau: int) -> None:
    """Maintain a sliding history of gradient norms for the stability metric."""
    for b, g in curr_grad_norms.items():
        grad_dc_hist[b].append(g)
        if len(grad_dc_hist[b]) > tau:
            grad_dc_hist[b].pop(0)


def compute_grad_stability(grad_dc_hist: Dict[int, List[float]],
                           tau: int, eps: float) -> Dict[int, float]:
    """Gradient stability over a window of tau rounds.

    stability = 1 - std(||g||_t) / (mean(||g||_t) + eps)
    Close to 1 = stable gradient, close to 0 = highly variable.
    """
    result = {}
    for b, window in grad_dc_hist.items():
        if len(window) < tau:
            continue
        arr = np.array(window[-tau:], dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        result[b] = float(1.0 - std / (mean + eps))
    return result
