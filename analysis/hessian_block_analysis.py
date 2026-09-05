# analysis/hessian_block_analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# Empirical measurement of the block structure of the Hessian of the global
# objective F(w), used to support Assumption 1 of the paper:
#
#   beta_l      = ||H_ll||_2      (Lipschitz constant of block l, diagonal)
#   L           = ||H||_2         (global Lipschitz constant)
#   beta_{l,l'} = ||H_{l,l'}||_2  (inter-block coupling, off-diagonal)
#   beta_max    = max_l beta_l
#   beta_bar    = max_l sum_{l' != l} beta_{l,l'}
#
# Two facts are guaranteed and are used as sanity checks on the output:
#   Cauchy interlacing : beta_max <= L
#   Block Gershgorin   : L <= beta_max + beta_bar
# so the gap L - beta_max is bounded by the coupling beta_bar.
#
# Everything is computed WITHOUT ever forming H: Hessian-vector products
# (double backprop) plus power iteration, restricting the subspace to the
# block(s) of interest.
#
# The SPECTRAL NORM (max |lambda|) is measured, not just the largest positive
# eigenvalue: that is the actual Lipschitz constant, including away from a
# minimum.
#
# BatchNorm: the model is put in .eval() so the running_mean/var buffers are
# fixed and the loss is a deterministic function of the trainable parameters
# (weights + gamma/beta). This is the Hessian of F that governs beta_l and L.
#
# Usage
# -----
#   python analysis/hessian_block_analysis.py --model resnet18 \
#       --num_classes 100 --num_blocks 21 --dataset cifar100 --tag c100_init
#
#   # from a trained checkpoint (produced by main.py --save_checkpoint)
#   python analysis/hessian_block_analysis.py --model resnet8 \
#       --num_classes 10 --num_blocks 10 --dataset cifar10 \
#       --ckpt checkpoints/global_ResNet8_C10_a50_B10_r150.pt --tag c10_trained
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys

# Allow `python analysis/hessian_block_analysis.py` from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, math, time, argparse
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

def _flat_dot(vs, ws):
    return sum((v * w).sum() for v, w in zip(vs, ws))
def _norm(vs):
    return math.sqrt(float(_flat_dot(vs, vs)))
def _project(vs, keep):
    return [v if i in keep else torch.zeros_like(v) for i, v in enumerate(vs)]
def _rand_like(params, keep, gen):
    out = []
    for i, p in enumerate(params):
        if keep is None or i in keep:
            out.append(torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype))
        else:
            out.append(torch.zeros_like(p))
    return out

class HessianBlockAnalyzer:
    def __init__(self, model, name2block, batches, device="cpu", seed=0):
        self.device = device
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.crit = nn.CrossEntropyLoss()
        self.batches = [(x.to(device), y.to(device)) for x, y in batches]
        self._gen = torch.Generator(device=device); self._gen.manual_seed(seed)
        self.params, self.p_block = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if n not in name2block: raise KeyError(f"Unmapped parameter: {n}")
            self.params.append(p); self.p_block.append(name2block[n])
        self.num_blocks = max(self.p_block) + 1
        self.keep = {b: {i for i, bb in enumerate(self.p_block) if bb == b}
                     for b in range(self.num_blocks)}
        self.block_nparams = {b: sum(self.params[i].numel() for i in self.keep[b])
                              for b in range(self.num_blocks)}
        tot = sum(p.numel() for p in self.params)
        print(f"[analyzer] device={self.device}  params={tot:,}  blocks={self.num_blocks}  "
              f"batches={len(self.batches)} "
              f"(size {self.batches[0][0].shape[0] if self.batches else 0})", flush=True)

    def _hvp(self, vec):
            acc = [torch.zeros_like(p) for p in self.params]
            for x, y in self.batches:
                loss = self.crit(self.model(x), y)
                grads = torch.autograd.grad(loss, self.params, create_graph=True)
                dot = _flat_dot(grads, vec)
                Hv = torch.autograd.grad(dot, self.params, retain_graph=False)
                for a, h in zip(acc, Hv):
                    a.add_(h.detach())
                del loss, grads, dot, Hv          # free this batch's graph
            if len(self.batches) > 1:
                for a in acc:
                    a.div_(len(self.batches))
            return acc

    def _top_eig(self, keep, iters, tol):
        v = _rand_like(self.params, keep, self._gen)
        n = _norm(v)
        if n == 0: return 0.0
        v = [x / n for x in v]; mu_prev = 0.0
        for _ in range(iters):
            Hv = self._hvp(v)
            if keep is not None: Hv = _project(Hv, keep)
            mu = float(_flat_dot(Hv, v)); n = _norm(Hv)
            if n == 0: return 0.0
            v = [x / n for x in Hv]
            if abs(mu - mu_prev) <= tol * (abs(mu) + 1e-12): break
            mu_prev = mu
        return mu

    def _offdiag(self, keep_l, keep_lp, iters, tol):
        v = _rand_like(self.params, keep_lp, self._gen)
        n = _norm(v)
        if n == 0: return 0.0
        v = [x / n for x in v]; lam_prev = 0.0
        for _ in range(iters):
            w = _project(self._hvp(v), keep_l)
            z = _project(self._hvp(w), keep_lp)
            lam = float(_flat_dot(z, v)); n = _norm(z)
            if n == 0: return 0.0
            v = [x / n for x in z]
            if abs(lam - lam_prev) <= tol * (abs(lam) + 1e-12): break
            lam_prev = lam
        return math.sqrt(max(lam, 0.0))

    def global_lipschitz(self, iters=100, tol=1e-4, verbose=False):
        t0 = time.time(); L = abs(self._top_eig(None, iters, tol))
        if verbose: print(f"  [L global] = {L:.4g}   ({time.time()-t0:.0f}s)", flush=True)
        return L

    def block_lipschitz(self, iters=100, tol=1e-4, verbose=False):
        out = {}; t0 = time.time()
        for b in range(self.num_blocks):
            out[b] = abs(self._top_eig(self.keep[b], iters, tol))
            if verbose:
                print(f"  [beta] block {b:2d}/{self.num_blocks-1}: {out[b]:.4g}"
                      f"   ({time.time()-t0:.0f}s)", flush=True)
        return out

    def coupling_matrix(self, iters=30, tol=1e-3, diag_from_blocks=None,
                        neighbors_only=False, verbose=False):
        B = self.num_blocks; M = np.zeros((B, B), dtype=np.float64)
        diag = diag_from_blocks or self.block_lipschitz(iters=max(iters, 60))
        for b in range(B): M[b, b] = diag[b]
        pairs = []
        for l in range(B):
            for lp in range(l + 1, B):
                if neighbors_only and not (lp == l + 1 or lp == B - 1): continue
                pairs.append((l, lp))
        t0 = time.time()
        for k, (l, lp) in enumerate(pairs):
            s = self._offdiag(self.keep[l], self.keep[lp], iters, tol)
            M[l, lp] = M[lp, l] = s
            if verbose and (k % 5 == 0 or k == len(pairs) - 1):
                print(f"  [coupling] {k+1:3d}/{len(pairs)} pairs   ({time.time()-t0:.0f}s)", flush=True)
        return M

    def summary(self, block_iters=100, global_iters=100, coupling=True,
                coupling_iters=30, neighbors_only=False, verbose=True):
        if verbose: print("[phase 1/3] per-block beta_l ...", flush=True)
        beta = self.block_lipschitz(iters=block_iters, verbose=verbose)
        if verbose: print("[phase 2/3] global L ...", flush=True)
        L = self.global_lipschitz(iters=global_iters, verbose=verbose)
        beta_max = max(beta.values())
        out = {"num_blocks": self.num_blocks, "block_nparams": self.block_nparams,
               "beta_l": beta, "L_global": L, "beta_max": beta_max,
               "beta_max_over_L": beta_max / L if L > 0 else float("nan")}
        if coupling:
            if verbose: print("[phase 3/3] inter-block coupling (heatmap) ...", flush=True)
            M = self.coupling_matrix(iters=coupling_iters, diag_from_blocks=beta,
                                     neighbors_only=neighbors_only, verbose=verbose)
            beta_bar = float((M.sum(axis=1) - np.diag(M)).max())
            out["coupling_matrix"] = M.tolist(); out["beta_bar"] = beta_bar
            out["diag_dominance"] = float(beta_bar / beta_max) if beta_max > 0 else float("nan")
            out["check_interlacing_betamax_le_L"] = bool(beta_max <= L + 1e-6)
            out["check_gershgorin_L_le_betamax_plus_betabar"] = bool(L <= beta_max + beta_bar + 1e-6)
        return out

def build_blocks_from_repo(model_name, num_classes, num_blocks):
    from models import load_model_and_blockmap
    from config import BN_BUFFERS
    model, block_map, nb = load_model_and_blockmap(model_name, num_classes, num_blocks)
    sd_keys = list(model.state_dict().keys())
    key2block = {k: block_map[i] for i, k in enumerate(sd_keys)}
    name2block = {n: key2block[n] for n, _ in model.named_parameters()
                  if not n.endswith(BN_BUFFERS)}
    return model, name2block

def load_checkpoint_(model, path):
    if path:
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=False)
    return model

def load_ndarrays_(model, ndarrays):
    sd_keys = list(model.state_dict().keys())
    model.load_state_dict({k: torch.tensor(a) for k, a in zip(sd_keys, ndarrays)}, strict=True)
    return model

_NORM = {"cifar10": ((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
         "cifar100": ((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))}

def build_batches(dataset, n_samples, batch_size, train, seed=0, tiny_path=None, data_root=None):
    from torch.utils.data import DataLoader, Subset
    from torchvision.transforms import Compose, ToTensor, Normalize
    dataset = dataset.lower()
    if dataset in ("cifar10", "cifar100"):
        from torchvision.datasets import CIFAR10, CIFAR100
        mean, std = _NORM[dataset]
        tf = Compose([ToTensor(), Normalize(mean, std)])
        root = data_root or (f"./{dataset.upper()}_data")
        cls = CIFAR10 if dataset == "cifar10" else CIFAR100
        ds = cls(root, train=train, download=True, transform=tf)
    elif dataset in ("tiny", "tiny-imagenet", "tinyimagenet"):
        from torchvision.datasets import ImageFolder
        mean, std = (0.4802,0.4481,0.3975),(0.2770,0.2691,0.2821)
        tf = Compose([ToTensor(), Normalize(mean, std)])
        assert tiny_path, "--tiny_path is required for Tiny-ImageNet"
        ds = ImageFolder(f"{tiny_path}/train", transform=tf)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_samples].tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False)
    return [(x, y) for x, y in dl]

def plot_block_bars(summary, title, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    beta = summary["beta_l"]; L = summary["L_global"]
    blocks = sorted(beta.keys()); vals = [beta[b] for b in blocks]
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(blocks)), 4))
    ax.bar(blocks, vals, color="#4C72B0", label=r"$\beta_\ell=\|H_{\ell\ell}\|_2$")
    ax.axhline(L, color="#C44E52", ls="--", lw=2, label=rf"$L=\|H\|_2={L:.3g}$")
    ax.set_yscale("log"); ax.set_xlabel("block $\\ell$"); ax.set_ylabel("top eigenvalue (log)")
    ax.set_title(title + rf"   ($\beta_{{\max}}/L={summary['beta_max_over_L']:.3f}$)")
    ax.set_xticks(blocks); ax.legend(); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_coupling_heatmap(summary, title, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = np.array(summary["coupling_matrix"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log10(M + 1e-12), cmap="viridis")
    ax.set_xlabel("block $\\ell'$"); ax.set_ylabel("block $\\ell$")
    ax.set_title(title + rf"   ($\bar\beta/\beta_{{\max}}={summary['diag_dominance']:.2f}$)")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}\|H_{\ell\ell'}\|_2$")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_ratio_vs_classes(points, out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    points = sorted(points, key=lambda p: p[1])
    C = [p[1] for p in points]; r1 = [p[2] for p in points]; r2 = [p[3] for p in points]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(C, r1, "o-", color="#4C72B0", label=r"$\beta_{\max}/L$")
    ax.plot(C, r2, "s--", color="#C44E52", label=r"$\bar\beta/\beta_{\max}$")
    for lbl, c, a, _ in points:
        ax.annotate(lbl, (c, a), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("number of classes $C$"); ax.set_ylabel("ratio")
    ax.set_ylim(0, 1.05); ax.set_title("Block-diagonal structure vs $C$"); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Per-block Hessian analysis (FedCarousel).")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--num_classes", type=int, default=100)
    ap.add_argument("--num_blocks", type=int, default=21)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n_samples", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--block_iters", type=int, default=100)
    ap.add_argument("--global_iters", type=int, default=100)
    ap.add_argument("--coupling_iters", type=int, default=30)
    ap.add_argument("--no_coupling", action="store_true")
    ap.add_argument("--offdiag_neighbors_only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out_dir", default="hessian_res")
    ap.add_argument("--tiny_path", default=None)
    args = ap.parse_args()
    print(f"[device] {args.device}  (cuda available: {torch.cuda.is_available()})", flush=True)
    if args.device == "cpu":
        print("[!] No GPU: this will be slow. Prefer a GPU node, and try --no_coupling first.", flush=True)
    model, name2block = build_blocks_from_repo(args.model, args.num_classes, args.num_blocks)
    model = load_checkpoint_(model, args.ckpt)
    batches = build_batches(args.dataset, args.n_samples, args.batch_size,
                            train=True, tiny_path=args.tiny_path)
    t0 = time.time()
    an = HessianBlockAnalyzer(model, name2block, batches, device=args.device)
    summ = an.summary(block_iters=args.block_iters, global_iters=args.global_iters,
                      coupling=not args.no_coupling, coupling_iters=args.coupling_iters,
                      neighbors_only=args.offdiag_neighbors_only)
    print(f"[total time] {time.time()-t0:.0f}s", flush=True)
    print(json.dumps({k: v for k, v in summ.items() if k != "coupling_matrix"}, indent=2))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/hessian_{args.tag}.json", "w") as f:
        json.dump(summ, f, indent=2)
    title = f"{args.model} / {args.dataset} (C={args.num_classes}, L={args.num_blocks})"
    plot_block_bars(summ, title, f"{args.out_dir}/hessian_blocks_{args.tag}.png")
    if not args.no_coupling:
        plot_coupling_heatmap(summ, title, f"{args.out_dir}/hessian_coupling_{args.tag}.png")
    print(f"\n[ok] JSON + PNG written to {args.out_dir}/ (tag={args.tag}).")

if __name__ == "__main__":
    main()