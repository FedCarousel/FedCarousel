# data.py
# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading, Dirichlet partitioning across clients, and client grouping.
#
# Supported datasets: CIFAR-10, CIFAR-100, Tiny-ImageNet (200 classes) and
# EMNIST/FEMNIST. Statistical heterogeneity is induced by a Dirichlet partition
# Dir_N(alpha); a smaller alpha means a stronger label skew.
# ─────────────────────────────────────────────────────────────────────────────

import shutil as _shutil

# Some EMNIST archives ship without every .gz split; ignore the missing ones
# instead of crashing during the torchvision extraction step.
_orig_rmtree = _shutil.rmtree


def _safe_rmtree(path, *args, **kwargs):
    try:
        _orig_rmtree(path, *args, **kwargs)
    except FileNotFoundError:
        pass


_shutil.rmtree = _safe_rmtree

import os
import random
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR100, CIFAR10, EMNIST, ImageFolder
from torchvision.transforms import (
    ToTensor, Normalize, Compose,
    RandomCrop, RandomHorizontalFlip, Lambda,
)
from sklearn.cluster import KMeans

from config import RUN_CFG


def set_seed(seed: int) -> None:
    """Full determinism: python / numpy / torch / cuDNN.

    Without the last four lines, two identical runs diverge by 1-3 accuracy
    points (data augmentation + non-deterministic cuDNN kernels), which makes
    any controlled comparison meaningless.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """Reseed each DataLoader worker from the process-level torch seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dirichlet_indices(labels: np.ndarray, num_clients: int,
                            alpha: float, seed: int) -> List[List[int]]:
    """Partition sample indices across clients following Dir_N(alpha)."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        cut_points = (np.cumsum(proportions) * len(idx)).astype(int)
        cut_points = np.minimum(cut_points, len(idx))
        cut_points = np.maximum.accumulate(cut_points)
        cut_points[-1] = len(idx)
        splits = np.split(idx, cut_points[:-1])
        for cid, part in enumerate(splits):
            client_indices[cid].extend(part.tolist())

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def build_random_uniform_clusters(num_clients: int, num_clusters: int,
                                  seed: int) -> np.ndarray:
    """Uniform random assignment of clients to groups.

    Each group receives exactly (or within +/-1 of) num_clients // num_clusters
    clients, which keeps the per-block aggregation weights balanced.
    """
    rng = np.random.RandomState(seed)
    assignments = np.array([i % num_clusters for i in range(num_clients)])
    rng.shuffle(assignments)
    return assignments


class TinyImageNetVal(Dataset):
    """Tiny-ImageNet validation split.

    val/ is a flat folder plus val_annotations.txt, so ImageFolder cannot be
    used directly. The same wnid -> index mapping as the training split is
    reused, otherwise the labels would be inconsistent.
    """

    def __init__(self, val_root, wnid_to_idx, transform=None):
        self.transform = transform
        self.samples, self.targets = [], []
        ann_file = os.path.join(val_root, "val_annotations.txt")
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, wnid = parts[0], parts[1]
                if wnid not in wnid_to_idx:
                    continue
                target = wnid_to_idx[wnid]
                self.samples.append((os.path.join(val_root, "images", fname), target))
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


def prepare_dataset_and_clusters():
    """Build the federated dataset.

    Returns
    -------
    trainloaders, valloaders : one DataLoader per client
    testloader               : centralized test set used for server evaluation
    cluster_assignments      : np.ndarray, group id per client
    client_to_cluster        : dict, client id -> group id
    server_loader            : DataLoader for server-side refinement (or None)
    trainset                 : the underlying training dataset
    server_pool_idx          : indices held by the server (or None)
    """
    set_seed(RUN_CFG["seed"])
    dataset = RUN_CFG.get("dataset", "cifar10").lower()

    # ── Dataset selection ────────────────────────────────────────────────────
    if dataset in ("tiny_imagenet", "tinyimagenet", "tiny-imagenet"):
        data_dir = RUN_CFG.get("tiny_imagenet_path", "./data/tiny-imagenet-200")
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2770, 0.2691, 0.2821)

        if RUN_CFG["use_data_augmentation"]:
            train_transform = Compose([
                RandomCrop(64, padding=4), RandomHorizontalFlip(),
                ToTensor(), Normalize(mean, std),
            ])
        else:
            train_transform = Compose([ToTensor(), Normalize(mean, std)])
        test_transform = Compose([ToTensor(), Normalize(mean, std)])

        trainset = ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
        wnid_to_idx = trainset.class_to_idx      # same mapping for val (crucial)
        testset = TinyImageNetVal(os.path.join(data_dir, "val"), wnid_to_idx,
                                  transform=test_transform)

        print(f"[DATA] tiny-imagenet | train={len(trainset)} val={len(testset)} "
              f"| classes={len(trainset.classes)}")

    elif dataset == "femnist":
        data_dir = "./EMNIST_data"
        mean, std = (0.1307,), (0.3081,)
        emnist_split = "byclass"                 # 62 classes
        # EMNIST images are transposed in torchvision; fix the orientation.
        fix_orient = Lambda(lambda x: x.squeeze(0).T.unsqueeze(0))

        train_transform = Compose([ToTensor(), fix_orient, Normalize(mean, std)])
        test_transform = Compose([ToTensor(), fix_orient, Normalize(mean, std)])

        trainset = EMNIST(data_dir, split=emnist_split, train=True,
                          download=True, transform=train_transform)
        testset = EMNIST(data_dir, split=emnist_split, train=False,
                         download=True, transform=test_transform)

    elif dataset == "cifar10" or RUN_CFG["num_classes"] == 10:
        data_dir = "./CIFAR10_data"
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        if RUN_CFG["use_data_augmentation"]:
            train_transform = Compose([
                RandomCrop(32, padding=4), RandomHorizontalFlip(),
                ToTensor(), Normalize(mean, std),
            ])
        else:
            train_transform = Compose([ToTensor(), Normalize(mean, std)])
        test_transform = Compose([ToTensor(), Normalize(mean, std)])
        trainset = CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        testset = CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    else:   # cifar100
        data_dir = "./CIFAR100_data"
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        if RUN_CFG["use_data_augmentation"]:
            train_transform = Compose([
                RandomCrop(32, padding=4), RandomHorizontalFlip(),
                ToTensor(), Normalize(mean, std),
            ])
        else:
            train_transform = Compose([ToTensor(), Normalize(mean, std)])
        test_transform = Compose([ToTensor(), Normalize(mean, std)])
        trainset = CIFAR100(data_dir, train=True, download=True, transform=train_transform)
        testset = CIFAR100(data_dir, train=False, download=True, transform=test_transform)

    labels = np.asarray(trainset.targets)
    n_total = len(trainset)
    rng_split = np.random.RandomState(RUN_CFG["seed"])
    all_idx = rng_split.permutation(n_total).tolist()

    # ── Optional server-side pool (server refinement ablation) ───────────────
    server_pool_idx = None
    server_loader = None
    if RUN_CFG.get("server_finetune", False):
        frac = RUN_CFG.get("server_finetune_fraction", 0.05)
        n_server = max(32, int(n_total * frac))
        # The server pool may be larger than the subset actually used, so that a
        # different subset can be redrawn every cycle.
        #   mult = 1 -> pool == subset: the server sees EXACTLY `frac` of the
        #               data and resampling is a no-op. This is the only setting
        #               under which "the server holds X% of the data" is exact.
        #   mult > 1 -> the server ends up seeing mult x frac of the data.
        mult = float(RUN_CFG.get("server_ft_pool_mult", 1.0))
        pool_size = min(n_total, int(n_server * mult))

        # Two regimes, to be stated explicitly when reporting results:
        #
        # DISJOINT (server_ft_disjoint=True): the server holds data the clients
        #   do NOT have — the standard "proxy / public data" setting (FedDF,
        #   SemiFL). Cost: the clients lose that data, which becomes impractical
        #   beyond roughly 10%.
        #
        # OVERLAP (server_ft_disjoint=False): the server samples from the SAME
        #   pool as the clients, which keep 100% of the data. This allows large
        #   server fractions without starving the clients, but assumes the
        #   server can read raw client data.
        disjoint = bool(RUN_CFG.get("server_ft_disjoint", True))

        if disjoint:
            if pool_size >= n_total:
                raise ValueError(
                    f"[DATA] Server pool = {pool_size} >= dataset ({n_total}): "
                    f"the clients would have no data left. Reduce "
                    f"server_finetune_fraction ({frac}) or server_ft_pool_mult "
                    f"({mult}), or set server_ft_disjoint=False."
                )
            server_pool_idx = all_idx[:pool_size]
            client_pool = all_idx[pool_size:]
        else:
            server_pool_idx = all_idx[:pool_size]
            client_pool = all_idx

        overlap_n = len(set(server_pool_idx) & set(client_pool))
        print(f"[DATA] server_ft: regime={'DISJOINT' if disjoint else 'OVERLAP'} "
              f"| server={pool_size} ({pool_size / n_total:.1%}) "
              f"| clients={len(client_pool)} ({len(client_pool) / n_total:.1%}) "
              f"-> {len(client_pool) // RUN_CFG['num_clients']} samples/client "
              f"| overlap={overlap_n} images "
              f"| pool_mult={mult} resample={RUN_CFG.get('server_ft_resample', False)}")

        # Initial loader; the strategy rebuilds it every cycle if resampling.
        first_subset = Subset(trainset, server_pool_idx[:n_server])
        server_loader = DataLoader(first_subset, batch_size=RUN_CFG["batch_size"],
                                   shuffle=True, num_workers=2, pin_memory=True)
        print(f"[DATA] Server pool: {pool_size} samples, "
              f"sampling {n_server} per cycle")
    else:
        client_pool = all_idx

    pool_labels = labels[client_pool]

    testloader = DataLoader(testset, batch_size=RUN_CFG["batch_size"],
                            shuffle=False, num_workers=2)

    # ── Dirichlet partition over the client pool ─────────────────────────────
    client_indices_local = build_dirichlet_indices(
        pool_labels, RUN_CFG["num_clients"], RUN_CFG["alpha"], RUN_CFG["seed"]
    )
    client_indices = [
        [client_pool[i] for i in local_idxs]
        for local_idxs in client_indices_local
    ]

    # ── Client grouping ──────────────────────────────────────────────────────
    clustering_mode = RUN_CFG.get("clustering_mode", "random").lower()
    if clustering_mode == "kmeans":
        # Heterogeneity-aware grouping: cluster clients by their label
        # distribution so that intra-group heterogeneity (psi) is small.
        C = RUN_CFG["num_classes"]
        sig = np.zeros((RUN_CFG["num_clients"], C), dtype=np.float32)
        for cid in range(RUN_CFG["num_clients"]):
            idxs = client_indices[cid]
            if not idxs:
                continue
            y = labels[idxs]
            counts = np.bincount(y, minlength=C).astype(np.float32)
            sig[cid] = counts / counts.sum()
        kmeans = KMeans(n_clusters=RUN_CFG["num_clusters"],
                        random_state=RUN_CFG["seed"], n_init=10)
        cluster_assignments = kmeans.fit_predict(sig)
    else:
        cluster_assignments = build_random_uniform_clusters(
            RUN_CFG["num_clients"], RUN_CFG["num_clusters"], RUN_CFG["seed"]
        )

    sizes = {k: int((cluster_assignments == k).sum())
             for k in range(RUN_CFG["num_clusters"])}
    print(f"[DATA] dataset={dataset} | clustering={clustering_mode} | sizes={sizes}")

    client_to_cluster = {cid: int(cluster_assignments[cid])
                         for cid in range(RUN_CFG["num_clients"])}

    # ── Per-client DataLoaders ───────────────────────────────────────────────
    rng = np.random.RandomState(RUN_CFG["seed"])
    trainloaders, valloaders = [], []

    for cid in range(RUN_CFG["num_clients"]):
        idxs = np.array(client_indices[cid], dtype=int)
        if idxs.size == 0:
            trainloaders.append(None)
            valloaders.append(None)
            continue
        rng.shuffle(idxs)
        n = len(idxs)
        val_size = max(1, n // 10)
        val_idxs = idxs[:val_size].tolist()
        tr_idxs = idxs[val_size:].tolist()
        if not tr_idxs:
            tr_idxs, val_idxs = val_idxs, []

        train_ds = Subset(trainset, tr_idxs)
        val_ds = Subset(trainset, val_idxs) if val_idxs else None

        # Per-client generator + worker_init_fn: without them, batch order and
        # augmentation draws depend on the Ray workers' global RNG and the run
        # is not reproducible.
        _g = torch.Generator()
        _g.manual_seed(RUN_CFG["seed"] * 100003 + cid)

        train_dl = DataLoader(train_ds,
                              batch_size=min(RUN_CFG["batch_size"], len(train_ds)),
                              shuffle=True, num_workers=2,
                              drop_last=True, pin_memory=True,
                              generator=_g, worker_init_fn=_seed_worker)
        val_dl = DataLoader(val_ds,
                            batch_size=min(RUN_CFG["batch_size"], len(val_ds)),
                            shuffle=False, num_workers=2,
                            pin_memory=True) if val_ds else None
        trainloaders.append(train_dl)
        valloaders.append(val_dl)

    return (trainloaders, valloaders, testloader, cluster_assignments,
            client_to_cluster, server_loader, trainset, server_pool_idx)


# Backward-compatible alias (the grouping is not necessarily k-means based).
prepare_dataset_and_kmeans_clusters = prepare_dataset_and_clusters
