# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for FedCarousel.
#
# Every entry can be overridden in two ways:
#   1. command-line flags of main.py  (see main.py::parse_args)
#   2. FL_* environment variables     (used to propagate the config to Ray
#      workers, see the _simple_env table at the bottom of this file)
#
# The command line takes precedence over the environment, which takes
# precedence over the defaults below.
# ─────────────────────────────────────────────────────────────────────────────

RUN_CFG = {
    # ── Reproducibility ──────────────────────────────────────────────────────
    "seed": 42,

    # ── Model / data ─────────────────────────────────────────────────────────
    "MODEL_NAME":  "ResNet18",     # ResNet8 | ResNet18 | ResNet34 | SimpleMLP | CNNFemnist
    "dataset":     "cifar100",     # cifar10 | cifar100 | tiny_imagenet | femnist
    "num_classes": 100,            # 10 | 100 | 200 | 62
    "batch_size":  64,
    "use_data_augmentation": True,
    "tiny_imagenet_path": "./data/tiny-imagenet-200",

    # ── Federated setting ────────────────────────────────────────────────────
    "alpha":        50,            # Dirichlet concentration: 0.1 / 0.5 / 50
    "num_clients":  100,           # N
    "num_clusters": 10,            # K  (the paper uses K = L)
    "num_blocks":   10,            # L  (number of layer-blocks)

    # Client grouping strategy.
    #   "random" : uniform random partition of clients into K groups.
    #   "kmeans" : groups built by k-means on the clients' label histograms,
    #              i.e. the heterogeneity-aware grouping of the paper.
    "clustering_mode": "random",

    # ── Block scheduling variant ─────────────────────────────────────────────
    # "carousel" : block = (cluster_id + position) % num_blocks
    #              -> the L blocks advance IN PARALLEL, one group per block
    #                 (FedCarousel, Jacobi-style).
    # "fedpart"  : block = position % num_blocks, identical for ALL clients
    #              -> a single block advances per round (FedPart baseline,
    #                 Gauss-Seidel-style), re-implemented inside THIS framework
    #                 so that the comparison is strictly controlled.
    "variant": "carousel",

    # ── Training schedule ────────────────────────────────────────────────────
    # scenario 1 : purely partial (no full-model round at all)
    # scenario 2 : `warmup_rounds` full-model rounds, then purely partial
    # scenario 3 : cyclic — G full-model rounds + L*block_repeat partial rounds
    # scenario 4 : cyclic like 3 until round T, then a fixed mode afterwards
    "training_scenario": 3,
    "global_rounds_per_cycle": 2,   # G: full-model synchronization rounds / cycle
    "block_repeat":  1,             # consecutive rounds spent on each block
    "warmup_rounds": 1,             # scenario 2 only

    # scenario 4 only
    "switch_round":     300,        # T: switching round
    "post_switch_mode": "partial",  # "partial" (drop sync) | "global" (full model)

    # Place the synchronization round(s) at the END of the cycle instead of the
    # beginning. False reproduces the historical behaviour.
    "sync_at_end_of_cycle": True,

    # ── Dynamic re-clustering (optional ablation) ────────────────────────────
    "dynamic_clustering":   False,  # rebuild the client groups periodically
    "cluster_rebuild_freq": 0,      # rebuild every N rounds (1 = every round,
                                    # 0 = never, even if dynamic_clustering=True)

    # ── Optimization ─────────────────────────────────────────────────────────
    "optimizer": "adam",            # "adam" | "sgd"

    # Adam learning rates
    "lr_global_adam":         0.001,
    "lr_partial_adam":        0.001,
    "lr_partial_block4_adam": 0.001,   # learning rate of the last block (head)

    # SGD learning rates
    "lr_global_sgd":         0.01,
    "lr_partial_sgd":        0.005,
    "lr_partial_block4_sgd": 0.005,

    "weight_decay":   1e-4,
    "momentum":       0.9,
    "epochs_global":  2,            # E, local epochs during full-model rounds
    "epochs_partial": 2,            # E, local epochs during block rounds
    "grad_clip_norm": 1,

    # ── BatchNorm handling ───────────────────────────────────────────────────
    "freeze_bn_gamma_beta_in_partial": False,

    # Aggregate the BN buffers (running_mean / running_var) of the ACTIVE block
    # during partial rounds. False = BN statistics frozen outside full rounds.
    "bn_partial_agg": False,

    # BN consistency. A single rule matters: the normalization seen at training
    # time must be the one used at evaluation time. Two settings are consistent:
    #
    #   A) bn_agg_all_blocks=True + bn_train_all_blocks=True
    #      All BN layers in train() on the client side, all buffers aggregated
    #      over all clients every round. Negligible uplink cost. RECOMMENDED,
    #      and the setting used for every experiment reported in the paper.
    #
    #   B) bn_agg_all_blocks=False + bn_train_all_blocks=False + BN in eval()
    #      gamma/beta are trained against the same running statistics as the
    #      ones used at evaluation, and nothing is aggregated.
    #
    # Any hybrid (gamma/beta trained against batch statistics while the buffers
    # are discarded) yields unstable, non-reproducible results.
    "bn_agg_all_blocks":   True,    # server: aggregate running stats of ALL blocks
    "bn_train_all_blocks": True,    # client: all BN layers in train() in partial rounds

    # ── Server-side refinement (Section VI-D of the paper) ───────────────────
    "server_finetune":          False,
    "server_finetune_fraction": 0.05,   # fraction of the training set held by the server
    "server_finetune_epochs":   4,
    "server_finetune_lr":       1e-3,
    "server_finetune_freq":     1,      # fine-tune every N cycles

    # Size of the server pool, as a multiple of the subset actually used.
    #   1.0 = the pool IS the subset: the server holds exactly
    #         `server_finetune_fraction` of the data and always sees the same
    #         samples (the honest reading of "the server holds X% of the data").
    #   >1  = the server redraws a different subset every cycle and therefore
    #         ends up seeing mult x fraction of the data.
    "server_ft_pool_mult":    1.0,
    "server_ft_resample":     False,    # no effect when pool_mult == 1.0
    "server_ft_weight_decay": 1e-4,

    # True  = the server holds data the clients do NOT have (standard "proxy
    #         data" setting, but it removes that data from the client pool).
    # False = the server samples from the same pool; clients keep 100% of the
    #         data. Required for server fractions >= 20%.
    "server_ft_disjoint": True,

    # Fraction of the local data used during full-model rounds (1 = all of it).
    "global_data_fraction": 1,

    # ── Simulation (Flower / Ray) ────────────────────────────────────────────
    "num_rounds": 150,
    "client_resources": {"num_cpus": 2, "num_gpus": 0.22},
    "ray_init_args": {
        "num_gpus": 1,
        "num_cpus": 20,
        "object_store_memory": 10 * 1024 * 1024 * 1024,
        "_memory": 100 * 1024 * 1024 * 1024,
        "include_dashboard": False,
    },

    # ── Logging / diagnostics ────────────────────────────────────────────────
    "debug": True,
    "debug_print_mapping": True,
    "debug_client_trainables": True,
    "debug_delta_norm": True,
    "debug_counts": True,

    # ── Diagnostic metrics (Section III-B: Magnitude Gradient & Directional
    #    Consistency, used to produce Figures 2 and 3) ─────────────────────────
    "tau_dc": 8,            # sliding window length for directional consistency
    "eps_metric": 1e-12,
    "debug_metrics": True,
}

BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")

RUN_CFG["num_partial_layers"] = RUN_CFG["num_blocks"] * RUN_CFG.get("block_repeat", 1)
RUN_CFG["cycle_length"] = RUN_CFG["global_rounds_per_cycle"] + RUN_CFG["num_partial_layers"]

ALGO = ""            # "fedavg" | "fedprox" | "fedcarousel" | "scaffold"
FEDPROX_MU = 0.01    # proximal coefficient of FedProx


# ─────────────────────────────────────────────────────────────────────────────
# Environment-variable overrides.
#
# Flower runs each client inside a Ray worker, i.e. a separate process that
# re-imports this module. Passing the configuration through FL_* environment
# variables is what keeps server and clients in sync.
# ─────────────────────────────────────────────────────────────────────────────
import os

_simple_env = {
    "MODEL_NAME":     ("FL_MODEL",       str),
    "num_classes":    ("FL_NUM_CLASSES", int),
    "alpha":          ("FL_ALPHA",       float),
    "num_rounds":     ("FL_NUM_ROUNDS",  int),
    "seed":           ("FL_SEED",        int),
    "optimizer":      ("FL_OPTIMIZER",   str),
    "epochs_global":  ("FL_EPOCHS_G",    int),
    "epochs_partial": ("FL_EPOCHS_P",    int),
    "num_blocks":     ("FL_NUM_BLOCKS",  int),
    "num_clusters":   ("FL_NUM_CLUSTERS", int),
    "clustering_mode": ("FL_CLUSTERING", str),
    "global_rounds_per_cycle": ("FL_GLOBAL_PER_CYCLE", int),
    "dynamic_clustering":   ("FL_DYNAMIC_CLUSTERING",
                             lambda x: x.strip().lower() in ("true", "1", "yes")),
    "cluster_rebuild_freq": ("FL_REBUILD_FREQ", int),
    "variant":           ("FL_VARIANT",      str),
    "training_scenario": ("FL_SCENARIO",     int),
    "block_repeat":      ("FL_BLOCK_REPEAT", int),
    "warmup_rounds":     ("FL_WARMUP",       int),
    "dataset":           ("FL_DATASET",      str),
    "lr_partial_adam":   ("FL_LR_PARTIAL",   float),

    # Server-side refinement
    "server_finetune":          ("FL_SERVER_FINETUNE",
                                 lambda x: x.strip().lower() in ("true", "1", "yes")),
    "server_finetune_fraction": ("FL_FINETUNE_FRAC",   float),
    "server_finetune_epochs":   ("FL_FINETUNE_EPOCHS", int),
    "server_finetune_lr":       ("FL_FINETUNE_LR",     float),
    "server_finetune_freq":     ("FL_FINETUNE_FREQ",   int),
    "server_ft_resample":       ("FL_FT_RESAMPLE",
                                 lambda x: x.strip().lower() in ("true", "1", "yes")),
    "server_ft_weight_decay":   ("FL_FT_WD",        float),
    "server_ft_pool_mult":      ("FL_FT_POOL_MULT", float),
    "server_ft_disjoint":       ("FL_FT_DISJOINT",
                                 lambda x: x.strip().lower() in ("true", "1", "yes")),

    # BatchNorm
    "bn_partial_agg":      ("FL_BN_PARTIAL_AGG",
                            lambda x: x.strip().lower() in ("true", "1", "yes")),
    "bn_agg_all_blocks":   ("FL_BN_AGG_ALL",
                            lambda x: x.strip().lower() in ("true", "1", "yes")),
    "bn_train_all_blocks": ("FL_BN_TRAIN_ALL",
                            lambda x: x.strip().lower() in ("true", "1", "yes")),

    # Scheduling
    "switch_round":         ("FL_SWITCH_ROUND",     int),
    "post_switch_mode":     ("FL_POST_SWITCH_MODE", str),
    "sync_at_end_of_cycle": ("FL_SYNC_AT_END",
                             lambda x: x.strip().lower() in ("true", "1", "yes")),

    "global_data_fraction": ("FL_GLOBAL_DATA_FRAC", float),
    "tiny_imagenet_path":   ("FL_TINY_PATH",        str),
}

for cfg_key, (env_key, cast) in _simple_env.items():
    val = os.environ.get(env_key)
    if val is not None:
        RUN_CFG[cfg_key] = cast(val)

# Global learning rate: override the right key depending on the resolved optimizer.
if os.environ.get("FL_LR_GLOBAL"):
    _lr_val = float(os.environ["FL_LR_GLOBAL"])
    if RUN_CFG["optimizer"].lower() == "sgd":
        RUN_CFG["lr_global_sgd"] = _lr_val
    else:
        RUN_CFG["lr_global_adam"] = _lr_val

RUN_CFG["num_partial_layers"] = RUN_CFG["num_blocks"] * RUN_CFG.get("block_repeat", 1)
RUN_CFG["cycle_length"]       = RUN_CFG["global_rounds_per_cycle"] + RUN_CFG["num_partial_layers"]
