# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Entry point: runs one federated simulation (Flower + Ray) and writes a JSON
# file containing the accuracy/loss curves, the per-block diagnostic metrics and
# the exact configuration used.
#
# Example
# -------
#   python main.py --algo fedcarousel --model ResNet8 --dataset cifar10 \
#       --num_classes 10 --alpha 0.1 --num_blocks 10 --num_clusters 10 \
#       --training_scenario 3 --global_rounds_per_cycle 2 --num_rounds 150 \
#       --optimizer adam --lr_global 1e-4 --lr_partial 1e-4 \
#       --epochs_global 2 --epochs_partial 2 --seed 42
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
from collections import OrderedDict

import torch
from torch import nn
import flwr as fl
from flwr.common import Context, parameters_to_ndarrays

from config import RUN_CFG
from data import prepare_dataset_and_clusters
from client import LayerWiseClient
from models import load_model_and_blockmap
from strategies.fedcarousel_strategy import LayerWiseFedAvg
from strategies.scaffold_strategy import SCAFFOLDStrategy


def parse_args():
    p = argparse.ArgumentParser(
        description="FedCarousel: layer-wise client parallelism in federated learning."
    )

    # ── Algorithm ────────────────────────────────────────────────────────────
    p.add_argument("--algo", type=str, default="fedcarousel",
                   choices=["fedavg", "fedprox", "fedcarousel", "scaffold"],
                   help="fedcarousel also covers the fedpart baseline via --variant")
    p.add_argument("--variant", type=str, default=None,
                   choices=["carousel", "fedpart"],
                   help="carousel = one block per group (parallel) | "
                        "fedpart = same block for every client (sequential baseline)")

    # ── Model and data ───────────────────────────────────────────────────────
    p.add_argument("--model", type=str, default=None,
                   choices=["ResNet34", "ResNet18", "ResNet8", "SimpleMLP", "CNNFemnist"])
    p.add_argument("--dataset", type=str, default=None,
                   choices=["cifar10", "cifar100", "femnist", "tiny_imagenet"])
    p.add_argument("--num_classes", type=int, default=None, choices=[10, 100, 62, 200])
    p.add_argument("--alpha", type=float, default=None,
                   help="Dirichlet concentration (0.1 / 0.5 / 50)")
    p.add_argument("--num_clients", type=int, default=None)

    # ── Layer-block decomposition and grouping ───────────────────────────────
    p.add_argument("--num_blocks", type=int, default=None, choices=[3, 4, 5, 8, 10, 21],
                   help="L: number of layer-blocks")
    p.add_argument("--num_clusters", type=int, default=None,
                   help="K: number of client groups (the paper uses K = L)")
    p.add_argument("--clustering", type=str, default=None, choices=["kmeans", "random"])
    p.add_argument("--dynamic_clustering", action="store_true", default=False,
                   help="rebuild the client groups periodically")
    p.add_argument("--cluster_rebuild_freq", type=int, default=None)

    # ── Training schedule ────────────────────────────────────────────────────
    p.add_argument("--training_scenario", type=int, default=None, choices=[1, 2, 3, 4],
                   help="1=purely partial | 2=warmup then partial | "
                        "3=cyclic (paper default) | 4=cyclic until T then fixed")
    p.add_argument("--global_rounds_per_cycle", type=int, default=None,
                   help="G: full-model synchronization rounds per cycle")
    p.add_argument("--block_repeat", type=int, default=None,
                   help="consecutive rounds spent on each block")
    p.add_argument("--warmup_rounds", type=int, default=None, help="scenario 2 only")
    p.add_argument("--switch_round", type=int, default=None, help="scenario 4 only: T")
    p.add_argument("--post_switch_mode", type=str, default=None,
                   choices=["partial", "global"], help="scenario 4 only")
    p.add_argument("--num_rounds", type=int, default=None)

    # ── Optimization ─────────────────────────────────────────────────────────
    p.add_argument("--optimizer", type=str, default=None, choices=["adam", "sgd"])
    p.add_argument("--lr_global", type=float, default=None)
    p.add_argument("--lr_partial", type=float, default=None)
    p.add_argument("--epochs_global", type=int, default=None)
    p.add_argument("--epochs_partial", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--global_data_fraction", type=float, default=None,
                   help="fraction of the local data used during full-model rounds")

    # ── Server-side refinement ───────────────────────────────────────────────
    p.add_argument("--server_finetune", action="store_true", default=False)
    p.add_argument("--server_finetune_fraction", type=float, default=None)
    p.add_argument("--server_finetune_epochs", type=int, default=None)
    p.add_argument("--server_finetune_lr", type=float, default=None)
    p.add_argument("--server_finetune_freq", type=int, default=None)

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument("--out_dir", type=str, default="results",
                   help="directory where the run JSON is written")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints",
                   help="directory for the final global model (used by the Hessian analysis)")
    p.add_argument("--save_checkpoint", action="store_true", default=False)

    return p.parse_args()


def apply_args_to_cfg(args):
    """Push the command-line values into RUN_CFG and the FL_* environment.

    The environment part matters: Flower clients run inside separate Ray worker
    processes that re-import config.py, so the configuration has to travel
    through environment variables to reach them.
    """
    # The optimizer must be resolved first: the learning-rate keys depend on it.
    if args.optimizer is not None:
        RUN_CFG["optimizer"] = args.optimizer
        os.environ["FL_OPTIMIZER"] = args.optimizer

    mapping = {
        "num_rounds":     ("FL_NUM_ROUNDS",  args.num_rounds,     int),
        "alpha":          ("FL_ALPHA",       args.alpha,          float),
        "seed":           ("FL_SEED",        args.seed,           int),
        "epochs_global":  ("FL_EPOCHS_G",    args.epochs_global,  int),
        "epochs_partial": ("FL_EPOCHS_P",    args.epochs_partial, int),
        "MODEL_NAME":     ("FL_MODEL",       args.model,          str),
        "num_classes":    ("FL_NUM_CLASSES", args.num_classes,    int),
        "num_clients":    ("FL_NUM_CLIENTS", args.num_clients,    int),
        "num_blocks":     ("FL_NUM_BLOCKS",  args.num_blocks,     int),
        "num_clusters":   ("FL_NUM_CLUSTERS", args.num_clusters,  int),
        "variant":        ("FL_VARIANT",     args.variant,        str),
        "clustering_mode": ("FL_CLUSTERING", args.clustering,     str),
        "global_rounds_per_cycle": ("FL_GLOBAL_PER_CYCLE", args.global_rounds_per_cycle, int),
        "training_scenario": ("FL_SCENARIO",     args.training_scenario, int),
        "block_repeat":      ("FL_BLOCK_REPEAT", args.block_repeat,      int),
        "warmup_rounds":     ("FL_WARMUP",       args.warmup_rounds,     int),
        "dataset":           ("FL_DATASET",      args.dataset,           str),
    }

    if args.switch_round is not None:
        RUN_CFG["switch_round"] = args.switch_round
        os.environ["FL_SWITCH_ROUND"] = str(args.switch_round)
    if args.post_switch_mode is not None:
        RUN_CFG["post_switch_mode"] = args.post_switch_mode
        os.environ["FL_POST_SWITCH_MODE"] = args.post_switch_mode

    if args.lr_partial is not None:
        if RUN_CFG["optimizer"].lower() == "sgd":
            RUN_CFG["lr_partial_sgd"] = args.lr_partial
            RUN_CFG["lr_partial_block4_sgd"] = args.lr_partial
        else:
            RUN_CFG["lr_partial_adam"] = args.lr_partial
            RUN_CFG["lr_partial_block4_adam"] = args.lr_partial
        os.environ["FL_LR_PARTIAL"] = str(args.lr_partial)

    # Dataset-specific defaults.
    if args.dataset == "tiny_imagenet":
        RUN_CFG["num_classes"] = 200
        os.environ["FL_NUM_CLASSES"] = "200"

    if args.dataset == "femnist":
        RUN_CFG["num_classes"] = 62
        RUN_CFG["num_blocks"] = 4
        RUN_CFG["num_clusters"] = 4
        os.environ["FL_NUM_CLASSES"] = "62"
        os.environ["FL_NUM_BLOCKS"] = "4"
        os.environ["FL_NUM_CLUSTERS"] = "4"

    for cfg_key, (env_key, val, cast) in mapping.items():
        if val is not None:
            RUN_CFG[cfg_key] = cast(val)
            os.environ[env_key] = str(val)

    # Global learning rate, after the optimizer has been resolved.
    if args.lr_global is not None:
        os.environ["FL_LR_GLOBAL"] = str(args.lr_global)
        if RUN_CFG["optimizer"].lower() == "sgd":
            RUN_CFG["lr_global_sgd"] = args.lr_global
        else:
            RUN_CFG["lr_global_adam"] = args.lr_global

    if args.dynamic_clustering:
        RUN_CFG["dynamic_clustering"] = True
        os.environ["FL_DYNAMIC_CLUSTERING"] = "true"
    if args.cluster_rebuild_freq is not None:
        RUN_CFG["cluster_rebuild_freq"] = args.cluster_rebuild_freq
        os.environ["FL_REBUILD_FREQ"] = str(args.cluster_rebuild_freq)

    if args.server_finetune:
        RUN_CFG["server_finetune"] = True
        os.environ["FL_SERVER_FINETUNE"] = "true"
    if args.server_finetune_fraction is not None:
        RUN_CFG["server_finetune_fraction"] = args.server_finetune_fraction
        os.environ["FL_FINETUNE_FRAC"] = str(args.server_finetune_fraction)
    if args.server_finetune_epochs is not None:
        RUN_CFG["server_finetune_epochs"] = args.server_finetune_epochs
        os.environ["FL_FINETUNE_EPOCHS"] = str(args.server_finetune_epochs)
    if args.server_finetune_lr is not None:
        RUN_CFG["server_finetune_lr"] = args.server_finetune_lr
        os.environ["FL_FINETUNE_LR"] = str(args.server_finetune_lr)
    if args.server_finetune_freq is not None:
        RUN_CFG["server_finetune_freq"] = args.server_finetune_freq
        os.environ["FL_FINETUNE_FREQ"] = str(args.server_finetune_freq)

    if args.global_data_fraction is not None:
        RUN_CFG["global_data_fraction"] = args.global_data_fraction
        os.environ["FL_GLOBAL_DATA_FRAC"] = str(args.global_data_fraction)

    # ── Derived quantities: cycle length ─────────────────────────────────────
    scenario = RUN_CFG.get("training_scenario")
    block_repeat = RUN_CFG.get("block_repeat", 1)
    G = RUN_CFG["global_rounds_per_cycle"]
    n_blocks = RUN_CFG["num_blocks"]

    RUN_CFG["num_partial_layers"] = n_blocks * block_repeat
    if scenario in (1, 2):
        # No full-model round inside the cycle.
        RUN_CFG["cycle_length"] = n_blocks * block_repeat
    else:  # scenarios 3 and 4
        RUN_CFG["cycle_length"] = G + n_blocks * block_repeat

    os.environ["FL_NUM_PARTIAL_LAYERS"] = str(RUN_CFG["num_partial_layers"])

    # Propagate every FL_* variable to the Ray workers.
    fl_env_vars = {k: v for k, v in os.environ.items() if k.startswith("FL_")}
    if fl_env_vars:
        RUN_CFG["ray_init_args"].setdefault("runtime_env", {})
        RUN_CFG["ray_init_args"]["runtime_env"]["env_vars"] = fl_env_vars


def get_evaluate_fn(testloader):
    """Centralized server-side evaluation on the held-out test set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model_and_blockmap(
        RUN_CFG["MODEL_NAME"], RUN_CFG["num_classes"], RUN_CFG["num_blocks"])
    model.to(device)

    def evaluate_fn(server_round, parameters, config):
        nds = parameters_to_ndarrays(parameters) if hasattr(parameters, "tensors") else parameters
        sd = OrderedDict({k: torch.tensor(v).to(device)
                          for k, v in zip(model.state_dict().keys(), nds)})
        model.load_state_dict(sd, strict=True)
        model.eval()
        crit = nn.CrossEntropyLoss()
        correct, tot_loss = 0, 0.0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                tot_loss += crit(out, y).item() * x.size(0)
                correct += out.argmax(1).eq(y).sum().item()
        acc = correct / len(testloader.dataset)
        return tot_loss / len(testloader.dataset), {"accuracy": acc}

    return evaluate_fn


def generate_client_fn(trainloaders, valloaders, client_to_cluster):
    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        clu = int(client_to_cluster[cid])
        return LayerWiseClient(trainloaders[cid], valloaders[cid], clu, cid).to_client()
    return client_fn


def build_output_name(algo: str) -> str:
    """Build a filename that uniquely identifies the run configuration."""
    dataset_tag = f"C{RUN_CFG['num_classes']}"
    model_tag = RUN_CFG["MODEL_NAME"]

    dyn_tag = (f"_dyn{RUN_CFG.get('cluster_rebuild_freq', 0)}"
               if RUN_CFG.get("dynamic_clustering", False) else "")

    opt = RUN_CFG["optimizer"].lower()
    lr_val = RUN_CFG["lr_global_adam"] if opt == "adam" else RUN_CFG["lr_global_sgd"]
    lr_tag = f"_lr{lr_val}"

    sc_tag = f"_sc{RUN_CFG.get('training_scenario', 1)}_rep{RUN_CFG.get('block_repeat', 1)}"
    if RUN_CFG.get("training_scenario") == 2:
        sc_tag += f"_warm{RUN_CFG.get('warmup_rounds', 5)}"
    if RUN_CFG.get("training_scenario") == 4:
        sc_tag += f"_T{RUN_CFG.get('switch_round')}_{RUN_CFG.get('post_switch_mode', 'partial')}"

    # The variant MUST appear in the name, otherwise a fedpart run overwrites
    # the matching carousel run and the comparison is lost.
    var_tag = f"_{RUN_CFG.get('variant', 'carousel')}"
    seed_tag = f"_s{RUN_CFG['seed']}"

    return (f"{algo}_{model_tag}_{dataset_tag}"
            f"{var_tag}{seed_tag}{sc_tag}"
            f"_alpha{RUN_CFG['alpha']}"
            f"_g{RUN_CFG['global_rounds_per_cycle']}"
            f"_{RUN_CFG['clustering_mode']}{dyn_tag}"
            f"_{RUN_CFG['optimizer']}{lr_tag}"
            f"_r{RUN_CFG['num_rounds']}"
            f"_B{RUN_CFG['num_blocks']}.json")


def main():
    args = parse_args()
    apply_args_to_cfg(args)
    algo = args.algo

    print(f"[RUN] algo={algo}")
    print("[RUN] RUN_CFG:")
    for k in sorted(RUN_CFG.keys()):
        print(f"  {k}: {RUN_CFG[k]}")

    (trainloaders, valloaders, testloader, cluster_assignments,
     client_to_cluster, server_loader, trainset, server_pool_idx) = \
        prepare_dataset_and_clusters()

    # ── Strategy ─────────────────────────────────────────────────────────────
    if algo == "fedcarousel":
        strategy = LayerWiseFedAvg(
            client_to_cluster=client_to_cluster,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            server_loader=server_loader,
            trainset=trainset,
            server_pool_idx=server_pool_idx,
            evaluate_fn=get_evaluate_fn(testloader),
        )

    elif algo == "scaffold":
        strategy = SCAFFOLDStrategy(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            evaluate_fn=get_evaluate_fn(testloader),
        )

    elif algo in ("fedavg", "fedprox"):
        # Inherit from LayerWiseFedAvg so that the baselines are evaluated and
        # logged through exactly the same code path as FedCarousel.
        class GlobalOnlyWithMetrics(LayerWiseFedAvg):
            def __init__(self, algo: str, **kwargs):
                super().__init__(**kwargs)
                self.algo = algo

            def configure_fit(self, server_round, parameters, client_manager):
                clients = client_manager.sample(
                    client_manager.num_available(),
                    min_num_clients=client_manager.num_available(),
                )
                ins = fl.common.FitIns(parameters,
                                       {"algo": self.algo, "round_num": int(server_round)})
                return [(c, ins) for c in clients]

            def get_mode(self, server_round: int) -> str:
                return "global"   # always full-model

        strategy = GlobalOnlyWithMetrics(
            algo=algo,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            evaluate_fn=get_evaluate_fn(testloader),
        )
    else:
        raise ValueError(algo)

    # ── Ray / Flower simulation ──────────────────────────────────────────────
    # Read the CPU/GPU budget from the scheduler rather than letting Ray probe
    # the whole machine (which ignores cgroup limits on shared clusters).
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    n_gpus = torch.cuda.device_count()

    ray_init_args = {
        "address": "local",
        "include_dashboard": False,
        "num_cpus": n_cpus,
        "num_gpus": n_gpus,
        "_temp_dir": os.environ.get("RAY_TMPDIR", "/tmp/ray_local"),
        "runtime_env": RUN_CFG["ray_init_args"].get("runtime_env", {}),
    }

    print("Starting simulation...")
    history = fl.simulation.start_simulation(
        client_fn=generate_client_fn(trainloaders, valloaders, client_to_cluster),
        num_clients=RUN_CFG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=RUN_CFG["num_rounds"]),
        strategy=strategy,
        client_resources=RUN_CFG["client_resources"],
        ray_init_args=ray_init_args,
    )

    # ── Optional checkpoint (input of analysis/hessian_block_analysis.py) ────
    if args.save_checkpoint:
        final_params = strategy.current_global_params
        if final_params is not None:
            m_ckpt, _, _ = load_model_and_blockmap(
                RUN_CFG["MODEL_NAME"], RUN_CFG["num_classes"], RUN_CFG["num_blocks"])
            sd_ckpt = OrderedDict({k: torch.tensor(v)
                                   for k, v in zip(m_ckpt.state_dict().keys(), final_params)})
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                args.ckpt_dir,
                f"global_{RUN_CFG['MODEL_NAME']}_C{RUN_CFG['num_classes']}"
                f"_a{RUN_CFG['alpha']}_B{RUN_CFG['num_blocks']}_r{RUN_CFG['num_rounds']}.pt")
            torch.save(sd_ckpt, ckpt_path)
            print(f"[ckpt] final global model saved -> {ckpt_path}")
        else:
            print("[ckpt] current_global_params is None, nothing to save")

    # ── Results ──────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "accuracy": history.metrics_centralized,
        "loss_centralised": history.losses_centralized,
        "loss_distributed": history.losses_distributed,
        "run_cfg": {k: v for k, v in RUN_CFG.items() if k != "ray_init_args"},
        "server_metrics_history": strategy.get_metrics_history(),
    }

    out_path = os.path.join(args.out_dir, build_output_name(algo))
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
