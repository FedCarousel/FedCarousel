# FedCarousel

Reference implementation of **"Rethinking Convergence Speed via Layer-Wise Client
Parallelism in Federated Learning"**.

FedCarousel partitions a neural network into `L` disjoint **layer-blocks** and
partitions the clients into `K` **groups**. In each communication round every
group trains a _different_ block, and the block assignment rotates cyclically so
that each block is trained by every group exactly once per cycle. Because the `L`
blocks advance **in parallel** rather than one after another, the server
reconstructs the complete model at every round, and the mismatch between the
reference a block was computed against and the current state of the other blocks
never exceeds a single round.

Each client uploads `d/L` parameters per round instead of `d`, which is where the
communication savings come from.

```
Round t          Round t+1        Round t+2
group 1 -> B1    group 1 -> B2    group 1 -> B3
group 2 -> B2    group 2 -> B3    group 2 -> B1     ... one full cycle
group 3 -> B3    group 3 -> B1    group 3 -> B2
        \___________ server reconstructs the full model each round ___________/
```

## Installation

```bash
git clone <repository-url> fedcarousel
cd fedcarousel

python -m venv .venv && source .venv/bin/activate    # or conda
pip install -r requirements.txt
```

Install the PyTorch build matching your CUDA version from
[pytorch.org](https://pytorch.org). CIFAR-10 and CIFAR-100 are downloaded
automatically on first use; Tiny-ImageNet has to be fetched manually (see
`scripts/run_tiny_imagenet.sh`).

### Check the installation

```bash
bash scripts/quickstart.sh
```

About 15 minutes on one GPU. It runs FedCarousel and FedAvg on a deliberately
small setting (20 clients, 22 rounds) and writes
`figures_quickstart/quickstart.png`. This is **not** a paper result — it only
verifies that the pipeline runs end to end.

---

A single run at reduced scale, to see the mechanism without the full budget:

```bash
python main.py --algo fedcarousel --variant carousel \
    --model ResNet8 --dataset cifar10 --num_classes 10 \
    --num_blocks 10 --num_clusters 10 --clustering kmeans \
    --training_scenario 3 --global_rounds_per_cycle 2 --block_repeat 1 \
    --num_clients 100 --num_rounds 150 --alpha 0.1 --seed 42 \
    --optimizer adam --lr_global 1e-4 --lr_partial 1e-4 \
    --epochs_global 2 --epochs_partial 2 \
    --out_dir results
```

## License

MIT — see [LICENSE](LICENSE).
