# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Clone and Install

```bash
git clone
cd FedCarousel
pip install -r requirements.txt
```

### 2. Download Datasets

**CIFAR-10 and CIFAR-100** will download automatically.

**Tiny ImageNet** needs manual download:

```bash
# Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/
```

### 3. Run Your First Experiment

```bash
# Start with CIFAR-10 (smallest, fastest)
python experiments/run_cifar10.py
```

### 4. View Results

Results are saved in JSON format:

```bash
cat results/cifar10_resnet8_random.json
```

## 📊 Quick Configuration Changes

### Change number of clients

Edit `config/cifar10_config.py`:

```python
config = {
    'num_clients': 50,  # Changed from 100
    # ...
}
```

### Change learning rate

```python
config = {
    'learning_rate': 0.005,  # Changed from 0.01
    # ...
}
```

### Use different clustering

```python
config = {
    'clustering_method': 'kmeans',  # Changed from 'random'
    # ...
}
```
