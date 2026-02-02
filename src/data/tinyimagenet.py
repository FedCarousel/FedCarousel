"""
Custom dataset loader for Tiny ImageNet.

Tiny ImageNet consists of:
- 200 classes
- 64x64 color images
- 500 training images per class (100,000 total)
- 50 validation images per class (10,000 total)
"""
import os
from typing import Tuple, Callable, Optional
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet dataset loader.
    
    Directory structure expected:
    root/
        train/
            n01443537/
                images/
                    n01443537_0.JPEG
                    ...
            ...
        val/
            images/
                val_0.JPEG
                ...
            val_annotations.txt
    """
    
    def __init__(self, 
                 root: str, 
                 split: str = 'train', 
                 transform: Optional[Callable] = None):
        """
        Args:
            root: Root directory of Tiny ImageNet dataset
            split: 'train' or 'val'
            transform: Optional transform to be applied on images
            
        Raises:
            ValueError: If split is not 'train' or 'val'
            FileNotFoundError: If dataset directory structure is invalid
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        if split not in ['train', 'val']:
            raise ValueError(f"Split must be 'train' or 'val', got '{split}'")
        
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")
        
        if split == 'train':
            self.data_dir = os.path.join(root, 'train')
            self.samples = self._load_train_samples()
        else:
            self.data_dir = os.path.join(root, 'val')
            self.samples = self._load_val_samples()
    
    def _load_train_samples(self) -> list:
        """Load training samples from directory structure."""
        samples = []
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Training directory not found: {self.data_dir}")
        
        class_folders = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_folders)}
        
        for class_name in class_folders:
            class_dir = os.path.join(self.data_dir, class_name, 'images')
            if not os.path.isdir(class_dir):
                continue
            
            label = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, label))
        
        if len(samples) == 0:
            raise RuntimeError(f"No training images found in {self.data_dir}")
        
        return samples
    
    def _load_val_samples(self) -> list:
        """Load validation samples from annotations file."""
        samples = []
        val_annotations = os.path.join(self.root, 'val', 'val_annotations.txt')
        
        if not os.path.exists(val_annotations):
            raise FileNotFoundError(f"Validation annotations not found: {val_annotations}")
        
        # Build class_to_idx from training directory
        train_dir = os.path.join(self.root, 'train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory needed for class mapping: {train_dir}")
        
        class_folders = sorted(os.listdir(train_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_folders)}
        
        # Parse validation annotations
        with open(val_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                    
                img_name = parts[0]
                class_name = parts[1]
                
                img_path = os.path.join(self.data_dir, 'images', img_name)
                
                if class_name not in self.class_to_idx:
                    continue
                    
                label = self.class_to_idx[class_name]
                samples.append((img_path, label))
        
        if len(samples) == 0:
            raise RuntimeError(f"No validation images found in {self.data_dir}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label) where image is a PIL Image
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """
        Get the distribution of classes in this dataset.
        
        Returns:
            Dictionary mapping class_idx to count
        """
        distribution = {}
        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
