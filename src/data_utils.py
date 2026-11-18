import os
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_cifar_subset(data_dir="./data", n_train=1000, n_test=250, seed=42):
    """Downloads a compact, stratified CIFAR-10 subset (1000 train / 250 test) with minimal RAM/disk footprint."""
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, f"cifar_subset_{n_train}_{n_test}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    print("Fetching compact CIFAR-10 slice from fast CDN...")
    train_ds = load_dataset("uoft-cs/cifar10", split="train[:5000]")
    test_ds = load_dataset("uoft-cs/cifar10", split="test[:1500]")

    X_train_pool = np.stack([np.array(img) for img in train_ds["img"]])
    y_train_pool = np.array(train_ds["label"])
    X_test_pool = np.stack([np.array(img) for img in test_ds["img"]])
    y_test_pool = np.array(test_ds["label"])

    sss_train = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=seed)
    train_idx, _ = next(sss_train.split(X_train_pool, y_train_pool))

    sss_test = StratifiedShuffleSplit(n_splits=1, train_size=n_test, random_state=seed)
    test_idx, _ = next(sss_test.split(X_test_pool, y_test_pool))

    X_train, y_train = X_train_pool[train_idx], y_train_pool[train_idx]
    X_test, y_test = X_test_pool[test_idx], y_test_pool[test_idx]

    np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test