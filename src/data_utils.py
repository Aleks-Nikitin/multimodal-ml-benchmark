import os
import numpy as np
from torchvision.datasets import CIFAR10
from sklearn.model_selection import StratifiedShuffleSplit

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_cifar_subset(data_dir="./data", n_train=5000, n_test=1000, seed=42):
    os.makedirs(data_dir, exist_ok=True)
    train_ds = CIFAR10(root=data_dir, train=True, download=True)
    test_ds = CIFAR10(root=data_dir, train=False, download=True)

    X_train_full, y_train_full = train_ds.data, np.array(train_ds.targets)
    X_test_full, y_test_full = test_ds.data, np.array(test_ds.targets)

    sss_train = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=seed)
    train_idx, _ = next(sss_train.split(X_train_full, y_train_full))

    sss_test = StratifiedShuffleSplit(n_splits=1, train_size=n_test, random_state=seed)
    test_idx, _ = next(sss_test.split(X_test_full, y_test_full))

    return (
        X_train_full[train_idx], y_train_full[train_idx],
        X_test_full[test_idx], y_test_full[test_idx]
    )