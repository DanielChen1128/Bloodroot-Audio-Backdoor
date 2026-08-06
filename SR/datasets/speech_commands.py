"""Feature dataset used by the Speech Commands victim models."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from ..common import get_classes
except ImportError:  # Direct import with SR/ on sys.path.
    from common import get_classes


class SpeechCommandsDataset(Dataset):
    """Load precomputed ``.npy`` features with a stable paper class map."""

    def __init__(self, root, num_classes=None, classes=None):
        self.root = Path(root).expanduser().resolve()
        if classes is None:
            classes = get_classes(num_classes or self._infer_num_classes())
        self.classes = tuple(classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if class_dir.is_dir():
                self.samples.extend((path, self.class_to_idx[class_name]) for path in sorted(class_dir.glob("*.npy")))
        if not self.samples:
            raise FileNotFoundError(f"No .npy Speech Commands features found under {self.root}")

    def _infer_num_classes(self):
        names = {path.name for path in self.root.iterdir() if path.is_dir()} if self.root.is_dir() else set()
        if names and names.issubset(set(get_classes(10))):
            return 10
        return 30

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        feature = np.load(path, allow_pickle=False)
        return torch.from_numpy(feature).float(), label
