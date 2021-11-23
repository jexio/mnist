import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import urllib

import torch
from torchvision.datasets import MNIST
from torchvision import transforms as transform_lib
from torch.utils.data import DataLoader, Dataset, random_split
from fulmo.core import BaseDataModule, BaseDataModuleParameters
from fulmo.settings import Stage


class _MNIST(Dataset):
    """Carbon copy of ``tests.helpers.datasets.MNIST``.
    We cannot import the tests as they are not distributed with the package.
    See https://github.com/PyTorchLightning/pytorch-lightning/pull/7614#discussion_r671183652 for more context.
    """

    RESOURCES = (
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/training.pt",
        "https://pl-public-data.s3.amazonaws.com/MNIST/processed/test.pt",
    )

    TRAIN_FILE_NAME = "training.pt"
    TEST_FILE_NAME = "test.pt"
    cache_folder_name = "complete"

    def __init__(
        self, root: str, train: bool = True, normalize: tuple = (0.1307, 0.3081), download: bool = True, **kwargs
    ):
        super().__init__()
        self.root = root
        self.train = train  # training set or test set
        self.normalize = normalize
        self.prepare_data(download)

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = self._try_load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx].float().unsqueeze(0)
        target = int(self.targets[idx])

        if self.normalize is not None and len(self.normalize) == 2:
            img = self.normalize_tensor(img, *self.normalize)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def cached_folder_path(self) -> str:
        return os.path.join(self.root, "MNIST", self.cache_folder_name)

    def _check_exists(self, data_folder: str) -> bool:
        existing = True
        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            existing = existing and os.path.isfile(os.path.join(data_folder, fname))
        return existing

    def prepare_data(self, download: bool = True):
        if download and not self._check_exists(self.cached_folder_path):
            self._download(self.cached_folder_path)
        if not self._check_exists(self.cached_folder_path):
            raise RuntimeError("Dataset not found.")

    def _download(self, data_folder: str) -> None:
        os.makedirs(data_folder, exist_ok=True)
        for url in self.RESOURCES:
            logging.info(f"Downloading {url}")
            fpath = os.path.join(data_folder, os.path.basename(url))
            urllib.request.urlretrieve(url, fpath)

    @staticmethod
    def _try_load(path_data, trials: int = 30, delta: float = 1.0):
        """Resolving loading from the same time from multiple concurrent processes."""
        res, exception = None, None
        assert trials, "at least some trial has to be set"
        assert os.path.isfile(path_data), f"missing file: {path_data}"
        for _ in range(trials):
            try:
                res = torch.load(path_data)
            # todo: specify the possible exception
            except Exception as e:
                exception = e
                time.sleep(delta * random.random())
            else:
                break
        if exception is not None:
            # raise the caught exception
            raise exception
        return res

    @staticmethod
    def normalize_tensor(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
        return tensor.sub(mean).div(std)


class MnistDataModule(BaseDataModule):
    """Standard MNIST, train, val, test splits and transforms.

    Attributes:
        data_dir: where to save/load the data
        val_split: how many of the training images to use for the validation split
        seed: starting seed for RNG.
    """

    def __init__(
        self,
        data_dir: str,
        parameters: Dict[str, BaseDataModuleParameters],
        val_split: int = 5000,
        seed: int = 42,
    ) -> None:
        """Create a new instance of MnistDataModule."""
        super().__init__(data_dir, parameters)
        self.data_dir = data_dir
        self.val_split = val_split
        self.normalize = True
        self.seed = seed

    def prepare_data(self):
        """Saves MNIST files to `data_dir`"""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset."""
        extra = dict(transform=self.default_transforms) if self.default_transforms else {}
        dataset = MNIST(self.data_dir, train=True, download=False, **extra)
        train_length = len(dataset)
        self.data_train, self.data_val = random_split(dataset, [train_length - self.val_split, self.val_split])
        self.data_test = MNIST(self.data_dir, train=False, download=False, **extra)

    @property
    def collate_fn(self) -> Optional[Callable[[Dict[str, Any]], Dict[str, torch.Tensor]]]:
        """Get collate_fn."""
        def collate(data):

            data, labels = zip(*data)
            data = torch.cat(data).unsqueeze(1)
            labels = torch.tensor(labels)
            return {"features": data, "target": labels}

        return collate

    @property
    def default_transforms(self):
        if self.normalize:
            mnist_transforms = transform_lib.Compose(
                [transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5,), std=(0.5,))]
            )
        else:
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms


__all__ = ["MnistDataModule"]
