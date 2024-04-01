import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from random import shuffle
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import os
from PIL import ImageFile
import pandas as pd
import shutil

# TODO make the class dependent on training/testing
# TODO prevent download/transform if data is already downloaded/transformed  -> DONE
# TODO think about how data is loaded with getitem
# TODO make the same script for the other datasets (the miniImageNet version of ILSVRC-2012) (2011 version of the Caltech UCSD bird dataset)


class DatasetBase(ABC):
    def __init__(self, base_dir: str | Path, k_way: int, k_shot: int, k_query: int, n_episodes: int, mode: str = "train", transform: transforms.Compose = None, target_transform: transforms.Compose = None) -> None:
        """
        Args:
            base_dir: str | Path, root directory of the dataset
            k_way: int, number of classes
            k_shot: int, number of support examples per class
            k_query: int, number of query examples per class
            n_episodes: int, number of episodes (each episode selects {k_way} new set of classes)
            mode: str, train/validation/test
            transform: transforms.Compose, transform to apply to the images
            target_transform: transforms.Compose, transform to apply to the labels

        """
        self.k_way = k_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.transform = transform or transforms.Compose([])
        self.target_transform = target_transform or transforms.Compose([])
        self.mode = mode
        self.base_dir = Path(base_dir)
        self.n_episodes = n_episodes
        self._load_data()

    @abstractmethod
    def _load_data(self) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> "DatasetBase":
        pass

    @abstractmethod
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class OmniglotDataset(DatasetBase):
    def __init__(self, base_dir: str | Path, k_way: int, k_shot: int, k_query: int, n_episodes: int, mode: str = "train", transform: transforms.Compose = None, target_transform: transforms.Compose = None) -> None:
        super().__init__(base_dir, k_way, k_shot, k_query, n_episodes, mode, transform or transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()]), target_transform)
        self.episode_count = 0

    def _load_data(self):
        data_exists = os.path.exists(os.path.join(f"{self.base_dir}"))

        if not data_exists:
            data_images, data_labels = self.download_data()
            dataset = TensorDataset(data_images, data_labels)
            torch.save(dataset, os.path.join(f"{self.base_dir}"))

    # TODO Let's make this and mini-imagenet more alike
    def _load_data(self):
        train_data_exists = os.path.exists(os.path.join(f"{self.base_dir}", "train_data.pt"))
        val_data_exists = os.path.exists(os.path.join(f"{self.base_dir}", "validation_data.pt"))
        test_data_exists = os.path.exists(os.path.join(f"{self.base_dir}", "test_data.pt"))
        data_already_exists = train_data_exists if self.mode == "train" else val_data_exists if self.mode == "validation" else test_data_exists

        if not data_already_exists:
            if self.mode == "train":
                dataset = torch.load(os.path.join(f"{self.base_dir}", "train_data.pt"))
            elif self.mode == "validation":
                dataset = torch.load(os.path.join(f"{self.base_dir}", "validation_data.pt"))
            else:
                dataset = torch.load(os.path.join(f"{self.base_dir}", "test_data.pt"))
        else:
            data_images, data_labels = self.download_data()

            if self.mode == "train" or self.mode == "validation":
                # Split the training data into training and validation sets
                train_size = 4112
                val_size = 688
                dataset = TensorDataset(data_images, data_labels)
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

                # save the data
                torch.save(train_dataset, os.path.join(f"{self.base_dir}", "train_data.pt"))
                torch.save(val_dataset, os.path.join(f"{self.base_dir}", "validation_data.pt"))
                if self.mode == "train":
                    dataset = train_dataset
                else:
                    dataset = val_dataset
            else: # test
                dataset = TensorDataset(data_images, data_labels)
                # save the data
                torch.save(dataset, os.path.join(f"{self.base_dir}", "test_data.pt"))

            # Deleting the Omniglot dataset folder to save space
            shutil.rmtree(os.path.join(f"{self.base_dir}", "omniglot-py"))
            os.rmdir(os.path.join(f"{self.base_dir}", "omniglot-py"))

        self.data_images, self.data_labels = dataset[:]

    def download_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        datasets.Omniglot(str(self.base_dir), download=True, background=self.mode == "train" or self.mode == "validation")   # background is non-test data
        if os.path.exists(os.path.join(f"{self.base_dir}", "omniglot-py")):
            os.rename(os.path.join(f"{self.base_dir}", "omniglot-py"), os.path.join(f"{self.base_dir}", "omniglot"))

        # TODO
        # resize images to 28x28
        # data_0 = [(img[0].resize((28, 28)), img[1]) for img in omniglot]
        # TODO rotation during data iteration
        # # add 90 degree rotations of images to the dataset
        # data_90 = [(img[0].rotate(90), img[1]) for img in data_0]
        # # add 180 degree rotations of images to the dataset
        # data_180 = [(img[0].rotate(180), img[1]) for img in data_0]
        # # add 270 degree rotations of images to the dataset
        # data_270 = [(img[0].rotate(270), img[1]) for img in data_0]

        # data = data_0 + data_90 + data_180 + data_270

        # convert images to tensors -> (tensor, label) pairs
        # data = [(transforms.PILToTensor()(img[0]), img[1]) for img in data_0]

        # Stack the tensors along a new dimension
        # data_images = torch.stack([img[0] for img in data])
        # data_labels = torch.tensor([img[1] for img in data])

        # return data_images, data_labels

    def __iter__(self) -> "OmniglotDataset":
        self.episode_count = 0
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.episode_count == self.n_episodes:
            raise StopIteration

        # Implement the iteration functionality
        # we have the labels in the data_labels and the images in the data_images
        # we need to create the episodes
        for _ in range(self.n_episodes):
            # Select k_way classes randomly
            classes = np.random.choice(list(self.data_labels), self.k_way, replace=False)
            datapoints: dict[str, list] = defaultdict(list)
            for cls_idx in classes:
                selected_indices = np.random.choice(self.data_labels[cls_idx], self.k_shot + self.k_query, replace=False)
                for idx in selected_indices:
                    # transform
                    image_transformed = self.transform(self.data_images[idx])
                    class_transformed = self.target_transform(cls_idx)

                    datapoints[class_transformed].append(image_transformed)
            # convert the list of tensors to a tensor
            for cls in datapoints:
                datapoints[cls] = torch.stack(datapoints[cls])
            yield datapoints


class MiniImageNetDataset(DatasetBase):
    """
    internal order of the dataset is important for the labels, so training and testing can have the same labels
    """
    def __init__(self, base_dir: str | Path, k_way: int, k_shot: int, k_query: int, n_episodes: int, mode: str = "train", transform: transforms.Compose = None, target_transform: transforms.Compose = None) -> None:
        super().__init__(base_dir, k_way, k_shot, k_query, n_episodes, mode, transform or transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()]), target_transform)
        self._organize_samples()
        self.episode_count = 0

    def _load_data(self):
        self.dataset = datasets.ImageFolder(os.path.join(self.base_dir, self.mode))

    def _organize_samples(self):
        self.samples_per_class = defaultdict(list)
        for idx, (path, label) in enumerate(self.dataset.samples):
            self.samples_per_class[label].append(idx)

    def __iter__(self) -> "MiniImageNetDataset":
        self.episode_count = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.episode_count == self.n_episodes:
            raise StopIteration

        # Organize samples by class
        classes = np.random.choice(list(self.samples_per_class.keys()), min(self.k_way, len(self.samples_per_class)), replace=False)
        data = []
        labels = []

        for cls_idx in classes:
            selected_indices = np.random.choice(self.samples_per_class[cls_idx], min(self.k_shot + self.k_query, len(self.samples_per_class[cls_idx])), replace=False)
            for idx in selected_indices:
                # transform
                image_transformed = self.transform(self.dataset[idx][0])
                class_transformed = self.target_transform(cls_idx)

                data.append(image_transformed)
                labels.append(class_transformed)

        self.episode_count += 1

        return torch.stack(data), torch.tensor(labels)