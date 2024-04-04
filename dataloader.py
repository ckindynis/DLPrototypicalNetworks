import random
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import as_completed
from pathlib import Path
from random import shuffle
from time import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import os
from PIL import ImageFile
import pandas as pd

# TODO make the class dependent on training/testing
# TODO prevent download/transform if data is already downloaded/transformed  -> DONE
# TODO think about how data is loaded with getitem
# TODO make the same script for the other datasets (the miniImageNet version of ILSVRC-2012) (2011 version of the Caltech UCSD bird dataset)
 

class OmnigotDataset(Dataset):
    def __init__(self, annotations_file=None, img_dir=None, transform=None, target_transform=None) -> None:
        self.img_dir = img_dir or os.path.join(f"{__file__}", "..", "data", "omniglot")
        # self.download_url_prefix = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python"
        self.alternative_url = "https://github.com/brendenlake/omniglot/tree/master/python"
        # What is this?
        zips_md5 = {   # what should these values be? 
        "images_background": "68d2efa1b9178cc56df9314c21c6e718",
        "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
        }

        # self.download()

        data_already_exists = os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data.pt")) and os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data.pt")) and os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data_labels.pt")) and os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data_labels.pt"))


        if data_already_exists:
            self.train_data_images = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data.pt"))
            self.test_data_images = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data.pt"))
            self.train_data_labels = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data_labels.pt"))
            self.test_data_labels = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data_labels.pt"))
        else:

            # load PIL images
            train_data = datasets.Omniglot(self.img_dir, download=True)
            test_data = datasets.Omniglot(self.img_dir, download=True, background=False)

            print(test_data[0])
            print(train_data[-1])

            # in directory "data/images_background"

            # resize images to 28x28
            train_data = [(img[0].resize((28, 28)), img[1]) for img in self.train_data]
            test_data = [(img[0].resize((28, 28)), img[1]) for img in self.test_data]

            # add 90 degree rotations of images to the dataset
            train_data += [(img[0].rotate(90), img[1]) for img in self.train_data]
            test_data += [(img[0].rotate(90), img[1]) for img in self.test_data]
            # add 180 degree rotations of images to the dataset
            train_data += [(img[0].rotate(180), img[1]) for img in self.train_data]
            test_data += [(img[0].rotate(180), img[1]) for img in self.test_data]
            # add 270 degree rotations of images to the dataset
            train_data += [(img[0].rotate(270), img[1]) for img in self.train_data]
            test_data += [(img[0].rotate(270), img[1]) for img in self.test_data]

            # convert images to tensors   -> (tensor, label) pairs
            train_data = [(transforms.PILToTensor()(img[0]), img[1]) for img in self.train_data]
            test_data = [(transforms.PILToTensor()(img[0]), img[1]) for img in self.test_data]
            # What stack does is it concatenates sequence of tensors along a new dimension.
            self.train_data_images = torch.stack([img[0] for img in self.train_data])
            self.test_data_images = torch.stack([img[0] for img in self.test_data])
            self.train_data_labels = [img[1] for img in self.train_data]
            self.test_data_labels = [img[1] for img in self.test_data]

            # save the transformed data to a file for faster loading 
            torch.save(self.train_data_images, os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data.pt"))
            torch.save(self.test_data_images, os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data.pt"))
            torch.save(self.train_data_labels, os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data_labels.pt"))
            torch.save(self.test_data_labels, os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data_labels.pt")) 
        

    def __len__(self):
        return len(self.training_dataset)
        
    def __getitem__(self, idx):
        return (self.train_data_images[idx], self.train_data_labels[idx])
        

    # def _check_integrity(self) -> bool:
    #     # filename = "images_background" if self.background else "images_evaluation"
    #     filename = "images_background"
    #     zip_filename = filename + ".zip"
    #     if not check_integrity(sys.path.join(self.img_dir, zip_filename), self.zips_md5[filename]):
    #         return False
    #     return True

    # def download(self):
    #     # download the dataset from the internet
    #     # check whether data is already downloaded or not and download if not
    #     # if self._check_integrity():
    #     #     print("Files already downloaded and verified")
    #     #     return
    #     filename = "images_background" if self.background else "images_evaluation"
    #     zip_filename = filename + ".zip"
    #     url = self.alternative_url + "/" + zip_filename
    #     download_and_extract_archive(url, self.img_dir, filename=zip_filename, md5=self.zips_md5[filename])


class DatasetBase(ABC, Dataset):
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
    #
    # @abstractmethod
    # def __iter__(self) -> "DatasetBase":
    #     pass
    #
    # @abstractmethod
    # def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     pass

class MiniImageNetDataset(DatasetBase, Dataset):
    """
    Same as MiniImageNetDataset but using PyTorch Dataset class
    """
    def __init__(self, base_dir: str | Path, k_way: int, k_shot: int, k_query: int, n_episodes: int,
                 mode: str = "train", transform: transforms.Compose = None,
                 target_transform: transforms.Compose = None) -> None:

        super().__init__(base_dir, k_way, k_shot, k_query, n_episodes, mode, transform or transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()]), target_transform)
        self.episode_count = 0

    def _load_data(self):
        start_time = time()
        data = datasets.ImageFolder(os.path.join(self.base_dir, self.mode))
        self.dataset = defaultdict(list)

        # Function to process each item
        def process_item(idx_label):
            idx, label = idx_label
            image, _ = data[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return label, image

        # Use ThreadPoolExecutor to parallelize the loading and transformation
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_item, idx_label) for idx_label in enumerate(data.targets)]
            for future in as_completed(futures):
                label, image = future.result()
                self.dataset[label].append(image)

        print(f"Time taken to load data: {time() - start_time:.2f} seconds for {self.mode} mode")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index >= self.n_episodes:
            raise IndexError("Index out of range")

        # Organize samples by class for the episode
        classes = np.random.choice(list(self.dataset.keys()), min(self.k_way, len(self.dataset)), replace=False)
        data, labels = [], []

        for cls_idx in classes:
            image_samples = random.sample(self.dataset[cls_idx], self.k_shot + self.k_query)
            data.extend(image_samples)
            labels.extend([cls_idx] * (self.k_shot + self.k_query))

        return torch.stack(data), torch.tensor(labels)
