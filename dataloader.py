import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from random import shuffle
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


class DatasetBase(ABC):
    def __init__(self, base_dir: str | Path, k_way: int, k_shot: int, k_query: int, n_episodes: int, mode: str = "train", transform: transforms.Compose = None, target_transform: transforms.Compose = None) -> None:
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
    def _load_data(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

class MiniImageNetDataset(DatasetBase):
    def __init__(self, base_dir: str | Path, k_way: int, k_shot: int, k_query: int, n_episodes: int, mode: str = "train", transform: transforms.Compose = None, target_transform: transforms.Compose = None) -> None:
        super().__init__(base_dir, k_way, k_shot, k_query, n_episodes, mode, transform or transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()]), target_transform)

    def _load_data(self):
        self.dataset = datasets.ImageFolder(os.path.join(self.base_dir, self.mode))

    def __iter__(self) -> dict[str, torch.Tensor]:
        # Organize samples by class
        self.samples_per_class = defaultdict(list)
        for idx, (path, label) in enumerate(self.dataset.samples):
            self.samples_per_class[label].append(idx)
        for _ in range(self.n_episodes):
            classes = np.random.choice(list(self.samples_per_class.keys()), self.k_way, replace=False)
            datapoints: dict[str, list] = defaultdict(list)
            for cls_idx in classes:
                selected_indices = np.random.choice(self.samples_per_class[cls_idx], self.k_shot + self.k_query, replace=False)
                for idx in selected_indices:
                    # transform
                    image_transformed = self.transform(self.dataset[idx][0])
                    class_transformed = self.target_transform(cls_idx)

                    datapoints[class_transformed].append(image_transformed)
            # convert the list of tensors to a tensor
            for cls in datapoints:
                datapoints[cls] = torch.stack(datapoints[cls])
            yield datapoints




                



