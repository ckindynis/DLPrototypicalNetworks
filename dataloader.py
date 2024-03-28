from abc import ABC, abstractmethod
import random
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
# import matplotlib.pyplot as plt
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

class OmniglotDataset(DatasetBase):
        
    def _load_data(self):    
        print("Initializing Omniglot dataset")
        self.img_dir = self.base_dir

        train_data_exists = os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data.pt")) and os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data_labels.pt"))
        val_data_exists = os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "val_data.pt")) and os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "val_data_labels.pt"))
        test_data_exists = os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data.pt")) and os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data_labels.pt"))                                                                
        data_already_exists = train_data_exists if self.mode == "train" else val_data_exists if self.mode == "validation" else test_data_exists

        if data_already_exists:
            print("Loading data from file")
            if self.mode == "train":
                self.data_images = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data.pt")) 
                self.data_labels = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data_labels.pt"))
            elif self.mode == "validation":
                self.data_images = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "val_data.pt"))
                self.data_labels = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "val_data_labels.pt"))
            else:
                self.data_images = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data.pt"))
                self.data_labels = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data_labels.pt"))
        else:
            print("Data not found, loading from Omniglot dataset")
            if self.mode == "train" or self.mode == "validation":
                data_images, data_labels = self.download_data()

                # Split the training data into training and validation sets
                train_size = int(0.8 * len(self.data_images))
                val_size = len(data_images) - train_size
                train_data_images, val_data_images = random_split(data_images, [train_size, val_size])
                train_data_labels, val_data_labels = random_split(data_labels, [train_size, val_size])                

                # save the transformed data to a file for faster loading 
                torch.save(train_data_images, os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data.pt"))  
                torch.save(train_data_labels, os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data_labels.pt"))
                torch.save(val_data_images, os.path.join(f"{__file__}", "..", "data", "omniglot", "val_data.pt"))
                torch.save(val_data_labels, os.path.join(f"{__file__}", "..", "data", "omniglot", "val_data_labels.pt"))
                if self.mode == "train":
                    data_images = train_data_images
                    data_labels = train_data_labels
                else:
                    data_images = val_data_images
                    data_labels = val_data_labels
            else: # test                
                self.data_images, self.data_labels = self.download_data()

                # save the transformed data to a file for faster loading
                torch.save(self.data_images, os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data.pt"))                
                torch.save(self.data_labels, os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data_labels.pt"))       
            print("Deleting the Omniglot dataset folder to save space")    
            shutil.rmtree(os.path.join(f"{__file__}", "..", "data", "omniglot", "omniglot-py"))
            os.rmdir(os.path.join(f"{__file__}", "..", "data", "omniglot", "omniglot-py")) 
        
    def download_data(self):
        data = datasets.Omniglot(self.img_dir, download=True, background=self.mode == "train" or self.mode == "validation")

        # resize images to 28x28
        data = [(img[0].resize((28, 28)), img[1]) for img in data]

        # add 90 degree rotations of images to the dataset
        data += [(img[0].rotate(90), img[1]) for img in data]
        # add 180 degree rotations of images to the dataset
        data += [(img[0].rotate(180), img[1]) for img in data]
        # add 270 degree rotations of images to the dataset
        data += [(img[0].rotate(270), img[1]) for img in data]

        # convert images to tensors   -> (tensor, label) pairs
        data = [(transforms.PILToTensor()(img[0]), img[1]) for img in data]

        # Stack the tensors along a new dimension
        data_images = torch.stack([img[0] for img in data])
        data_labels = [img[1] for img in data]

        return data_images, data_labels
    ''''
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
    '''
    def __iter__(self) -> dict[str, torch.Tensor]: # type: ignore
        print("Data: ", self.data_labels)    #  <torch.utils.data.dataset.Subset object at 0x000001E6FCA31288>
        print("Type: ", type(self.data_labels)) #  <class 'torch.utils.data.dataset.Subset'>
        print("Data of images: ", self.data_images)
        print("Type of images: ", type(self.data_images))   # <class 'torch.utils.data.dataset.Subset'>
        # Access the first element of the Subset object
        print("First element of data: ", self.data_labels[0])  # 0

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


if __name__ == "__main__":
    # test the Omniglot dataset
    print("Testing the Omniglot dataset")
    omniglot_train = OmniglotDataset(mode="train", base_dir=os.path.join(f"{__file__}", "..", "data", "omniglot"), k_way=5, k_shot=1, k_query=1, n_episodes=10)
    omniglot_val = OmniglotDataset(mode="validation", base_dir=os.path.join(f"{__file__}", "..", "data", "omniglot"), k_way=5, k_shot=1, k_query=1, n_episodes=10)
    omniglot_test = OmniglotDataset(mode="test", base_dir=os.path.join(f"{__file__}", "..", "data", "omniglot"), k_way=5, k_shot=1, k_query=1, n_episodes=10)
    # check if iterating over the dataset works
    for episode in omniglot_train.__iter__():
        print(episode)

    # # test the MiniImageNet dataset
    # mini_image_net = MiniImageNetDataset(path=os.path.join(f"{__file__}", "..", "data", "mini-imagenet"))