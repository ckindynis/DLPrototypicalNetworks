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
 

class OmnigotDataset(Dataset):
    def _init_(self, annotations_file=None, img_dir=None, transform=None, target_transform=None, train=True) -> None:
        self.img_dir = img_dir or os.path.join(f"{__file__}", "..", "data", "omniglot")
        self.training = train

        train_data_exists = os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data.pt")) and os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data_labels.pt"))
        test_data_exists = os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data.pt")) and os.path.exists(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data_labels.pt"))                                                                
        data_already_exists = train_data_exists if self.training else test_data_exists

        if data_already_exists:
            print("Loading data from file")
            if self.training:
                self.train_data_images = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data.pt")) 
                self.train_data_labels = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data_labels.pt"))
            else:
                self.test_data_images = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data.pt"))
                self.test_data_labels = torch.load(os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data_labels.pt"))
        else:
            print("Data not found, loading from Omniglot dataset")
            if self.training:
                self.train_data_images, self.train_data_labels = self.download_data()

                # save the transformed data to a file for faster loading 
                torch.save(self.train_data_images, os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data.pt"))  
                torch.save(self.train_data_labels, os.path.join(f"{__file__}", "..", "data", "omniglot", "train_data_labels.pt"))
            else:                 
                self.test_data_images, self.test_data_labels = self.download_data()

                # save the transformed data to a file for faster loading
                torch.save(self.test_data_images, os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data.pt"))                
                torch.save(self.test_data_labels, os.path.join(f"{__file__}", "..", "data", "omniglot", "test_data_labels.pt"))       
            print("Deleting the Omniglot dataset folder to save space")    
            shutil.rmtree(os.path.join(f"{__file__}", "..", "data", "omniglot", "omniglot-py"))
            os.rmdir(os.path.join(f"{__file__}", "..", "data", "omniglot", "omniglot-py"))
                
            

    def _len_(self):
        return len(self.train_data_images) if self.training else len(self.test_data_images)
        
    def _getitem_(self, idx):
        return (self.train_data_images[idx], self.train_data_labels[idx]) if self.training else (self.test_data_images[idx], self.test_data_labels[idx])

    def download_data(self):
        data = datasets.Omniglot(self.img_dir, download=True, background=self.training)

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



# class MiniImageNetDataset(Dataset):
#     def __init__(self, path : str | Path, train_val_test_class_frac: 'list[float]' = None, train_val_test_example_size: 'list[int]' = None, mode: str = "train") -> None:
#         if train_val_test_class_frac is None:
#             train_val_test_class_frac = [0.64, 0.16, 0.2]
#         if train_val_test_example_size is None:
#             train_val_test_example_size = [1, 1, 15]

#         assert sum(train_val_test_class_frac) == 1, "The sum of the fractions must be equal to 1"

#         self.mode = mode
#         self.class_frac = {"train": train_val_test_class_frac[0],
#                            "validation": train_val_test_class_frac[1],
#                            "test": train_val_test_class_frac[2]
#                            }
#         self.example_size = {"train": train_val_test_example_size[0],
#                              "validation": train_val_test_example_size[1],
#                              "test": train_val_test_example_size[2]
#                              }
#         self.path = path
#         self.transform = transforms.Compose([
#             transforms.Resize((84, 84)),
#             transforms.ToTensor()
#         ])
#         self.transform = None
#         self.target_transform = None

#         # load the data
#         mini_image_net_data = datasets.ImageFolder(root=path)

#         class_splits = self._split_classes(mini_image_net_data)
#         self.subsets = self._create_subsets(mini_image_net_data, class_splits)
        
#     def _split_classes(self, data: datasets.ImageFolder) -> dict[str, list[str]]:
#         # Split the data classes into training, validation and test sets
#         all_classes = data.classes
#         shuffle(all_classes)

#         # Split the classes into training, validation and test classes
#         train_classes = all_classes[:int(self.class_frac["train"] * len(all_classes))]
#         validation_classes = all_classes[int(self.class_frac["train"] * len(all_classes)):int((self.class_frac["train"] + self.class_frac["validation"]) * len(all_classes))]
#         test_classes = all_classes[int((self.class_frac["train"] + self.class_frac["validation"]) * len(all_classes)):]

#         return {"train": train_classes, "validation": validation_classes, "test": test_classes}


#     def _create_subsets(self, data: datasets.ImageFolder, class_splits: dict[str, list[str]]) -> dict[str, Subset]:
#         # From each class, sample the number of examples specified in train_val_test_example_size
#         subsets = {}
#         indices_by_class = defaultdict(list)
#         for idx, (_, label) in enumerate(data.samples):
#             indices_by_class[label].append(idx)

#         for subset, classes in class_splits.items():
#             selected_idx = []
#             for cls in classes:
#                 cls_idx = data.class_to_idx[cls]
#                 indices = indices_by_class[cls_idx]
#                 if len(indices) < self.example_size[subset]:
#                     selected_idx.extend(indices)
#                 else:
#                     selected_idx.extend(random.sample(indices, self.example_size[subset]))
#             subsets[subset] = Subset(data, selected_idx)

#         return subsets

#     def __len__(self):
#         return len(self.subsets[self.mode])

#     def __getitem__(self, index: int) -> Any:
#         data = self.subsets[self.mode]
#         img, label = data[index]

#         if self.transform:
#             img = self.transform(img)
#         if self.target_transform:
#             label = self.target_transform(label)

#         return img, label
    
#     def set_mode(self, mode: str) -> None:
#         self.mode = mode


if __name__ == "__main__":
    # test the Omniglot dataset
    omniglot = OmnigotDataset()
    print(omniglot[0])

    # # test the MiniImageNet dataset
    # mini_image_net = MiniImageNetDataset(path=os.path.join(f"{__file__}", "..", "data", "mini-imagenet"))