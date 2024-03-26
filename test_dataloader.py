import os
import sys
from collections import defaultdict
from pathlib import Path
from random import shuffle

import numpy as np

from DLPrototypicalNetworks.dataloader import MiniImageNetDataset

# class MiniImageNetDataset:
#     def __init__(self, root, mode, n_shot, k_way, k_query, batchsz, resize, startidx):
#         """
#         Args:
#             root: str, root directory of the dataset
#             mode: str, train/val/test
#             n_shot: int, number of support examples per class
#             k_way: int, number of classes
#             k_query: int, number of query examples per class
#             batchsz: int, number of tasks per batch
#             resize: int, resize image to this size
#             startidx: int, start index of the dataset
#         """
#         self.root = root
#         self.mode = mode
#         self.n_shot = n_shot
#         self.k_way = k_way
#         self.k_query = k_query
#         self.batchsz = batchsz
#         self.resize = resize
#         self.startidx = startidx
#
#     def __len__(self):
#         return 100
#
#     def __getitem__(self, idx):
#         """
#         Args:
#             idx: int, index of the task
#         Returns:
#             support: torch.Tensor, support examples of the task
#             query: torch.Tensor, query examples of the task
#
#         """
#         return torch.randn(self.n_shot, 3, self.resize, self.resize), torch.randn(self.k_query, 3, self.resize, self.resize)

def test_mini_image_net_dataset():
    # training set
    dataset = MiniImageNetDataset(base_dir="data/mini-imagenet", mode="train", k_shot=1, k_way=5, k_query=5, n_episodes=100)
    # check whether one episode has 5 classes, and each class has (k_shot + k_query) examples + check number of episodes
    n_episodes = 0
    for datapoints in dataset:
        print(datapoints)
        assert len(datapoints) == 5
        for _, examples in datapoints.items():
            assert len(examples) == 6
        n_episodes += 1

    assert n_episodes == 100

    # now the test set
    dataset = MiniImageNetDataset(base_dir="data/mini-imagenet", mode="test", k_shot=1, k_way=5, k_query=5, n_episodes=100)
    # check whether one episode has 5 classes, and each class has (k_shot + k_query) examples + check number of episodes
    n_episodes = 0
    for datapoints in dataset:
        print(datapoints)
        assert len(datapoints) == 5
        for _, examples in datapoints.items():
            assert len(examples) == 6
        n_episodes += 1

    assert n_episodes == 100

    # now the validation set
    dataset = MiniImageNetDataset(base_dir="data/mini-imagenet", mode="val", k_shot=1, k_way=5, k_query=5, n_episodes=100)
    # check whether one episode has 5 classes, and each class has (k_shot + k_query) examples + check number of episodes
    n_episodes = 0
    for datapoints in dataset:
        print(datapoints)
        assert len(datapoints) == 5
        for _, examples in datapoints.items():
            assert len(examples) == 6
        n_episodes += 1

    assert n_episodes == 100



