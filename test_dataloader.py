from collections import defaultdict

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
    dataset = MiniImageNetDataset(path="data/mini-imagenet", mode="train")
    assert len(dataset) == 64
    # assert length of each subset is correct
    assert len(dataset.subsets["train"]) == 64
    assert len(dataset.subsets["validation"]) == 16
    assert len(dataset.subsets["test"]) == 20 * 15

    # assert that the number of examples in each subset is correct
    assert len(dataset.subsets["train"].indices) == 64
    assert len(dataset.subsets["validation"].indices) == 16
    assert len(dataset.subsets["test"].indices) == 20 * 15

    # get items from each mode and check that the number of items is correct for each subset class
    for mode in ["train", "validation", "test"]:
        dataset.set_mode(mode)
        mode_data = defaultdict(list)
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            mode_data[label].append(img)
        for key, value in mode_data.items():
            assert len(value) == dataset.example_size[mode]


