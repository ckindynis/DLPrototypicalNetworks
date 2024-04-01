from constants import Datasets
from dataclasses import dataclass


@dataclass
class DatasetConfiguration:
    learning_rate: float
    lr_decay_step: int
    lr_decay_gamma: float
    num_episodes_train: int
    num_episodes_test: int
    num_classes_train: int
    num_support_train: int
    num_query_train: int
    num_classes_val: int
    num_support_val: int
    num_query_val: int


configuration = {
    Datasets.MINIIMAGE: DatasetConfiguration(
        learning_rate=1e-3,
        lr_decay_step=2000,
        lr_decay_gamma=0.5,
        num_episodes_train=100,
        num_episodes_test=1000,
        num_classes_train=20,
        num_support_train=5,
        num_query_train=15,
        num_classes_val=5,
        num_support_val=5,
        num_query_val=15,
    ),
    Datasets.OMNIGLOT: DatasetConfiguration(
        learning_rate=1e-3,
        lr_decay_step=2000,
        lr_decay_gamma=0.5,
        num_episodes_train=100,
        num_episodes_test=1000,
        num_classes_train=60,
        num_support_train=5,
        num_query_train=5,
        num_classes_val=5,
        num_support_val=5,
        num_query_val=15,
    )
}
