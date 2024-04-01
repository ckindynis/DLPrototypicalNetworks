from enum import Enum


class Datasets(str, Enum):
    OMNIGLOT = "omniglot"
    MINIIMAGE = "miniimage"


class DistanceMetric(str, Enum):
    EUCLID = "euclid"
    COSINE = "cosine"


class Modes(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "validation"


class AssetNames(str, Enum):
    MODEL = "model.pth"