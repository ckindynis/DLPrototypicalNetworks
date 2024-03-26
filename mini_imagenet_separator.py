from pathlib import Path
from random import shuffle

if __name__ == '__main__':
    # separate the mini-imagenet dataset into 64 training and 20 testing and 16 validation classes
    base_dir = ...
    # iterate over subdirectories in the base directory
    subdirs = [subdir for subdir in Path(base_dir).iterdir() if subdir.is_dir()]
    shuffle(subdirs)
    # 64 training classes
    train_classes = subdirs[:64]
    # 20 testing classes
    test_classes = subdirs[64:84]
    # 16 validation classes
    val_classes = subdirs[84:]
    # create the directories
    for mode, classes in zip(["train", "test", "val"], [train_classes, test_classes, val_classes]):
        for cls in classes:
            Path(base_dir, mode, cls.name).mkdir(parents=True, exist_ok=True)
    # move the images to the appropriate directories
    for mode, classes in zip(["train", "test", "val"], [train_classes, test_classes, val_classes]):
        for cls in classes:
            for img in cls.iterdir():
                img.rename(Path(base_dir, mode, cls.name, img.name))
