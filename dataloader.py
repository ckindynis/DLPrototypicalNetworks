import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import os
from PIL import Image

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


if __name__ == "__main__":
    test = OmnigotDataset()