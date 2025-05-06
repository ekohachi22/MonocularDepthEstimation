import glob
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

def load_data(base_path: str) -> dict:
    DEPTH_PREFIX = "depth"
    IMAGE_PREFIX = "image"
    ret = {
        "depth": [],
        "img": []
    }
    depth_file_paths = glob.glob(os.path.join(base_path, DEPTH_PREFIX, "*.npy"))
    image_file_paths = glob.glob(os.path.join(base_path, IMAGE_PREFIX, "*.npy"))

    # Ensure the files are loaded in the same order
    depth_file_paths.sort()
    image_file_paths.sort() 

    for d_path, i_path in zip(depth_file_paths, image_file_paths):
        assert os.path.split(d_path)[1] == os.path.split(i_path)[1]

        d_image = np.load(d_path)
        i_image = np.load(i_path)

        ret['depth'].append(d_image)
        ret['img'].append(i_image)
    
    return ret


class ImageDepthDataset(Dataset):
    def __init__(self, X_list, Y_list, input_only_transform=None, shared_transform=None):
        self.X_list = X_list
        self.Y_list = Y_list
        self.input_only_transform = input_only_transform
        self.shared_transform = shared_transform

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        image = self.X_list[idx].astype(np.uint8)
        depth = self.Y_list[idx].astype(np.float32)

        if self.input_only_transform:
            augmented = self.input_only_transform(image=image)
            image = augmented["image"]

        if self.shared_transform:
            augmented = self.shared_transform(image=image, depth=depth)
            image = augmented["image"]
            depth = augmented["depth"]

        return image, depth
