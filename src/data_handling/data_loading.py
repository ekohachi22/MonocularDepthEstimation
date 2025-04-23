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
    def __init__(self, X_list, Y_list, transform=None):
        assert len(X_list) == len(Y_list), "Input and target lists must be the same length"
        self.X_list = X_list  
        self.Y_list = Y_list
        self.transform = transform

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        image = self.X_list[idx].astype(np.uint8)   
        depth = self.Y_list[idx].astype(np.float32) 

        if depth.ndim == 2:
            depth = np.expand_dims(depth, axis=-1)  

        if self.transform:
            augmented = self.transform(image=image, depth=depth)
            image = augmented["image"]
            depth = augmented["depth"]

        return image, depth
