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
    def __init__(self, X_list, Y_list, transform_x: callable=None, transform_y: callable=None):
        assert len(X_list) == len(Y_list), "Input and target lists must be the same length"
        self.X_list = torch.Tensor(np.transpose(X_list, (0, 3, 1, 2)))
        self.Y_list = torch.Tensor(np.transpose(Y_list, (0, 3, 1, 2)))
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = self.X_list[idx]
        y = self.Y_list[idx]

        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)

        return x, y
