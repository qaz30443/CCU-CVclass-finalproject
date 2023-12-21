"""" Modified version of https://github.com/jeffwen/road_building_extraction/blob/master/src/utils/data_utils.py """
from __future__ import print_function, division
from typing import Any
from torch.utils.data import Dataset
from skimage import io
import glob
import os
import torch
from torchvision import transforms
import numpy as np

# train_path =  "../training/mask*.png"
# valid_path =  "../testing/mask*.png"

train_path =  "../CV_class/training_Canny/mask*.png"
valid_path =  "../CV_class/testing_Canny/mask*.png"


class ImageDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads', 'mass_roads_crop', or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        self.path = train_path if train else valid_path
        self.mask_list = glob.glob(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        maskpath = self.mask_list[idx]
        image = io.imread(maskpath.replace("mask", "retina"))
        mask = io.imread(maskpath, as_gray=True)
        
        sample = image
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)
        
        if self.transform:
            sample = self.transform(sample)
            mask = self.transform(mask)

        return {"sat_img" : sample, "map_img" : mask}

class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        sat_img, map_img = sample["sat_img"], sample["map_img"]

        # Convert ndarray to PIL Image and then back to ndarray
        sat_img = transforms.functional.to_pil_image(sat_img)
        sat_img = transforms.functional.resize(sat_img, self.output_size)
        sat_img = np.array(sat_img)

        map_img = transforms.functional.to_pil_image(map_img)
        map_img = transforms.functional.resize(map_img, self.output_size)
        map_img = np.array(map_img)  # Convert back to NumPy ndarray
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {
            # "sat_img": transforms.functional.to_tensor(sat_img),
            "sat_img": sat_img,
            "map_img": torch.from_numpy(map_img).unsqueeze(0).float().div(255),
        }  # unsqueeze for the channel dimension


class NormalizeTarget(transforms.Normalize):
    """Normalize a tensor and also return the target"""

    def __call__(self, sample):
        return {
            "sat_img": transforms.functional.normalize(
                sample["sat_img"], self.mean, self.std
            ),
            "map_img": sample["map_img"],
        }


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
