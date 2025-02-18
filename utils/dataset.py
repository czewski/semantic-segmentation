import pickle
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torch

## load data
def load_data(root, valid_portion=0.2):
    return 

## define custom dataset
class DeepGlobe(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_filenames = [f for f in os.listdir(data_dir) if f.endswith('_sat.jpg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        mask_filename = image_filename.replace('_sat.jpg', '_mask.png')
        
        image = Image.open(os.path.join(self.data_dir, image_filename)).convert("RGB")
        mask = Image.open(os.path.join(self.data_dir, mask_filename)).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # convert mask to class indices (0-6)
        mask = np.array(mask)
        mask = self.rgb_to_class_indices(mask)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
    
    def rgb_to_class_indices(self, mask):
        color_to_class = {
            # r g b
            (255, 255, 0): 0,  # agriculture_land
            (255, 255, 255): 1, # barren_land
            (0, 255, 0): 2,    # forest_land
            (255, 0, 255): 3,  # rangeland
            (0, 0, 0): 4,      # unknown
            (0, 255, 255): 5,  # urban_land
            (0, 0, 255): 6,    # water
        }
        class_indices = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for rgb, class_idx in color_to_class.items():
            class_indices[(mask == rgb).all(axis=-1)] = class_idx
        return class_indices
    
