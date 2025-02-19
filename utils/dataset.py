import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as t_functional
import torch
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 256 

transform_img = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])
transform_mask = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.NEAREST),  
    # this changes anything?
    # if not nearest, than weird behavior can happen with grayscale https://discuss.pytorch.org/t/weird-behaviour-when-mapping-masks/99798/8
    #transforms.ToTensor() # to tensor actually normalizes, so i lose direct class data
])

#background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7

## define custom dataset
class LoveDADataset(Dataset):
    def __init__(self, root_dir, split="train", transform=True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        # self.transform_img = transform_img
        # self.transform_mask = transform_mask

        self.image_dir = os.path.join(root_dir, split, "images_png")
        self.mask_dir = os.path.join(root_dir, split, "masks_png")

        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")  

        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        mask = Image.open(mask_path).convert("L") 

        if self.transform:
            image = transform_img(image).to(device)
            mask = transform_mask(mask)
            mask = t_functional.pil_to_tensor(mask).to(device)
            # print(mask.size())
            # print("hi")

        # print(image.size())
        # print(mask.size())

        return image, mask  # (C, H, W) (C, H, W)
