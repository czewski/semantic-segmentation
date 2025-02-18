import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

IMAGE_SIZE = 256 
transform_img = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])
transform_mask = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor()
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
            image = transform_img(image)
            mask = transform_mask(mask)

        # print(image.size())
        # print(mask.size())

        return image, mask  # (C, H, W) (C, H, W)
