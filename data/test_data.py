import numpy as np
import cv2
from PIL import Image


### TEST 1

# mask = cv2.imread("data/masks_png/3514.png", cv2.IMREAD_GRAYSCALE)  # Load as grayscale
# unique_classes = np.unique(mask)
# print("Classes in mask:", unique_classes)
#background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7


### TEST 2

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# image_path = "data/train/images_png/3514.png"
# mask_path  = "data/train/masks_png/3514.png"

# # Load raw
# image = Image.open(image_path).convert("RGB")
# mask  = Image.open(mask_path).convert("L")

# # Check shapes and unique values
# print("Image size:", image.size)
# print("Mask size:", mask.size)

# mask_np = np.array(mask)
# print("Unique mask values:", np.unique(mask_np))

# # Show them side by side
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(image)
# plt.title("Raw Image")

# plt.subplot(1,2,2)
# plt.imshow(mask_np, cmap="jet")
# plt.title("Raw Mask")
# plt.show()

# print("hi")



### TEST 3 
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import dataset
import torch

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

# actually using the validation split for initial shape tests
train_dataset = dataset.LoveDADataset(root_dir="data", split="train", transform=transform_img, transform_mask=transform_mask)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

def visualize_sample(dataset, index=0):
    image, mask = dataset[index] 
    image = np.array(image.permute(1,2,0))  # swap shape (H, W, C)
    mask = np.array(mask.permute(1,2,0))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Mask")
    plt.show()

visualize_sample(train_dataset, index=10)
print("hi")