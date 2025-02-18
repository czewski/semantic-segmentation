import torch
from numpy import random
import matplotlib.pyplot as plt

# randomize a percentage of image as points
def create_random_points(mask, percentage, size):
    total_size = mask.shape[2]*mask.shape[3] # C, H, W
    num_points = int(total_size * percentage)
    random_mask = torch.zeros(mask.size())
    # print(mask.size())
    # print(random_mask.size())

    for point in range(num_points):
        random_x, random_y = random.randint(mask.shape[-2]), random.randint(mask.shape[-1])
        random_mask[..., random_x, random_y] = mask[..., random_x, random_y]

    return random_mask

def plot_mask(mask):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask[0].squeeze().cpu().numpy(), cmap="gray")  # ignore batch, just for testing
    plt.title("random mask")
    plt.axis("off")  
    plt.show()
    return
