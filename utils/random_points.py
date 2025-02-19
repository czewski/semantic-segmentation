import torch
from numpy import random
import matplotlib.pyplot as plt

# randomize a percentage of image as points
def create_random_points(mask, percentage, size):
    # TODO: implement size (kernel_size)
    total_size = mask.shape[2]*mask.shape[3] # B, C, H, W
    num_points = int(total_size * percentage)
    # print(num_points)
    # print('\n')
    random_mask = torch.full_like(mask, 255)
    # print(mask.size())
    # print(random_mask.size())
    for item_batch in range(mask.shape[0]):
        for point in range(num_points):
            random_x, random_y = random.randint(mask.shape[-2]), random.randint(mask.shape[-1])
            random_mask[item_batch, ..., random_x, random_y] = mask[item_batch, ..., random_x, random_y] - 1
            # everything that is 0 (class = other), will also be -1 (transforms to 255 and is ignored)
            # print(random_mask[item_batch, ..., random_x, random_y])
            # print(mask[item_batch, ..., random_x, random_y])

    return random_mask

def plot_mask(mask, batch_num):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask[batch_num].squeeze().cpu().numpy(), cmap="gray")  # ignore batch, just for testing
    plt.title("random mask")
    plt.axis("off")  
    plt.show()
    return
