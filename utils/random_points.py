import torch
import matplotlib.pyplot as plt

# randomize a percentage of image as points
def create_random_points(mask, percentage, size):
    torch.manual_seed(522)
    B, C, H, W = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    total_size = H * W 
    num_points = int(total_size * percentage)
    # print(num_points)
    # print(mask.size())
    # print(random_mask.size())

    ## second attempt: generate random bitwise (faster)
    rand_mask = torch.rand(B, H, W, device=mask.device)  # (B, H, W)
    topk_indices = torch.topk(rand_mask.flatten(1), num_points, dim=1).indices  # (B, num_points)
    binary_mask = torch.zeros(B, H * W, device=mask.device)  # (B, H * W)
    binary_mask.scatter_(1, topk_indices, 1.0)  # (B, H * W)
    binary_mask = binary_mask.view(B, C, H, W)  # (B, C, H, W)
    
    # this operation reduces 1 from main classes 
    # transform 0 to 255
    # and applies binary mask 
    result = (mask - 1) * binary_mask + 255 * (1 - binary_mask)
    return result

    # first attempt: 
    #random_mask = torch.full_like(mask, 255)
    # for item_batch in range(mask.shape[0]):
    #     for point in range(num_points):
    #         random_x, random_y = torch.randint(0, mask.shape[2]), torch.randint(0, mask.shape[3])
    #         random_mask[item_batch, ..., random_x, random_y] = mask[item_batch, ..., random_x, random_y] - 1
    #         # everything that is 0 (class = other), will also be -1 (transforms to 255 and is ignored)
    #         # print(random_mask[item_batch, ..., random_x, random_y])
    #         # print(mask[item_batch, ..., random_x, random_y])
    # return random_mask

def plot_mask(mask, batch_num):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask[batch_num].squeeze().cpu().numpy(), cmap="gray")  # ignore batch, just for testing
    plt.title("random mask")
    plt.axis("off")  
    plt.show()
    return
