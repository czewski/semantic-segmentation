import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

data_dir = ""
# train_image_dir = os.path.join(data_dir, "test", "images_png")
# train_mask_dir = os.path.join(data_dir, "train", "masks_png")
# valid_image_dir = os.path.join(data_dir, "valid", "images_png")
valid_mask_dir = os.path.join(data_dir, "valid", "masks_png")

def count_files(directory):
    return len([f for f in os.listdir(directory) if f.endswith('.png')])

def get_class_distribution(mask_dir):
    class_counts = defaultdict(int)
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):
            mask = np.array(Image.open(os.path.join(mask_dir, mask_file)))
            unique_classes, counts = np.unique(mask, return_counts=True)
            for cls, cnt in zip(unique_classes, counts):
                class_counts[cls] += cnt
    return class_counts

def plot_class_distribution(class_counts, title):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=counts)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Pixel Count")
    plt.show()

# train_image_count = count_files(train_image_dir)
# train_mask_count = count_files(train_mask_dir)
# valid_image_count = count_files(valid_image_dir)
# valid_mask_count = count_files(valid_mask_dir)
# print(train_image_count)
# print(train_mask_count)
# print(valid_image_count)
# print(valid_mask_count)

# train_class_dist = get_class_distribution(train_mask_dir)
valid_class_dist = get_class_distribution(valid_mask_dir)
# plot_class_distribution(train_class_dist, "Train Set Class Distribution")
plot_class_distribution(valid_class_dist, "Valid Set Class Distribution")
