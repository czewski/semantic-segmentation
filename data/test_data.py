import numpy as np
import cv2

mask = cv2.imread("data/masks_png/3514.png", cv2.IMREAD_GRAYSCALE)  # Load as grayscale
unique_classes = np.unique(mask)
print("Classes in mask:", unique_classes)

#background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7