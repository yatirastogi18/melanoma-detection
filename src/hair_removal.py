import cv2
import numpy as np
from pathlib import Path
# Load image

image_path = r"PH2Dataset\PH2 Dataset images\IMD002\IMD002_Dermoscopic_Image\IMD002.bmp"

image = cv2.imread(image_path)

# Convert to grayscale

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blackhat operation to detect hairs

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (17, 17)
)

blackhat = cv2.morphologyEx(
    gray,
    cv2.MORPH_BLACKHAT,
    kernel
)

# Threshold to create hair mask

_, hair_mask = cv2.threshold(
    blackhat,
    10,
    255,
    cv2.THRESH_BINARY
)

# Remove hairs using inpainting

hair_removed = cv2.inpaint(
    image,
    hair_mask,
    1,
    cv2.INPAINT_TELEA
)

# Save outputs

cv2.imwrite("original.jpg", image)
cv2.imwrite("hair_mask.jpg", hair_mask)
cv2.imwrite("hair_removed.jpg", hair_removed)

print("Hair removal completed")