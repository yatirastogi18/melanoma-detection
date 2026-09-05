import cv2
import numpy as np
from pathlib import Path


INPUT_IMAGES = Path("processed/ph2")
INPUT_MASKS = Path("segmentation/watershed/ph2")

OUTPUT_DIR = Path("final_dataset/ph2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


for image_path in INPUT_IMAGES.glob("*.jpg"):

    mask_path = INPUT_MASKS / f"{image_path.stem}_mask.png"

    if not mask_path.exists():
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Find lesion contour
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        continue

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    # 10% padding
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)

    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    roi = image[y1:y2, x1:x2]

    # Resize
    roi = cv2.resize(
        roi,
        (224, 224)
    )

    # Pixel normalization
    roi = roi.astype(np.float32) / 255.0

    # Save back as image
    roi = (roi * 255).astype(np.uint8)

    cv2.imwrite(
        str(OUTPUT_DIR / image_path.name),
        roi
    )

print("Final Dataset Ready")