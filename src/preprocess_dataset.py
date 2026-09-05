import cv2
import numpy as np
from pathlib import Path

# -----------------------------
# Hair Removal
# -----------------------------
def remove_hair(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    result = cv2.inpaint(
        image,
        hair_mask,
        1,
        cv2.INPAINT_TELEA
    )

    return result


# -----------------------------
# CLAHE
# -----------------------------
def apply_clahe(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )


# -----------------------------
# Color Normalization
# -----------------------------
def normalize_image(image):

    return cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )


# -----------------------------
# Full Pipeline
# -----------------------------
def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image = remove_hair(image)

    image = apply_clahe(image)

    image = normalize_image(image)

    return image


# -----------------------------
# Process Folder
# -----------------------------
def process_dataset(input_dir, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    ]

    count = 0

    for image_path in input_dir.rglob("*"):

        if image_path.suffix.lower() not in image_extensions:
            continue

        processed = preprocess_image(image_path)

        if processed is None:
            continue

        output_path = (
            output_dir /
            f"{image_path.stem}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            processed
        )

        count += 1

    print(f"Processed {count} images")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    process_dataset(
        Path("PH2Dataset/PH2 Dataset images"),
        Path("processed/ph2")
    )
    process_dataset(
    Path("ISIC2018_Task3_Training_Input"),
    Path("processed/isic2018")
    )

    process_dataset(
    Path("dataverse_files/HAM10000_images_combined_600x450"),
    Path("processed/ham10000")
    )