import cv2
import numpy as np
from pathlib import Path


def watershed_segmentation(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=3
    )

    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    _, sure_fg = cv2.threshold(
        dist_transform,
        0.5 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    markers = cv2.watershed(
        image,
        markers
    )

    # Create binary mask
    mask = np.zeros(gray.shape, dtype=np.uint8)

    mask[markers > 1] = 255

    return mask


def process_dataset(input_dir, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    for image_path in input_dir.glob("*.jpg"):

        mask = watershed_segmentation(
            image_path
        )

        if mask is None:
            continue

        output_path = (
            output_dir /
            f"{image_path.stem}_mask.png"
        )

        cv2.imwrite(
            str(output_path),
            mask
        )

        count += 1

    print(f"Masks Generated: {count}")


if __name__ == "__main__":

    process_dataset(
        Path("processed/ph2"),
        Path("segmentation/watershed/ph2")
    )