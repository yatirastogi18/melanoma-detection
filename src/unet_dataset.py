from pathlib import Path

import cv2
import torch

from torch.utils.data import Dataset
from torchvision import transforms


class UNetDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(
            list(
                self.image_dir.glob("*.jpg")
            )
        )

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        mask_path = (
            self.mask_dir /
            f"{image_path.stem}_mask.png"
        )

        image = cv2.imread(
            str(image_path)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(
            image,
            (224, 224)
        )

        mask = cv2.resize(
            mask,
            (224, 224)
        )

        image = self.transform(image)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        mask = mask / 255.0

        return image, mask


if __name__ == "__main__":

    dataset = UNetDataset(
        "processed/ph2",
        "segmentation/watershed/ph2"
    )

    print(
        "Total Samples:",
        len(dataset)
    )

    image, mask = dataset[0]

    print(
        "Image Shape:",
        image.shape
    )

    print(
        "Mask Shape:",
        mask.shape
    )