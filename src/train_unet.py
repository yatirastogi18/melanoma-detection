import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from unet_model import UNet
from unet_dataset import UNetDataset


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

dataset = UNetDataset(
    "processed/ph2",
    "segmentation/watershed/ph2"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = UNet().to(device)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):

    running_loss = 0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {running_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "unet_ph2.pth"
)

print("Training Completed")