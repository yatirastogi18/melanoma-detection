import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from classification_dataset import MelanomaDataset
from train_transforms import train_transform, val_test_transform

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

train_dataset = MelanomaDataset(
    "train.csv",
    "processed/isic2018",
    train_transform
)

val_dataset = MelanomaDataset(
    "val.csv",
    "processed/isic2018",
    val_test_transform
)

test_dataset = MelanomaDataset(
    "test.csv",
    "processed/isic2018",
    val_test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

model = models.swin_t(
    weights="DEFAULT"
)

model.head = nn.Linear(
    model.head.in_features,
    1
)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 3

print("Training Started")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        labels = (
            labels.float()
            .unsqueeze(1)
            .to(device)
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}"
            )

    print(
        f"Epoch {epoch+1} Loss: {running_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "swin_isic2018.pth"
)

print("Model Saved")

model.eval()

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).int()

        y_true.extend(labels.numpy())

        y_pred.extend(
            preds.cpu()
            .numpy()
            .flatten()
        )

        y_prob.extend(
            probs.cpu()
            .numpy()
            .flatten()
        )

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

sensitivity = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_prob
)

print("\n==============================")
print("ISIC2018 Swin Results")
print("==============================")
print("Accuracy    :", round(accuracy,4))
print("Precision   :", round(precision,4))
print("Sensitivity :", round(sensitivity,4))
print("F1 Score    :", round(f1,4))
print("ROC AUC     :", round(auc,4))
print("==============================")