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

# ==================================
# DEVICE
# ==================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ==================================
# DATASETS
# ==================================

train_dataset = MelanomaDataset(
    "train.csv",
    "processed/ham10000",
    train_transform
)

val_dataset = MelanomaDataset(
    "val.csv",
    "processed/ham10000",
    val_test_transform
)

test_dataset = MelanomaDataset(
    "test.csv",
    "processed/ham10000",
    val_test_transform
)

print("Train Size:", len(train_dataset))
print("Val Size:", len(val_dataset))
print("Test Size:", len(test_dataset))

# ==================================
# DATALOADERS
# ==================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("All DataLoaders Created")

# ==================================
# MODEL
# ==================================

model = models.efficientnet_v2_s(
    weights="DEFAULT"
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    1
)

model = model.to(device)

print("EfficientNetV2 Loaded")

# ==================================
# LOSS + OPTIMIZER
# ==================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ==================================
# TRAINING
# ==================================

epochs = 1

print("Training Started")

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        labels = (
            labels
            .float()
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
                f"Epoch {epoch+1} | Batch {batch_idx}"
            )

    print(
        f"Epoch {epoch+1} Loss = {running_loss:.4f}"
    )

# ==================================
# SAVE MODEL
# ==================================

torch.save(
    model.state_dict(),
    "efficientnetv2_ham10000.pth"
)

print("Model Saved")

# ==================================
# EVALUATION
# ==================================

model.eval()

y_true = []
y_pred = []
y_prob = []

print("Evaluation Started")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (
            probs > 0.5
        ).int()

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
print("EfficientNetV2 Results")
print("==============================")
print("Accuracy    :", round(accuracy,4))
print("Precision   :", round(precision,4))
print("Sensitivity :", round(sensitivity,4))
print("F1 Score    :", round(f1,4))
print("ROC AUC     :", round(auc,4))
print("==============================")