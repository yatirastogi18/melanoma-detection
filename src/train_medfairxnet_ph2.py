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
from train_transforms import (
    train_transform,
    val_test_transform
)

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

# DATASETS
train_dataset = MelanomaDataset(
    "ph2_train_final.csv",
    "",
    train_transform
)

test_dataset = MelanomaDataset(
    "ph2_test_final.csv",
    "",
    val_test_transform
)

print("Train Size:", len(train_dataset))
print("Test Size :", len(test_dataset))

# DATALOADER
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

# BACKBONE
backbone = models.efficientnet_v2_s(
    weights="DEFAULT"
)

num_features = backbone.classifier[1].in_features

backbone.classifier = nn.Identity()

# PROPOSED MODEL
class MedFairXNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.backbone(x)

        x = self.classifier(x)

        return x

model = MedFairXNet().to(device)

print("MedFairXNet Loaded")

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# TRAIN
epochs = 5

print("Training Started")

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = (
            labels
            .unsqueeze(1)
            .float()
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

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss={running_loss:.4f}"
    )

# SAVE
torch.save(
    model.state_dict(),
    "medfairxnet_ph2.pth"
)

print("Model Saved")

# EVALUATION
model.eval()

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.sigmoid(outputs)

        preds = (
            probs > 0.5
        ).int()

        y_true.extend(
            labels.numpy()
        )

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
print("PH2 MedFairXNet Results")
print("==============================")
print("Accuracy    :", round(accuracy,4))
print("Precision   :", round(precision,4))
print("Sensitivity :", round(sensitivity,4))
print("F1 Score    :", round(f1,4))
print("ROC AUC     :", round(auc,4))
print("==============================")