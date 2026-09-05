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

# ==================================
# DEVICE
# ==================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

# ==================================
# DATASET
# ==================================

train_dataset = MelanomaDataset(
    "train.csv",
    "processed/isic2018",
    train_transform
)

test_dataset = MelanomaDataset(
    "test.csv",
    "processed/isic2018",
    val_test_transform
)

print("Train Size:", len(train_dataset))
print("Test Size :", len(test_dataset))

# ==================================
# DATALOADER
# ==================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# ==================================
# MEDFAIRXNET
# ==================================

class MedFairXNet(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )

        self.features = backbone.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.attention = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 1280),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(

            nn.Dropout(0.4),

            nn.Linear(
                1280,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                1
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        attention_weights = self.attention(x)

        x = x * attention_weights

        x = self.classifier(x)

        return x


model = MedFairXNet().to(device)

print("MedFairXNet Loaded")

# ==================================
# LOSS
# ==================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ==================================
# TRAIN
# ==================================

epochs = 1

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

        if batch_idx % 50 == 0:

            print(
                f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}"
            )

    print(
        f"Loss: {running_loss:.4f}"
    )

# ==================================
# SAVE
# ==================================

torch.save(
    model.state_dict(),
    "medfairxnet_isic2018.pth"
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

# ==================================
# METRICS
# ==================================

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
print("MedFairXNet Results")
print("==============================")
print("Accuracy    :", round(accuracy,4))
print("Precision   :", round(precision,4))
print("Sensitivity :", round(sensitivity,4))
print("F1 Score    :", round(f1,4))
print("ROC AUC     :", round(auc,4))
print("==============================")