import pandas as pd
from sklearn.model_selection import train_test_split

gt = pd.read_csv(
    r"ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv"
)

gt["label"] = gt["MEL"]

gt = gt[["image", "label"]]

train_df, temp_df = train_test_split(
    gt,
    test_size=0.30,
    stratify=gt["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df.to_csv(
    "isic_train_final.csv",
    index=False
)

val_df.to_csv(
    "isic_val_final.csv",
    index=False
)

test_df.to_csv(
    "isic_test_final.csv",
    index=False
)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

print("\nTrain Labels")
print(train_df["label"].value_counts())

print("\nVal Labels")
print(val_df["label"].value_counts())

print("\nTest Labels")
print(test_df["label"].value_counts())