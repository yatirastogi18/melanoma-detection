import pandas as pd
from sklearn.model_selection import train_test_split

# Load PH2
df = pd.read_excel(
    "PH2Dataset/PH2_dataset.xlsx",
    header=12
)

# Create labels
dataset = pd.DataFrame()

dataset["image_id"] = df["Image Name"]

dataset["label"] = df["Melanoma"].apply(
    lambda x: 1 if str(x).strip() == "X" else 0
)

# Train 70%
train_df, temp_df = train_test_split(
    dataset,
    test_size=0.30,
    stratify=dataset["label"],
    random_state=42
)

# Val 15%
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_df.to_csv("ph2_train.csv", index=False)
val_df.to_csv("ph2_val.csv", index=False)
test_df.to_csv("ph2_test.csv", index=False)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

print("\nTrain Labels")
print(train_df["label"].value_counts())

print("\nVal Labels")
print(val_df["label"].value_counts())

print("\nTest Labels")
print(test_df["label"].value_counts())