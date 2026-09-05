# from pathlib import Path

# image_id = "ISIC_0027419"

# for path in Path(".").rglob(f"{image_id}.jpg"):
#     print(path)

# import pandas as pd

# df = pd.read_csv("train.csv")

# print(df["dataset"].value_counts())
import pandas as pd

df = pd.read_csv("train.csv")

print(df["label"].value_counts())