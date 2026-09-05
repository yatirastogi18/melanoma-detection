# import pandas as pd

# ham = pd.read_csv("HAM10000_metadata.csv")

# print(ham.columns)
# print(ham.head())
# print(ham["dx"].value_counts())
import pandas as pd

# =========================
# HAM10000
# =========================

ham = pd.read_csv("HAM10000_metadata.csv")

ham["label"] = ham["dx"].apply(
    lambda x: 1 if x.lower() == "mel" else 0
)

ham["dataset"] = "ham10000"

ham_output = ham[
    ["image_id", "lesion_id", "label", "dataset"]
]

ham_output.to_csv(
    "ham10000_harmonized.csv",
    index=False
)

print("HAM10000 Harmonized")
print(ham_output["label"].value_counts())
