from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

folders = [
    Path("PH2Dataset/PH2 Dataset images"),
    Path("ISIC2018_Task3_Training_Input"),
    Path("dataverse_files/HAM10000_images_combined_600x450")
]

for folder in folders:

    print(f"\nProcessing {folder}")

    count = 0

    for ext in IMAGE_EXTENSIONS:

        for img_path in folder.rglob(f"*{ext}"):

            try:

                img = Image.open(img_path)

                img = img.convert("RGB")

                img.save(img_path)

                count += 1

            except Exception as e:

                print("Error:", img_path)

    print("Standardized:", count)