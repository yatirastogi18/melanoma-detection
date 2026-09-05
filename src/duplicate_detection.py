from pathlib import Path
from PIL import Image
import imagehash

image_dir = Path(
    "dataverse_files/HAM10000_images_combined_600x450"
)

hashes = {}
duplicates = []

for img_path in image_dir.glob("*.jpg"):

    try:
        img_hash = str(
            imagehash.phash(
                Image.open(img_path)
            )
        )

        if img_hash in hashes:

            duplicates.append(
                (str(img_path),
                 str(hashes[img_hash]))
            )

        else:
            hashes[img_hash] = img_path

    except Exception as e:
        print("Error:", img_path)

print("\nTotal Images:", len(hashes))
print("Duplicates Found:", len(duplicates))

for d in duplicates[:20]:
    print(d)