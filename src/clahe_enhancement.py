import cv2

image_path = "hair_removed.jpg"

image = cv2.imread(image_path)

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

l_clahe = clahe.apply(l)

enhanced_lab = cv2.merge(
    (l_clahe, a, b)
)

enhanced = cv2.cvtColor(
    enhanced_lab,
    cv2.COLOR_LAB2BGR
)

cv2.imwrite(
    "clahe_output.jpg",
    enhanced
)

print("CLAHE completed")