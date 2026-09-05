import cv2

image = cv2.imread("clahe_output.jpg")

normalized = cv2.normalize(
    image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

cv2.imwrite(
    "normalized_output.jpg",
    normalized
)

print("Color normalization completed")