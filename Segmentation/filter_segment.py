import cv2
import numpy as np

def segment_mammogram(image_path):
    # Read the mammogram
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Applying GaussianBlur to reduce image noise and improve contour detection
    img_blur = cv2.GaussianBlur(img, (5,5), 0)

    # Apply Otsu's thresholding
    _, binary_img = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area (optional)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # Draw contours
    segmented_image = cv2.drawContours(np.zeros_like(img), filtered_contours, -1, (255), thickness=cv2.FILLED)

    return segmented_image

segmented_image = segment_mammogram("/Users/MayraBerrones/Documents/DDSM-CIBIS/Crop/Calc-Test_P_00077_LEFT_MLO_CROP_1.png")
cv2.imwrite("/Users/MayraBerrones/Desktop/Codigo_articulo/Segmented_Calc-Test_P_00077_LEFT_MLO_CROP_1.png", segmented_image)