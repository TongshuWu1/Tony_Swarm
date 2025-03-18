import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def detect_green_balloon(image_path, output_path, processing_path):
    # Load the image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (320, 240))  # Resize for consistency
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    # Define green color range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply Gaussian Blur to reduce background noise
    blurred = cv2.GaussianBlur(image_resized, (9, 9), 3)

    # Apply edge detection only inside the green mask
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.bitwise_and(edges, green_mask)  # Restrict edges to green areas

    # Apply erosion to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    # Apply dilation to connect fragmented edges
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    edges_eroded = cv2.erode(edges_dilated, kernel, iterations=2)



    # Save intermediate processing results with categorized naming
    base_filename = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
    cv2.imwrite(os.path.join(processing_path, f"1_blurred_{base_filename}.jpg"), blurred)
    cv2.imwrite(os.path.join(processing_path, f"2_filtered_{base_filename}.jpg"), green_mask)
    cv2.imwrite(os.path.join(processing_path, f"3_edges_eroded_{base_filename}.jpg"), edges_eroded)
    cv2.imwrite(os.path.join(processing_path, f"4_edges_dilated_{base_filename}.jpg"), edges_dilated)

    # Find contours from the masked edge-detected image
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on detected balloon
    result = image_resized.copy()
    max_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:  # Keep track of the largest detected object
            max_area = area
            max_contour = contour

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box: Bounding box around the detected object

        # Optional: Fit an ellipse if the contour is large enough
        if len(max_contour) > 5:
            ellipse = cv2.fitEllipse(max_contour)
            cv2.ellipse(result, ellipse, (255, 0, 0), 2)  # Red/blue ellipse: Best-fit shape around the detected object

    # Save the processed result
    cv2.imwrite(output_path, result)


def process_all_images(input_dir, output_dir, processing_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(processing_dir):
        os.makedirs(processing_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            detect_green_balloon(input_path, output_path, processing_dir)


# Example usage
input_directory = "../../data/Balloon"
output_directory = "../detection"
processing_directory = "../processing"
process_all_images(input_directory, output_directory, processing_directory)
