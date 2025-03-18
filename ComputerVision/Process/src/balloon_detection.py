import cv2
import numpy as np
import os


def convert_lab_to_opencv(l_min, l_max, a_min, a_max, b_min, b_max):
    """
    Convert standard LAB values to OpenCV's LAB range.

    Parameters:
    l_min, l_max : int - Lightness min and max (unchanged)
    a_min, a_max : int - A channel min and max (shifted by +128)
    b_min, b_max : int - B channel min and max (shifted by +128)

    Returns:
    Tuple of np.uint8 arrays containing lower and upper LAB thresholds for OpenCV
    """
    lab_lower = np.array([l_min, a_min + 128, b_min + 128], dtype=np.uint8)
    lab_upper = np.array([l_max, a_max + 128, b_max + 128], dtype=np.uint8)

    return lab_lower, lab_upper
def detect_green_balloon_lab(image_path, output_path, processing_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"ERROR: File not found - {image_path}")
        return

    # Try reading the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: OpenCV failed to load image - {image_path}")
        return

    print(f"Image successfully loaded: {image_path}")

    # Resize image while maintaining aspect ratio
    image_resized = cv2.resize(image, (320, 240))

    # Convert image to LAB color space
    lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)

    # Convert standard LAB values to OpenCV's range
    lower_lab, upper_lab = convert_lab_to_opencv(0, 100, -19, -8, -9, 28)

    # Create a mask for the LAB color range
    lab_mask = cv2.inRange(lab, lower_lab, upper_lab)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image_resized, (3, 3), 1)

    # Apply edge detection only inside the LAB mask
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.bitwise_and(edges, lab_mask)  # Restrict edges to LAB mask areas

    # Apply morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    # Save intermediate processing results
    base_filename = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
    cv2.imwrite(os.path.join(processing_path, f"1_blurred_{base_filename}.jpg"), blurred)
    cv2.imwrite(os.path.join(processing_path, f"2_lab_mask_{base_filename}.jpg"), lab_mask)
    cv2.imwrite(os.path.join(processing_path, f"3_edges_dilated_{base_filename}.jpg"), edges_dilated)

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

def process_all_images_lab(input_dir, output_dir, processing_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(processing_dir):
        os.makedirs(processing_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            detect_green_balloon_lab(input_path, output_path, processing_dir)

# Example usage
input_directory = "../../data/Balloon"
output_directory = "../detection"
processing_directory = "../processing"
process_all_images_lab(input_directory, output_directory, processing_directory)
