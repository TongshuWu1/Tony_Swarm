import cv2
import numpy as np
import os


def convert_lab_to_opencv(l_min, l_max, a_min, a_max, b_min, b_max):
    lab_lower = np.array([l_min, a_min + 128, b_min + 128], dtype=np.uint8)
    lab_upper = np.array([l_max, a_max + 128, b_max + 128], dtype=np.uint8)
    return lab_lower, lab_upper


def detect_green_balloon_lab(image_path, output_path, processing_path):
    if not os.path.exists(image_path):
        print(f"ERROR: File not found - {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: OpenCV failed to load image - {image_path}")
        return

    print(f"Image successfully loaded: {image_path}")

    # Resize image for consistent processing
    image_resized = cv2.resize(image, (320, 240))

    # Apply Gaussian Blur BEFORE color conversion (Preserves smoothness)
    blurred = cv2.GaussianBlur(image_resized, (3, 3), 2)

    # Convert blurred image to LAB color space
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # Apply multiple LAB thresholding ranges to capture shadows and highlights
    lower_lab1, upper_lab1 = convert_lab_to_opencv(0, 100, -29, -9, -4, 30)  # Normal balloon color
    lower_lab2, upper_lab2 = convert_lab_to_opencv(43, 100, -32, -7, 13, 46) # Handling highlights

    mask1 = cv2.inRange(lab, lower_lab1, upper_lab1)
    mask2 = cv2.inRange(lab, lower_lab2, upper_lab2)

    # Ensure masks are the same size and type
    assert mask1.shape == mask2.shape, "Mask dimensions do not match!"
    assert mask1.dtype == mask2.dtype, "Mask data types do not match!"

    # Morphological operations
    kernel = np.ones((4, 4), np.uint8)

    # Apply Opening to remove noise in mask1 and mask2
    opened_mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=1)
    opened_mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply Closing to fill gaps in opened_mask1 and opened_mask2
    closed_mask1 = cv2.morphologyEx(opened_mask1, cv2.MORPH_CLOSE, kernel, iterations=3)
    closed_mask2 = cv2.morphologyEx(opened_mask2, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Combine masks correctly
    lab_mask = cv2.bitwise_or(closed_mask1, closed_mask2)  # Logical OR to merge both masks

    # Optional Debugging: Visualize Combined Mask
    combined_mask_visual = cv2.addWeighted(closed_mask1, 0.5, closed_mask2, 0.5, 0)  # Merge for visualization

    # Save intermediate results for debugging
    base_filename = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
    cv2.imwrite(os.path.join(processing_path, f"2_mask1_{base_filename}.jpg"), mask1)
    cv2.imwrite(os.path.join(processing_path, f"3_mask2_{base_filename}.jpg"), mask2)
    cv2.imwrite(os.path.join(processing_path, f"4_opened_mask1_{base_filename}.jpg"), opened_mask1)
    cv2.imwrite(os.path.join(processing_path, f"5_opened_mask2_{base_filename}.jpg"), opened_mask2)
    cv2.imwrite(os.path.join(processing_path, f"6_closed_mask1_{base_filename}.jpg"), closed_mask1)
    cv2.imwrite(os.path.join(processing_path, f"7_closed_mask2_{base_filename}.jpg"), closed_mask2)
    cv2.imwrite(os.path.join(processing_path, f"8_combined_mask_{base_filename}.jpg"), lab_mask)
    cv2.imwrite(os.path.join(processing_path, f"9_combined_mask_visual_{base_filename}.jpg"), combined_mask_visual)

    # Apply Opening First (removes small noise)
    opened_mask = cv2.morphologyEx(lab_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply Closing After (fills gaps)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours of the detected regions
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image_resized.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter out small contours
            # Compute convex hull to fill missing parts due to reflections
            hull = cv2.convexHull(contour)
            cv2.drawContours(result, [hull], -1, (0, 255, 0), 2)

            # Flood fill to close small gaps in the detected balloon
            mask = np.zeros((closed_mask.shape[0] + 2, closed_mask.shape[1] + 2), np.uint8)
            cv2.floodFill(closed_mask, mask, (hull[0][0][0], hull[0][0][1]), 255)

            # Draw bounding ellipse
            if len(hull) > 5:
                ellipse = cv2.fitEllipse(hull)
                cv2.ellipse(result, ellipse, (255, 0, 0), 2)  # Red ellipse

    # Save final result
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