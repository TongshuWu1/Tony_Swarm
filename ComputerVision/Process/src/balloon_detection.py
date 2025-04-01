import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def convert_lab_to_opencv(l_min, l_max, a_min, a_max, b_min, b_max):
    lab_lower = np.array([l_min, a_min + 128, b_min + 128], dtype=np.uint8)
    lab_upper = np.array([l_max, a_max + 128, b_max + 128], dtype=np.uint8)
    lab_center = np.array([(l_min + l_max) // 2, (a_min + a_max) // 2 + 128, (b_min + b_max) // 2 + 128], dtype=np.uint8)
    return lab_lower, lab_upper, lab_center

def compute_lab_confidence(lab_image, center, sigma):
    lab_flat = lab_image.reshape((-1, 3)).astype(np.float32)
    diff = lab_flat - center.astype(np.float32)
    d_squared = np.sum((diff / sigma)**2, axis=1)
    confidence = np.exp(-0.5 * d_squared)
    return confidence.reshape((lab_image.shape[0], lab_image.shape[1]))

def save_lab_confidence_map(confidence_map, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confidence_map, cmap='hot', interpolation='nearest')
    ax.set_title("LAB Gaussian Confidence")
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Confidence", rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

def save_confidence_grid_overlay(confidence_map, output_path, cell_size=20):
    """Overlay 5x5 grid on the confidence map and annotate average confidence in each cell."""
    h, w = confidence_map.shape
    num_rows, num_cols = h // cell_size, w // cell_size

    norm_conf_map = (confidence_map * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(norm_conf_map, cv2.COLORMAP_HOT)
    overlay = color_map.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1

    for i in range(num_rows):
        for j in range(num_cols):
            y_start, y_end = i * cell_size, (i + 1) * cell_size
            x_start, x_end = j * cell_size, (j + 1) * cell_size

            # Draw cell border
            cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)

            # Calculate and annotate average confidence
            cell_conf = np.mean(confidence_map[y_start:y_end, x_start:x_end])
            text = f"{cell_conf:.2f}"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = x_start + (cell_size - text_size[0]) // 2
            text_y = y_start + (cell_size + text_size[1]) // 2
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, overlay)
def detect_green_balloon_lab(image_path, output_path, processing_path):
    if not os.path.exists(image_path):
        print(f"ERROR: File not found - {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: OpenCV failed to load image - {image_path}")
        return

    print(f"Image successfully loaded: {image_path}")

    image_resized = cv2.resize(image, (320, 240))
    blurred = cv2.GaussianBlur(image_resized, (3, 3), 2)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # Gaussian center points
    _, _, center_lab1 = convert_lab_to_opencv(0, 100, -29, -9, -4, 30)
    _, _, center_lab2 = convert_lab_to_opencv(42, 100, -30, 4, 9, 45)
    sigma_lab = np.array([15, 10, 10])

    # Confidence computation
    conf1 = compute_lab_confidence(lab, center_lab1, sigma_lab)
    conf2 = compute_lab_confidence(lab, center_lab2, sigma_lab)
    confidence_map = 0.5 * conf1 + 0.5 * conf2

    # Threshold for binary mask (used only for contour detection)
    lab_mask = ((conf1 > 0.15) | (conf2 > 0.15)).astype(np.uint8) * 255

    # Save diagnostics
    base_filename = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
    confidence_map_path = os.path.join(processing_path, f"confidence_map_{base_filename}.jpg")
    binary_mask_path = os.path.join(processing_path, f"binary_mask_{base_filename}.jpg")
    grid_overlay_path = os.path.join(processing_path, f"confidence_grid_{base_filename}.jpg")

    save_lab_confidence_map(confidence_map, confidence_map_path)
    save_confidence_grid_overlay(confidence_map, grid_overlay_path)
    cv2.imwrite(binary_mask_path, lab_mask)

    # Find contours directly from thresholded confidence
    contours, _ = cv2.findContours(lab_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image_resized.copy()
    h, w = confidence_map.shape
    cell_size = 5
    num_rows, num_cols = h // cell_size, w // cell_size
    cell_confidence = np.zeros((num_rows, num_cols))

    # Compute average confidence for each cell
    for i in range(num_rows):
        for j in range(num_cols):
            y_start, y_end = i * cell_size, (i + 1) * cell_size
            x_start, x_end = j * cell_size, (j + 1) * cell_size
            cell = confidence_map[y_start:y_end, x_start:x_end]
            cell_confidence[i, j] = np.mean(cell)

    # Find best contour
    best_score = -1
    best_contour = None
    best_hull = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            hull = cv2.convexHull(contour)
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [hull], -1, 255, -1)

            score = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    y_start, y_end = i * cell_size, (i + 1) * cell_size
                    x_start, x_end = j * cell_size, (j + 1) * cell_size
                    if y_end > h or x_end > w:
                        continue
                    cell_mask = contour_mask[y_start:y_end, x_start:x_end]
                    if np.any(cell_mask):
                        score += cell_confidence[i, j]

            if score > best_score:
                best_score = score
                best_contour = contour
                best_hull = hull

    # Draw result
    if best_contour is not None:
        cv2.drawContours(result, [best_hull], -1, (0, 255, 0), 2)

        if len(best_hull) > 5:
            ellipse = cv2.fitEllipse(best_hull)
            cv2.ellipse(result, ellipse, (255, 0, 0), 2)

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
