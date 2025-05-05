import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from ultralytics import YOLO

def load_and_preprocess_image(image_path):
    """Load and preprocess the image containing all subimages and reference icon."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể tải hình ảnh từ {image_path}")
    return image

def extract_reference_icon(yolo_model, image):
    """Extract the reference icon (marked in red)."""
    results_yolo = yolo_model(image)
    for result in results_yolo:
        boxes = result.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            if y1 >= 200 and y2 <= 400 and x1 >= 0 and x2 <= 125:
                return image[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None

def extract_detected_icons(yolo_model, image):
    """Extract all detected icons in the top area of the image."""
    results_yolo = yolo_model(image)
    icons = []
    icon_positions = []
    for result in results_yolo:
        boxes = result.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            if y1 <= 200:  # Only consider icons in the top area
                icon = image[y1:y2, x1:x2]
                icons.append(icon)
                icon_positions.append((x1, y1, x2, y2))
    return icons, icon_positions

def compute_features(icon):
    """Compute features for comparison (using color histogram and SIFT features)."""
    if icon is None or icon.size == 0:
        return np.zeros(128 + 96)  # Return zeros if icon is invalid
        
    # Resize to standard size
    icon_resized = cv2.resize(icon, (64, 64))
    gray = cv2.cvtColor(icon_resized, cv2.COLOR_BGR2GRAY)

    # 1. Color histogram features
    hist_features = []
    for i in range(3):  # For each color channel
        hist = cv2.calcHist([icon_resized], [i], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        hist_features.extend(hist.flatten())
    
    # 2. SIFT features (if image is large enough)
    try:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None and len(descriptors) > 0:
            # Use average of descriptors as feature
            sift_feature = np.mean(descriptors, axis=0)
        else:
            sift_feature = np.zeros(128)  # Default SIFT descriptor size
    except:
        sift_feature = np.zeros(128)
    
    # Combine features
    all_features = np.concatenate((np.array(hist_features), sift_feature))
    return all_features

def compare_icons(reference_features, icons, icon_positions, threshold=0.3):
    """Compare reference icon with all detected icons and return matches."""
    matches = []
    similarity_scores = []
    
    for i, icon in enumerate(icons):
        icon_features = compute_features(icon)
        similarity = 1 - cosine(reference_features, icon_features)
        similarity_scores.append(similarity)
        
        if similarity > threshold:
            matches.append((i, icon_positions[i], similarity))
    
    # Sort matches by similarity score (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches, similarity_scores

def identify_subimages(matches, image_width, num_subimages=10):
    """Identify which subimage each matching icon belongs to."""
    subimage_width = image_width // num_subimages
    results = []
    
    for match_idx, (x1, y1, x2, y2), similarity in matches:
        # Calculate the center of the icon
        center_x = (x1 + x2) // 2
        
        # Determine which subimage this belongs to
        subimage_idx = center_x // subimage_width
        
        results.append((subimage_idx, match_idx, similarity))
    
    return results

def visualize_results(image, reference_icon_coords, matches, icon_positions, subimage_results):
    """Visualize the matching results."""
    result_img = image.copy()
    
    # Draw reference icon
    x1, y1, x2, y2 = reference_icon_coords
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw all detected icons
    for x1, y1, x2, y2 in icon_positions:
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Highlight matching icons
    for match_idx, (x1, y1, x2, y2), _ in matches:
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw subimage divisions
    subimage_width = image.shape[1] // 10
    for i in range(1, 10):
        x = i * subimage_width
        cv2.line(result_img, (x, 0), (x, image.shape[0]), (255, 255, 255), 1)
    
    # Add subimage numbers
    for i in range(10):
        x = i * subimage_width + subimage_width // 2
        cv2.putText(result_img, str(i+1), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("Icon Matching Results")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('captcha_results.png')
    plt.show()
    
    # Print results
    print("Biểu tượng khớp được tìm thấy trong các ảnh con sau:")
    subimage_matches = {}
    for subimage_idx, match_idx, similarity in subimage_results:
        subimage_number = subimage_idx + 1
        if subimage_number not in subimage_matches:
            subimage_matches[subimage_number] = []
        subimage_matches[subimage_number].append((match_idx, similarity))
    
    for subimage_number in sorted(subimage_matches.keys()):
        matches_in_subimage = subimage_matches[subimage_number]
        print(f"Ảnh con {subimage_number}:")
        for match_idx, similarity in matches_in_subimage:
            print(f"  - Icon {match_idx} (Độ tương đồng: {similarity:.2f})")

def process_captcha(image_path, yolo_model, threshold=0.3, num_subimages=10):
    """Main function to process the CAPTCHA image and find matching icons."""
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)
    
    # Extract reference icon
    reference_icon, ref_coords = extract_reference_icon(yolo_model, image)
    if reference_icon is None:
        print("Không tìm thấy biểu tượng tham chiếu! Vui lòng kiểm tra điều kiện phát hiện.")
        return []
    
    # Extract detected icons
    icons, icon_positions = extract_detected_icons(yolo_model, image)
    if len(icons) == 0:
        print("Không phát hiện được biểu tượng nào trong vùng trên cùng!")
        return []
    
    print(f"Đã phát hiện {len(icons)} biểu tượng trong vùng trên cùng.")
    
    # Compute features for reference icon
    reference_features = compute_features(reference_icon)
    
    # Compare reference icon with all detected icons
    matches, similarity_scores = compare_icons(reference_features, icons, icon_positions, threshold)
    print(f"Tìm thấy {len(matches)} biểu tượng khớp với ngưỡng {threshold}.")
    
    # Identify which subimage each matching icon belongs to
    subimage_results = identify_subimages(matches, image.shape[1], num_subimages)
    
    # Visualize results
    visualize_results(image, ref_coords, matches, icon_positions, subimage_results)
    
    # Get unique subimage indices to click
    subimages_to_click = sorted(list(set([subimg_idx + 1 for subimg_idx, _, _ in subimage_results])))
    print(f"\nGiải pháp CAPTCHA: Nhấp vào các ảnh con {subimages_to_click}")
    
    return subimages_to_click

def solve_captcha(image_path, model_path, threshold=0.3):
    """Solve CAPTCHA using the trained YOLOv8 model."""
    # Load YOLOv8 model
    model = YOLO(model_path)
    
    # Process the CAPTCHA
    solution = process_captcha(image_path, model, threshold)
    
    return solution


# Sử dụng ví dụ:
if __name__ == "__main__":
    image_path = "CloneProject-7/train/images/9835266098_jpg.rf.44b8b548246399a93efacc6c48a12eaa.jpg"
    model_path = "weights/best.pt"
    threshold = 0.3
    solution = solve_captcha(image_path, model_path, threshold)
    print(f"Giải pháp CAPTCHA: {solution}")