import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import normalize
from ultralytics import YOLO
import time
from skimage.feature import hog
from skimage import exposure

def load_and_preprocess_image(image_path):
    """Load and preprocess the image containing all subimages and reference icon."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể tải hình ảnh từ {image_path}")
    
    # Cải thiện: Áp dụng các bước tiền xử lý
    # 1. Chuyển đổi sang không gian màu khác để tăng độ tương phản
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. Cân bằng histogram cho kênh L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 3. Hợp nhất các kênh
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 4. Làm giảm nhiễu
    enhanced_img = cv2.fastNlMeansDenoisingColored(enhanced_img, None, 10, 10, 7, 21)
    
    return enhanced_img

def extract_reference_icon(yolo_model, image, confidence_threshold=0.5):
    """Extract the reference icon (marked in red) with improved confidence threshold."""
    results_yolo = yolo_model(image, conf=confidence_threshold)
    
    # Tìm kiếm biểu tượng tham chiếu với độ tin cậy cao nhất trong vùng dưới
    best_confidence = 0
    best_box = None
    
    for result in results_yolo:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(confidences[i])
            
            # Kiểm tra nếu hộp nằm trong vùng tham chiếu (phía dưới)
            if y1 >= 200 and conf > best_confidence:
                best_confidence = conf
                best_box = (x1, y1, x2, y2)
    
    if best_box:
        x1, y1, x2, y2 = best_box
        return image[y1:y2, x1:x2], best_box, best_confidence
    
    # Thử phương pháp dự phòng nếu YOLO không tìm thấy
    # Phát hiện vùng đỏ (thường dùng để đánh dấu biểu tượng tham chiếu)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    red_mask = mask1 + mask2
    
    # Tìm contour trong vùng đỏ
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        if y >= 200:  # Vùng dưới
            return image[y:y+h, x:x+w], (x, y, x+w, y+h), 0.5
    
    return None, None, 0

def extract_detected_icons(yolo_model, image, confidence_threshold=0.5):
    """Extract all detected icons in the top area of the image with improved confidence threshold."""
    results_yolo = yolo_model(image, conf=confidence_threshold)
    icons = []
    icon_positions = []
    confidences = []
    
    for result in results_yolo:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(confs[i])
            
            # Chỉ xem xét biểu tượng ở vùng trên với độ tin cậy đủ cao
            if y1 <= 200:
                icon = image[y1:y2, x1:x2]
                icons.append(icon)
                icon_positions.append((x1, y1, x2, y2))
                confidences.append(conf)
    
    return icons, icon_positions, confidences

def compute_features(icon, feature_types=["color", "hog", "sift"]):
    """
    Compute multiple feature types for better comparison:
    - Color histograms (RGB and HSV)
    - HOG features
    - SIFT features
    """
    if icon is None or icon.size == 0:
        # Return zeros with appropriate size based on feature types
        feature_size = 0
        if "color" in feature_types: feature_size += 96
        if "hog" in feature_types: feature_size += 324
        if "sift" in feature_types: feature_size += 128
        return np.zeros(feature_size)
        
    # Resize to standard size
    icon_resized = cv2.resize(icon, (64, 64))
    gray = cv2.cvtColor(icon_resized, cv2.COLOR_BGR2GRAY)
    
    all_features = []
    
    # 1. Color histogram features (RGB and HSV)
    if "color" in feature_types:
        # RGB histogram
        color_features = []
        for i in range(3):  # For each color channel
            hist = cv2.calcHist([icon_resized], [i], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            color_features.extend(hist.flatten())
        
        # HSV histogram for better illumination invariance
        hsv = cv2.cvtColor(icon_resized, cv2.COLOR_BGR2HSV)
        hsv_features = []
        channels = [0, 1]  # Use only Hue and Saturation
        for i in channels:
            hist = cv2.calcHist([hsv], [i], None, [16], [0, 256])
            cv2.normalize(hist, hist)
            hsv_features.extend(hist.flatten())
            
        all_features.extend(color_features)
        all_features.extend(hsv_features)
    
    # 2. HOG features for shape description
    if "hog" in feature_types:
        try:
            fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)
            all_features.extend(fd)
        except:
            all_features.extend(np.zeros(324))  # Default HOG size for these params
    
    # 3. SIFT features for distinct keypoints
    if "sift" in feature_types:
        try:
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is not None and len(descriptors) > 0:
                # Use average of descriptors as feature
                sift_feature = np.mean(descriptors, axis=0)
            else:
                sift_feature = np.zeros(128)  # Default SIFT descriptor size
            all_features.extend(sift_feature)
        except:
            all_features.extend(np.zeros(128))
    
    # Normalize the combined features
    feature_vector = np.array(all_features)
    if len(feature_vector) > 0:
        feature_vector = normalize(feature_vector.reshape(1, -1))[0]
    
    return feature_vector

def orb_feature_matching(reference_icon, target_icon):
    """Sử dụng ORB để so khớp đặc trưng cục bộ."""
    # Chuyển sang ảnh xám
    if reference_icon is None or target_icon is None:
        return 0.0
        
    try:
        ref_gray = cv2.cvtColor(reference_icon, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_icon, cv2.COLOR_BGR2GRAY)
        
        # Khởi tạo ORB
        orb = cv2.ORB_create(nfeatures=500)
        
        # Tìm keypoint và descriptors
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        kp2, des2 = orb.detectAndCompute(target_gray, None)
        
        # Kiểm tra xem có đủ điểm đặc trưng không
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0
        
        # Tạo matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # So khớp các descriptors
        matches = bf.match(des1, des2)
        
        # Sắp xếp theo khoảng cách (thấp hơn là tốt hơn)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Tính điểm tương đồng dựa trên số lượng khớp tốt và khoảng cách trung bình
        good_matches = matches[:min(10, len(matches))]
        if len(good_matches) > 0:
            avg_distance = sum(m.distance for m in good_matches) / len(good_matches)
            match_ratio = len(good_matches) / max(len(kp1), len(kp2))
            # Chuyển đổi khoảng cách thành điểm tương đồng
            similarity = match_ratio * (1 - min(avg_distance / 100.0, 1.0))
        else:
            similarity = 0.0
        
        return similarity
    except Exception as e:
        print(f"Lỗi trong orb_feature_matching: {e}")
        return 0.0

def compare_icons(reference_features, icons, icon_positions, icon_confidences=None, 
                 similarity_methods=["cosine", "euclidean", "orb"], adaptive_threshold=True):
    """
    Compare reference icon with all detected icons using multiple similarity measures
    and optional adaptive thresholding.
    """
    matches = []
    similarity_scores = []
    
    # Tính toán ngưỡng thích ứng nếu được bật
    if adaptive_threshold:
        all_similarities = []
        for i, icon in enumerate(icons):
            icon_features = compute_features(icon)
            
            # Tính toán điểm tương đồng sử dụng nhiều phương pháp
            similarities = []
            if "cosine" in similarity_methods:
                cosine_sim = 1 - cosine(reference_features, icon_features)
                similarities.append(cosine_sim)
            
            if "euclidean" in similarity_methods:
                # Chuyển đổi khoảng cách Euclidean thành điểm tương đồng
                euclidean_dist = euclidean(reference_features, icon_features)
                euclidean_sim = 1 / (1 + euclidean_dist)  # Chuyển đổi thành khoảng [0,1]
                similarities.append(euclidean_sim)
                
            # Thêm so khớp đặc trưng ORB
            if "orb" in similarity_methods and len(reference_icon_cache) > 0:
                orb_sim = orb_feature_matching(reference_icon_cache[0], icon)
                similarities.append(orb_sim)
            
            # Tính điểm tương đồng trung bình
            avg_similarity = np.mean(similarities)
            all_similarities.append(avg_similarity)
        
        # Tính ngưỡng thích ứng dựa trên phân phối điểm tương đồng
        if all_similarities:
            similarities_array = np.array(all_similarities)
            mean_sim = np.mean(similarities_array)
            std_sim = np.std(similarities_array)
            
            # Ngưỡng thích ứng: chọn các điểm nằm trên X độ lệch chuẩn so với giá trị trung bình
            threshold = mean_sim + 1.0 * std_sim
            
            # Đảm bảo ngưỡng nằm trong khoảng hợp lý
            threshold = max(0.3, min(threshold, 0.8))
        else:
            threshold = 0.3  # Ngưỡng mặc định
    else:
        threshold = 0.3  # Ngưỡng cố định
    
    # So sánh thực tế với ngưỡng đã xác định
    for i, icon in enumerate(icons):
        icon_features = compute_features(icon)
        
        # Tính toán điểm tương đồng sử dụng nhiều phương pháp
        similarities = []
        if "cosine" in similarity_methods:
            cosine_sim = 1 - cosine(reference_features, icon_features)
            similarities.append(cosine_sim)
        
        if "euclidean" in similarity_methods:
            euclidean_dist = euclidean(reference_features, icon_features)
            euclidean_sim = 1 / (1 + euclidean_dist)
            similarities.append(euclidean_sim)
            
        # Thêm so khớp đặc trưng ORB
        if "orb" in similarity_methods and len(reference_icon_cache) > 0:
            orb_sim = orb_feature_matching(reference_icon_cache[0], icon)
            similarities.append(orb_sim)
        
        # Tính điểm tương đồng trung bình
        avg_similarity = np.mean(similarities)
        similarity_scores.append(avg_similarity)
        
        # Áp dụng hệ số tin cậy YOLO nếu có
        if icon_confidences:
            confidence_factor = icon_confidences[i]
            # Kết hợp điểm tương đồng với độ tin cậy YOLO
            weighted_similarity = 0.7 * avg_similarity + 0.3 * confidence_factor
        else:
            weighted_similarity = avg_similarity
            
        if weighted_similarity > threshold:
            matches.append((i, icon_positions[i], weighted_similarity))
    
    # Sắp xếp các kết quả khớp theo điểm tương đồng (cao nhất đầu tiên)
    matches.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Ngưỡng tương đồng được sử dụng: {threshold:.3f}")
    return matches, similarity_scores, threshold

# Add a global cache for reference icon
reference_icon_cache = []

def identify_subimages(matches, image_width, num_subimages=10):
    """Identify which subimage each matching icon belongs to."""
    subimage_width = image_width // num_subimages
    results = []
    
    for match_idx, (x1, y1, x2, y2), similarity in matches:
        # Tính toán tâm của biểu tượng
        center_x = (x1 + x2) // 2
        
        # Xác định biểu tượng thuộc ảnh con nào
        subimage_idx = center_x // subimage_width
        
        results.append((subimage_idx, match_idx, similarity))
    
    return results

def visualize_results(image, reference_icon_coords, ref_confidence, matches, icon_positions, 
                      icon_confidences, subimage_results, threshold):
    """Visualize the matching results with improved visualization."""
    result_img = image.copy()
    
    # Vẽ biểu tượng tham chiếu
    if reference_icon_coords:
        x1, y1, x2, y2 = reference_icon_coords
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(result_img, f"Reference {ref_confidence:.2f}", (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Vẽ tất cả các biểu tượng đã phát hiện với độ tin cậy
    for i, ((x1, y1, x2, y2), conf) in enumerate(zip(icon_positions, icon_confidences)):
        label = f"{i}:{conf:.2f}"
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(result_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Đánh dấu các biểu tượng khớp
    for match_idx, (x1, y1, x2, y2), similarity in matches:
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"Match:{similarity:.2f}"
        cv2.putText(result_img, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Vẽ đường phân chia ảnh con
    subimage_width = image.shape[1] // 10
    for i in range(1, 10):
        x = i * subimage_width
        cv2.line(result_img, (x, 0), (x, image.shape[0]), (255, 255, 255), 1)
    
    # Thêm số thứ tự ảnh con
    for i in range(10):
        x = i * subimage_width + subimage_width // 2
        cv2.putText(result_img, str(i+1), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Tạo ảnh kết quả phụ hiển thị các ảnh con được chọn
    subimages_to_click = sorted(list(set([subimg_idx + 1 for subimg_idx, _, _ in subimage_results])))
    solution_img = image.copy()
    
    # Đánh dấu các ảnh con cần nhấp vào
    for subimg_idx in [idx-1 for idx in subimages_to_click]:
        x1 = subimg_idx * subimage_width
        x2 = (subimg_idx + 1) * subimage_width
        cv2.rectangle(solution_img, (x1, 0), (x2, 200), (0, 255, 255), 3)
        
    # Hiển thị thông tin về ngưỡng và kết quả
    cv2.putText(result_img, f"Threshold: {threshold:.3f}", (10, image.shape[0]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("Icon Matching Results")
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(solution_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Solution: Click on subimages {subimages_to_click}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('captcha_results.png', dpi=300)
    plt.show()
    
    # In kết quả
    print(f"\nNgưỡng tương đồng được sử dụng: {threshold:.3f}")
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

def process_captcha(image_path, yolo_model, threshold=None, num_subimages=10, 
                    confidence_threshold=0.5, adaptive_threshold=True):
    """Main function to process the CAPTCHA image and find matching icons with improvements."""
    global reference_icon_cache
    start_time = time.time()
    
    # Tải và tiền xử lý hình ảnh
    image = load_and_preprocess_image(image_path)
    
    # Trích xuất biểu tượng tham chiếu
    reference_icon, ref_coords, ref_confidence = extract_reference_icon(yolo_model, image, confidence_threshold)
    if reference_icon is None:
        print("Không tìm thấy biểu tượng tham chiếu! Vui lòng kiểm tra điều kiện phát hiện.")
        return []
    
    # Lưu reference icon vào cache
    reference_icon_cache = [reference_icon]
    
    # Trích xuất các biểu tượng đã phát hiện
    icons, icon_positions, icon_confidences = extract_detected_icons(yolo_model, image, confidence_threshold)
    if len(icons) == 0:
        print("Không phát hiện được biểu tượng nào trong vùng trên cùng!")
        return []
    
    print(f"Đã phát hiện {len(icons)} biểu tượng trong vùng trên cùng.")
    
    # Tính toán đặc trưng cho biểu tượng tham chiếu
    reference_features = compute_features(reference_icon, feature_types=["color", "hog", "sift"])
    
    # So sánh biểu tượng tham chiếu với tất cả các biểu tượng đã phát hiện
    matches, similarity_scores, used_threshold = compare_icons(
        reference_features, icons, icon_positions, icon_confidences,
        similarity_methods=["cosine", "euclidean", "orb"], 
        adaptive_threshold=adaptive_threshold
    )
    
    print(f"Tìm thấy {len(matches)} biểu tượng khớp với ngưỡng {used_threshold:.3f}.")
    
    # Xác định mỗi biểu tượng khớp thuộc ảnh con nào
    subimage_results = identify_subimages(matches, image.shape[1], num_subimages)
    
    # Hiển thị kết quả
    visualize_results(image, ref_coords, ref_confidence, matches, icon_positions, 
                     icon_confidences, subimage_results, used_threshold)
    
    # Lấy các chỉ số ảnh con duy nhất để nhấp vào
    subimages_to_click = sorted(list(set([subimg_idx + 1 for subimg_idx, _, _ in subimage_results])))
    
    # In thời gian xử lý
    elapsed_time = time.time() - start_time
    print(f"\nThời gian xử lý: {elapsed_time:.2f} giây")
    print(f"Giải pháp CAPTCHA: Nhấp vào các ảnh con {subimages_to_click}")
    
    return subimages_to_click

def rotate_icon(icon, angle):
    """Rotate an icon by the specified angle."""
    height, width = icon.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(icon, rotation_matrix, (width, height), 
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated

def data_augmentation(icon):
    """Generate augmented versions of the icon for more robust matching."""
    augmented_icons = []
    
    # Thêm icon gốc
    augmented_icons.append(icon)
    
    # Xoay icon
    for angle in [10, -10, 20, -20]:
        rotated = rotate_icon(icon, angle)
        augmented_icons.append(rotated)
    
    # Thay đổi độ sáng
    bright = cv2.convertScaleAbs(icon, alpha=1.2, beta=10)
    dark = cv2.convertScaleAbs(icon, alpha=0.8, beta=-10)
    augmented_icons.append(bright)
    augmented_icons.append(dark)
    
    # Lật ngang
    flipped = cv2.flip(icon, 1)
    augmented_icons.append(flipped)
    
    return augmented_icons

def solve_captcha(image_path, model_path, confidence_threshold=0.5, adaptive_threshold=True):
    """Solve CAPTCHA using the trained YOLOv8 model with improvements."""
    # Tải model YOLOv8
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return []
    
    # Xử lý CAPTCHA
    solution = process_captcha(
        image_path, 
        model, 
        confidence_threshold=confidence_threshold,
        adaptive_threshold=adaptive_threshold
    )
    
    return solution


# Sử dụng ví dụ:
if __name__ == "__main__":
    image_path = "CloneProject-7/train/images/9835266098_jpg.rf.44b8b548246399a93efacc6c48a12eaa.jpg"
    model_path = "weights/best.pt"
    confidence_threshold = 0.89
    adaptive_threshold = True
    solution = solve_captcha(image_path, model_path, confidence_threshold, adaptive_threshold)
    print(f"Giải pháp CAPTCHA: {solution}")