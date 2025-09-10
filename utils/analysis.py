import cv2
import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN

# --- Cải tiến từ notebook, phiên bản Pro hơn ---
def analyze_defect_types_pro(img_cv):
    """
    Phân tích và khoanh vùng các loại lỗi dựa trên hình thái học (morphology)
    và các đặc trưng của contour.
    """
    if img_cv is None:
        raise ValueError("Ảnh đầu vào không hợp lệ.")

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = float(h * w)
    img_out = img_cv.copy()

    # 1. Black-hat morphology để làm nổi bật các vùng tối (lỗi)
    k_size = max(15, (min(h, w) // 25) | 1) # Kích thước kernel thích ứng
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 2. Lọc nhiễu và tạo mask nhị phân
    blurred = cv2.GaussianBlur(blackhat, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)


    # 3. Tìm contours trên mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defect_types = []
    min_area = img_area * 0.0003 # Bỏ nhiễu siêu nhỏ
    max_area = img_area * 0.05   # Bỏ các vùng lớn có thể là nền

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0: continue

        # --- Phân loại dựa trên đặc trưng hình học ---
        circularity = 4 * np.pi * area / (peri * peri)
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        aspect_ratio = max(w_cnt, h_cnt) / (min(w_cnt, h_cnt) + 1e-6)

        defect_type = "Khong xac dinh"
        color = (0, 255, 255) # Vàng cho không xác định

        # Quy tắc phân loại
        if circularity < 0.35 and aspect_ratio > 2.5:
            defect_type = "Nut"
            color = (255, 0, 0) # Xanh dương cho Nứt
        elif circularity > 0.6 and aspect_ratio < 1.8:
            defect_type = "Lo khi"
            color = (0, 0, 255) # Đỏ cho Lỗ khí
        else:
            # Kiểm tra vị trí để xác định "Mẻ"
            border_margin = int(0.03 * min(h, w))
            if (x <= border_margin or y <= border_margin or
                x + w_cnt >= w - border_margin or y + h_cnt >= h - border_margin):
                defect_type = "Me"
                color = (0, 255, 0) # Xanh lá cho Mẻ
            else:
                 defect_type = "Tray xuoc"
                 color = (255, 0, 255) # Tím cho Trầy xước


        defect_types.append(defect_type)
        # Vẽ contour và ghi tên lỗi
        cv2.drawContours(img_out, [cnt], -1, color, 2)
        cv2.putText(img_out, defect_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    return len(defect_types), defect_types, img_out


# --- Hàm phân vùng ảnh ---
def segment_image_hybrid(img_resized):
    """
    Phân vùng ảnh sử dụng kết hợp DBSCAN và Fuzzy C-Means.
    """
    pixels = img_resized.reshape((-1, 3))
    pixels_norm = pixels / 255.0

    # 1. DBSCAN để tìm các pixel ngoại lai (outliers) - thường là lỗi
    dbscan = DBSCAN(eps=0.3, min_samples=20)
    db_labels = dbscan.fit_predict(pixels_norm)
    outlier_mask = (db_labels == -1)

    # 2. Fuzzy C-Means cho các pixel không phải outlier
    pixels_in = pixels_norm[~outlier_mask].T
    n_clusters = 2  # Nền và vật thể chính
    try:
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            pixels_in, n_clusters, 2, error=0.005, maxiter=1000
        )
    except Exception: # Xử lý trường hợp không còn pixel inlier
        labels_soft = np.zeros(len(pixels))
    else:
        labels_soft = np.zeros(len(pixels))
        labels_soft[~outlier_mask] = np.argmax(u, axis=0)

    # Gán outlier vào một cụm riêng
    labels_soft[outlier_mask] = n_clusters

    # 3. Tạo ảnh kết quả với màu ngẫu nhiên cho từng cụm
    segmented_img = np.zeros_like(pixels)
    unique_labels = np.unique(labels_soft)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3))

    for i, lbl in enumerate(unique_labels):
        segmented_img[labels_soft == lbl] = colors[i]

    return segmented_img.reshape(img_resized.shape)