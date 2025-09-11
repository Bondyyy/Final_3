import streamlit as st
from PIL import Image
import cv2
import numpy as np

# --- Các tham số cho việc phát hiện lỗi (Từ file s.py) ---
# Bạn có thể tinh chỉnh các giá trị này để thay đổi độ nhạy
PARAMS = {
    "min_area_ratio": 0.0001,
    "max_area_ratio": 0.1,
    "crack_aspect_ratio_thresh": 4.0,
    "chip_circularity_thresh": 0.5,
    "pore_circularity_thresh": 0.7,
}

# --- Các hàm xử lý ảnh (Từ file s.py) ---

def preprocess_blackhat(gray_image):
    """Sử dụng Black Hat để làm nổi bật các vùng tối trên nền sáng."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    return mask

def classify_contour(contour, image_area, image_shape):
    """Phân loại một contour thành Nứt, Mẻ, hoặc Lỗ khí."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # 1. Phân loại "Nứt" (Crack) dựa trên tỷ lệ khung hình
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    if aspect_ratio > PARAMS["crack_aspect_ratio_thresh"]:
        return "Nứt", area, perimeter

    # 2. Phân loại "Mẻ" (Chip) và "Lỗ khí" (Pore) dựa trên độ tròn
    if perimeter > 0:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < PARAMS["chip_circularity_thresh"]:
            return "Mẻ", area, perimeter
        elif circularity >= PARAMS["pore_circularity_thresh"]:
            return "Lỗ khí", area, perimeter
            
    return None, area, perimeter

def analyze_image_cv(image_pil):
    """
    Phân tích ảnh sử dụng các phương pháp Computer Vision (CV).
    
    Args:
        image_pil (PIL.Image): Ảnh đầu vào.

    Returns:
        tuple: Ảnh đã được đánh dấu và một dictionary chứa số lượng từng loại lỗi.
    """
    img_cv = np.array(image_pil.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Tiền xử lý ảnh để tìm các vùng khả nghi
    mask = preprocess_blackhat(gray)

    # Tìm các đường viền (contours) từ mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    img_area = float(h * w)
    
    counts = {"Nứt": 0, "Mẻ": 0, "Lỗ khí": 0}
    img_annotated = img_cv.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Bỏ qua các nhiễu quá nhỏ hoặc các vùng quá lớn
        if area < PARAMS["min_area_ratio"] * img_area or area > PARAMS["max_area_ratio"] * img_area:
            continue

        defect_type, _, _ = classify_contour(cnt, img_area, gray.shape)
        
        if defect_type:
            counts[defect_type] += 1
            
            # Chọn màu để vẽ lên ảnh
            color = (0, 0, 255) # Đỏ cho Nứt
            if defect_type == "Mẻ":
                color = (0, 255, 0) # Xanh lá cho Mẻ
            elif defect_type == "Lỗ khí":
                color = (255, 0, 0) # Xanh dương cho Lỗ khí

            # Vẽ đường viền và ghi nhãn
            cv2.drawContours(img_annotated, [cnt], -1, color, 3)
            x, y, _, _ = cv2.boundingRect(cnt)
            cv2.putText(img_annotated, defect_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Chuyển đổi ảnh openCV (BGR) ngược lại thành PIL (RGB) để hiển thị
    img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(img_annotated_rgb)

    return annotated_pil, counts

# --- Giao diện ứng dụng Streamlit ---
def main():
    st.set_page_config(layout="wide", page_title="Phân Tích Lỗi Sản Phẩm (CV)")
    st.title("Hệ thống Phân tích Lỗi bằng Xử lý ảnh")
    
    st.info("Đây là phiên bản sử dụng các thuật toán xử lý ảnh truyền thống (OpenCV), tương tự code của bạn bạn.")

    # --- File uploader ---
    uploaded_file = st.file_uploader(
        "Tải lên ảnh sản phẩm cần phân tích", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh Gốc")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Kết quả Phân tích")
            with st.spinner('Đang phân tích...'):
                annotated_image, defect_counts = analyze_image_cv(image)
                
                total_defects = sum(defect_counts.values())

                if total_defects > 0:
                    st.image(annotated_image, caption="Các lỗi đã được phát hiện.", use_column_width=True)
                    st.warning(f"**Tổng số lỗi phát hiện: {total_defects}**")
                    
                    st.subheader("Chi tiết các loại lỗi:")
                    for defect, count in defect_counts.items():
                        if count > 0:
                            st.write(f"- **{defect}**: {count} vùng")
                else:
                    st.image(image, use_column_width=True)
                    st.success("**Sản phẩm không có lỗi.**")

if __name__ == "__main__":
    main()
