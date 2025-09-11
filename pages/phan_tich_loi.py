import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from collections import Counter
import io

# ==============================================================================
# PHẦN 1: CÁC HÀM XỬ LÝ ẢNH (COMPUTER VISION - CV)
# Giữ nguyên từ file phan_tich_loi.py gốc của bạn
# ==============================================================================

PARAMS = {
    "min_area_ratio": 0.0001,
    "max_area_ratio": 0.1,
    "crack_aspect_ratio_thresh": 4.0,
    "chip_circularity_thresh": 0.5,
    "pore_circularity_thresh": 0.7,
}

def preprocess_blackhat(gray_image):
    """Sử dụng Black Hat để làm nổi bật các vùng tối trên nền sáng."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)
    return mask

def classify_contour(contour, image_area):
    """Phân loại một contour thành Nứt, Mẻ, hoặc Lỗ khí."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    if aspect_ratio > PARAMS["crack_aspect_ratio_thresh"]:
        return "Nứt"

    if perimeter > 0:
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < PARAMS["chip_circularity_thresh"]:
            return "Mẻ"
        elif circularity >= PARAMS["pore_circularity_thresh"]:
            return "Lỗ khí"
            
    return None

def analyze_image_cv(image_pil):
    """
    Phân tích chi tiết các loại lỗi trên ảnh đã được xác định là có lỗi.
    """
    img_cv = np.array(image_pil.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    mask = preprocess_blackhat(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(gray.shape[0] * gray.shape[1])
    
    img_annotated = img_cv.copy()
    found_defects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (PARAMS["min_area_ratio"] * img_area < area < PARAMS["max_area_ratio"] * img_area):
            continue

        defect_type = classify_contour(cnt, img_area)
        
        if defect_type:
            found_defects.append(defect_type)
            color = {"Nứt": (0, 0, 255), "Mẻ": (0, 255, 0), "Lỗ khí": (255, 0, 0)}.get(defect_type)
            cv2.drawContours(img_annotated, [cnt], -1, color, 3)
            x, y, _, _ = cv2.boundingRect(cnt)
            cv2.putText(img_annotated, defect_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    annotated_pil = Image.fromarray(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
    defect_counts = Counter(found_defects)
    return annotated_pil, defect_counts

# ==============================================================================
# PHẦN 2: CÁC HÀM LIÊN QUAN ĐẾN MÔ HÌNH AI (PYTORCH)
# Tích hợp từ file model.py và app.py của bạn
# ==============================================================================

def get_model(num_classes=2, dropout_rate=0.3):
    """Tạo một mô hình EfficientNetV2-S."""
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model

@st.cache_resource
def load_pytorch_model(model_path='final_model.pth'):
    """Tải và cache mô hình PyTorch để không phải tải lại mỗi lần."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy tệp '{model_path}'. Vui lòng đảm bảo tệp mô hình nằm đúng vị trí.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi tải mô hình: {e}")
        return None

def predict_image_class(model, image_pil):
    """Dự đoán một ảnh là Tốt (OK) hay có Lỗi (Defective)."""
    if model is None: return None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = image_pil.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        # Chú ý: class_names cần khớp với lúc bạn huấn luyện model
        # Giả định: 0 = Lỗi (Defective), 1 = Tốt (OK)
        class_names = {0: 'Lỗi', 1: 'Tốt'}
        predicted_class = class_names.get(predicted_idx.item())
        return predicted_class, confidence.item()

# ==============================================================================
# PHẦN 3: GIAO DIỆN ỨNG DỤNG STREAMLIT (ĐÃ CẬP NHẬT LOGIC)
# ==============================================================================

def main():
    st.set_page_config(layout="wide", page_title="Phân Tích Lỗi Sản Phẩm")
    st.title("Hệ Thống Kiểm Tra Chất Lượng Sản Phẩm (AI + CV)")
    
    st.info("Quy trình: **AI phân loại (Tốt/Lỗi) ➔ Nếu Lỗi, CV phân tích chi tiết.**")

    # Tải mô hình AI
    with st.spinner("Đang tải mô hình AI..."):
        model = load_pytorch_model()

    uploaded_file = st.file_uploader(
        "Tải lên ảnh sản phẩm cần phân tích", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh Gốc")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Kết quả Phân tích")
            with st.spinner('Bước 1: AI đang phân loại...'):
                # BƯỚC 1: DÙNG AI ĐỂ PHÂN LOẠI TRƯỚC
                predicted_class, confidence = predict_image_class(model, image)
                confidence_percent = confidence * 100

            if predicted_class == 'Tốt':
                st.success(f"**Kết quả AI: TỐT** (Độ tin cậy: {confidence_percent:.2f}%)")
                st.image(image, caption="Sản phẩm không phát hiện lỗi.", use_column_width=True)
            
            elif predicted_class == 'Lỗi':
                st.error(f"**Kết quả AI: LỖI** (Độ tin cậy: {confidence_percent:.2f}%)")
                with st.spinner('Bước 2: CV đang phân tích chi tiết loại lỗi...'):
                    # BƯỚC 2: NẾU LỖI, MỚI DÙNG CV PHÂN TÍCH
                    annotated_image, defect_counts = analyze_image_cv(image)
                    total_defects = sum(defect_counts.values())

                    if total_defects > 0:
                        st.image(annotated_image, caption="Các loại lỗi đã được khoanh vùng.", use_column_width=True)
                        st.warning(f"**Tổng số vùng lỗi phát hiện bởi CV: {total_defects}**")
                        st.subheader("Chi tiết các loại lỗi:")
                        for defect, count in defect_counts.items():
                            st.write(f"- **{defect}**: {count} vùng")
                    else:
                        st.info("AI phát hiện có lỗi, nhưng các thuật toán CV không xác định được loại lỗi cụ thể. Cần kiểm tra thủ công.")
            else:
                st.warning("Không thể phân loại ảnh này. Vui lòng thử ảnh khác.")

if __name__ == "__main__":
    main()