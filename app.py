import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from utils.model import get_model, load_model
import numpy as np
import io

# Cấu hình trang
st.set_page_config(
    page_title="Phân loại Lỗi Sản Phẩm Đúc",
    page_icon="🏭",
    layout="wide",
)

# --- CSS Tùy chỉnh ---
st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .st-emotion-cache-1v0mbdj {
        max-width: 95%;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #4682B4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
    }
    .result-ok {
        color: #2E8B57;
        background-color: #D4EDDA;
    }
    .result-def {
        color: #A52A2A;
        background-color: #F8D7DA;
    }
</style>
""", unsafe_allow_html=True)


# --- Hàm xử lý ---
@st.cache_resource
def load_pytorch_model():

    model = get_model()
    try:
       
        model_path = 'final_model.pth'
        model = load_model(model, model_path)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy tệp 'final_model.pth'. Vui lòng tải tệp mô hình và đặt vào thư mục dự án.")
        return None

def predict(model, image_data):
    """Dự đoán ảnh là OK hay Lỗi."""
    if model is None:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = predicted_idx.item()

    return predicted_class, confidence.item()

# --- Giao diện ứng dụng ---

# Tiêu đề
st.markdown("<h1 class='main-title'>🏭 Ứng dụng Phát hiện Lỗi Sản phẩm Đúc</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Tải lên hình ảnh sản phẩm để kiểm tra chất lượng</p>", unsafe_allow_html=True)

# Tải mô hình
model = load_pytorch_model()

# Cột để hiển thị
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("🖼️ Tải ảnh lên")
    uploaded_files = st.file_uploader(
        "Chọn một ảnh sản phẩm...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True  
    )

    if uploaded_files and model is not None:
        st.header("💡 Kết quả Phân loại")

    # Tạo 3 cột để hiển thị kết quả 
    cols = st.columns(3)
    col_index = 0

    # Lặp qua từng file ảnh đã được tải lên
    for uploaded_file in uploaded_files:
        image_data = uploaded_file.getvalue()

        # Đặt kết quả của mỗi ảnh vào một cột riêng
        with cols[col_index]:
            st.image(image_data, caption=f"Ảnh: {uploaded_file.name}", width=True)

            # Phân loại và hiển thị kết quả
            predicted_class, confidence = predict(model, image_data)
            # ... (code hiển thị kết quả 'Tốt' hoặc 'Lỗi') ...

        # Chuyển sang cột tiếp theo cho ảnh kế tiếp
        col_index = (col_index + 1) % len(cols)

with col2:
    st.header("💡 Kết quả Phân loại")
    if uploaded_files is not None and model is not None:
        with st.spinner("Đang phân tích..."):
            predicted_class, confidence = predict(model, image_data)

            class_names = {0: 'Lỗi', 1: 'Tốt'}
            result = class_names.get(predicted_class, "Không xác định")
            confidence_percent = confidence * 100

            if result == 'Tốt':
                st.markdown(f"<div class='result-text result-ok'>✔️ Sản phẩm TỐT</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-text result-def'>❌ Sản phẩm có LỖI</div>", unsafe_allow_html=True)

            st.write("")
            st.metric(label="Độ tin cậy", value=f"{confidence_percent:.2f}%")

            if result == 'Lỗi':
                 st.info("Để xem chi tiết và khoanh vùng lỗi, hãy chuyển qua trang **'🔍 Phân Tích Loại Lỗi'**.")

    else:
        st.info("Vui lòng tải ảnh lên để bắt đầu phân loại.")

# Hướng dẫn
st.sidebar.title("📖 Hướng dẫn")
st.sidebar.info(
    """
    1. **Tải ảnh lên:** Nhấn vào 'Browse files' và chọn ảnh sản phẩm đúc bạn muốn kiểm tra.
    2. **Xem kết quả:** Mô hình sẽ tự động phân loại ảnh là 'Tốt' hoặc 'Lỗi'.
    3. **Phân tích sâu:** Nếu sản phẩm bị lỗi,  có thể vào trang 'Phân Tích Loại Lỗi' từ thanh điều hướng bên cạnh để xem vị trí và loại lỗi chi tiết.
    """
)
st.sidebar.title("Về dự án")
st.sidebar.success(
    """
    Dự án này sử dụng mô hình học sâu **EfficientNetV2-S** để phân loại sản phẩm.
    Đồng thời, áp dụng các thuật toán xử lý ảnh để xác định các loại lỗi cụ thể như 'Nứt', 'Mẻ', 'Lỗ khí'.
    """
)