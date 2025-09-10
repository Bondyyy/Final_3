import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from utils.model import get_model, load_model
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
    /* General layout improvements */
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .st-emotion-cache-1v0mbdj {
        max-width: 95%;
    }
    /* Title styles */
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
    /* Result text styles */
    .result-text {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        padding: 0.5rem;
        border-radius: 10px;
        margin-top: 10px;
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
    """Tải và cache mô hình PyTorch."""
    model = get_model()
    try:
        model_path = 'final_model.pth'
        model = load_model(model, model_path)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy tệp 'final_model.pth'. Vui lòng đảm bảo tệp mô hình nằm trong thư mục gốc của dự án.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không mong muốn khi tải mô hình: {e}")
        return None

def predict(model, image_data):
    """Dự đoán một ảnh là Tốt (OK) hay có Lỗi (Defective)."""
    if model is None:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Các bước tiền xử lý ảnh phải giống với lúc huấn luyện mô hình
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), # Chuyển ảnh xám thành 3 kênh để hợp với mô hình pre-trained
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = predicted_idx.item()

        return predicted_class, confidence.item()
    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán: {e}")
        return None, None


# --- Giao diện ứng dụng ---

# Tiêu đề
st.markdown("<h1 class='main-title'>🏭 Ứng dụng Phát hiện Lỗi Sản phẩm Đúc</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Tải lên hình ảnh sản phẩm để kiểm tra chất lượng bằng AI</p>", unsafe_allow_html=True)

# Tải mô hình với spinner
with st.spinner("Đang tải mô hình, vui lòng chờ..."):
    model = load_pytorch_model()

# Widget tải file
uploaded_files = st.file_uploader(
    "Chọn một hoặc nhiều ảnh sản phẩm...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.divider()

# Xử lý và hiển thị kết quả cho từng file
if uploaded_files and model is not None:
    st.header("💡 Kết quả Phân loại")
    
    # Thiết lập layout dạng lưới với 3 cột
    num_columns = 3
    cols = st.columns(num_columns)
    
    # Lặp qua từng tệp đã tải lên
    for index, uploaded_file in enumerate(uploaded_files):
        image_data = uploaded_file.getvalue()
        
        # Đặt kết quả vào cột tương ứng (col 0, 1, 2, rồi quay lại 0, ...)
        col = cols[index % num_columns]
        
        with col:
            # Dùng container để đóng khung cho mỗi kết quả
            with st.container(border=True):
                st.image(image_data, caption=f"{uploaded_file.name}",width=True)
                
                # Dự đoán và lấy kết quả cho từng ảnh
                predicted_class, confidence = predict(model, image_data)
                
                if predicted_class is not None:
                    class_names = {0: 'Lỗi', 1: 'Tốt'} # Giả định 0 là Lỗi, 1 là Tốt
                    result = class_names.get(predicted_class, "Không xác định")
                    confidence_percent = confidence * 100

                    if result == 'Tốt':
                        st.markdown(f"<div class='result-text result-ok'>✔️ TỐT ({confidence_percent:.1f}%)</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='result-text result-def'>❌ LỖI ({confidence_percent:.1f}%)</div>", unsafe_allow_html=True)
                else:
                    st.warning("Không thể phân tích ảnh này.")

elif not uploaded_files:
    st.info("Vui lòng tải ảnh lên để bắt đầu phân loại.")

# --- Sidebar ---
st.sidebar.title("📖 Hướng dẫn")
st.sidebar.info(
    """
    1. **Tải ảnh lên:** Nhấn vào 'Browse files' và chọn một hoặc nhiều ảnh sản phẩm đúc bạn muốn kiểm tra.
    2. **Xem kết quả:** Mô hình sẽ tự động phân loại từng ảnh là 'Tốt' hoặc 'Lỗi' và hiển thị kết quả ngay bên dưới.
    3. **Phân tích sâu:** Nếu sản phẩm bị lỗi, bạn có thể chuyển qua trang **'🔍 Phân Tích Loại Lỗi'** (nếu có) để xem chi tiết.
    """
)
st.sidebar.title("Về dự án")
st.sidebar.success(
    """
    Dự án này sử dụng mô hình học sâu **EfficientNetV2-S** để phân loại sản phẩm.
    Đồng thời, áp dụng các thuật toán xử lý ảnh để xác định các loại lỗi cụ thể như 'Nứt', 'Mẻ', 'Lỗ khí'.
    """
)
