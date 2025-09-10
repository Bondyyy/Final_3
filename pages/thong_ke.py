import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Thông Kê Huấn Luyện",
    page_icon="📊",
    layout="wide",
)

st.markdown("<h1 style='text-align: center; color: #007BFF;'>📊 Báo Cáo và Thống Kê Huấn Luyện Mô Hình</h1>", unsafe_allow_html=True)
st.write("")

st.info(
    """
    Phần này hiển thị các thông tin và kết quả được trích xuất từ quá trình huấn luyện mô hình
    trên Google Colab. Các biểu đồ và số liệu này giúp đánh giá hiệu năng của mô hình.
    """
)

# --- Tạo dữ liệu báo cáo (dựa trên output từ notebook) ---
performance_data = {
    "Chỉ số": [
        "Accuracy CV trung bình (Train)",
        "Accuracy CV trung bình (Validation)",
        "**Test Accuracy**",
        "Loss CV trung bình (Train)",
        "Loss CV trung bình (Validation)",
        "Thời gian huấn luyện CV tổng",
        "Thời gian huấn luyện Final Model",
        "Tiêu tốn RAM hệ thống (Colab)",
        "Tiêu tốn GPU Memory (Colab)"
    ],
    "Giá trị": [
        "~99.xx %", # Thay bằng giá trị thực từ notebook của bạn
        "99.69 %",
        "**99.86 %**",
        "~0.01xx",
        "0.0125",
        "~1200 s",
        "~300 s",
        "~30 %",
        "~2.5 GB"
    ]
}

df_performance = pd.DataFrame(performance_data)

# --- Hiển thị ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Biểu đồ Huấn luyện (Cross-Validation)")
    st.write("Các biểu đồ dưới đây thể hiện Loss và Accuracy trung bình qua các Fold trong quá trình K-Fold Cross-Validation.")
    # Bạn cần lưu các biểu đồ từ notebook thành file ảnh (ví dụ loss.png, acc.png)
    # và đặt chúng vào cùng thư mục dự án.
    try:
        st.image("cv_loss_accuracy.png", caption="Loss và Accuracy trung bình (CV)")
    except FileNotFoundError:
        st.warning("Không tìm thấy file 'cv_loss_accuracy.png'. Vui lòng lưu biểu đồ từ notebook và đặt vào thư mục dự án.")

with col2:
    st.subheader("📋 Báo cáo Hiệu suất Tổng thể")
    st.dataframe(df_performance, width='stretch', hide_index=True)
    st.success("Mô hình đạt độ chính xác rất cao trên tập kiểm tra, cho thấy khả năng tổng quát hóa tốt.")

st.subheader("🖼️ Phân bố Dữ liệu")
st.write("Biểu đồ phân bố số lượng ảnh trong tập huấn luyện và kiểm tra.")
try:
    st.image("data_distribution.png", caption="Phân bố ảnh trong dataset")
except FileNotFoundError:
    st.warning("Không tìm thấy file 'data_distribution.png'. Vui lòng lưu biểu đồ từ notebook và đặt vào thư mục dự án.")

st.subheader("CSV Dự đoán trên tập Test")
st.write("Dưới đây là 10 dự đoán đầu tiên trên tập dữ liệu kiểm tra.")

try:
    # Bạn cần có file test_predictions.csv
    df_preds = pd.read_csv("test_predictions.csv")
    st.dataframe(df_preds.head(10), width='stretch')
except FileNotFoundError:
    st.warning("Không tìm thấy file 'test_predictions.csv'. Vui lòng đặt file này vào thư mục dự án.")