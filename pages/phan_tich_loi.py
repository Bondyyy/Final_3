import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.analysis import analyze_defect_types_pro, segment_image_hybrid

st.set_page_config(
    page_title="Phân Tích Chi Tiết Lỗi",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-title'>🔍 Phân Tích và Khoanh Vùng Lỗi</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Tải lên một ảnh sản phẩm bị lỗi để phân tích...",
    type=["jpg", "jpeg", "png"],
    key="analysis_uploader"
)

if uploaded_file is not None:
    # Đọc ảnh từ file đã upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # --- Cột hiển thị ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Phân tích loại lỗi")
        with st.spinner("Đang tìm và phân loại lỗi..."):
            try:
                num_defects, defect_types, img_out = analyze_defect_types_pro(img_cv.copy())

                st.image(img_out, channels="BGR", caption=f"Phát hiện {num_defects} vùng lỗi.")

                if num_defects > 0:
                    st.success(f"**Số lỗi phát hiện:** {num_defects}")
                    st.write("**Các loại lỗi có thể có:**")
                    # Đếm số lượng mỗi loại lỗi
                    defect_counts = {t: defect_types.count(t) for t in set(defect_types)}
                    for dtype, count in defect_counts.items():
                        st.markdown(f"- **{dtype}:** {count} vùng")
                else:
                    st.info("Không phát hiện được vùng lỗi rõ ràng bằng phương pháp này.")

            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi phân tích: {e}")

    # with col2:
    #     st.subheader("Phân vùng ảnh (Segmentation)")
    #     with st.spinner("Đang thực hiện phân vùng ảnh..."):
    #         try:
    #             # Resize để xử lý nhanh hơn
    #             h, w, _ = img_cv.shape
    #             img_resized = cv2.resize(img_cv, (256, int(256 * h/w)))

    #             segmented_img = segment_image_hybrid(img_resized)

    #             # Hiển thị
    #             display_col1, display_col2 = st.columns(2)
    #             with display_col1:
    #                 st.image(img_resized, channels="BGR", caption="Ảnh gốc (resized)")
    #             with display_col2:
    #                 st.image(segmented_img, channels="BGR", caption="Ảnh đã phân vùng")

    #             st.info(
    #             """
    #             **Giải thích:**
    #             - Phương pháp này sử dụng thuật toán gom cụm (DBSCAN + Fuzzy C-Means) để nhóm các pixel có màu sắc tương tự nhau.
    #             - Các vùng có màu khác biệt (được tô màu ngẫu nhiên) có thể là các vùng lỗi hoặc các vùng có đặc điểm bề mặt khác thường.
    #             """
    #             )

    #         except Exception as e:
    #             st.error(f"Đã xảy ra lỗi khi phân vùng: {e}")

else:
    st.info("Vui lòng tải lên một ảnh để bắt đầu phân tích.")