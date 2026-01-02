import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import tempfile

# Cấu hình giao diện
st.set_page_config(page_title="AI Vision Pro", page_icon="🤖", layout="centered")

# CSS tùy chỉnh để làm đẹp giao diện
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; border-radius: 5px; background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; border: 1px solid #e6e9ef; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Trí tuệ Nhân tạo Nhận diện Ảnh & Video")
st.info("Hệ thống đang chạy trên môi trường Python 3.11 & TensorFlow 2.15")

# --- QUẢN LÝ MÔ HÌNH ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "MobileNetV2.keras")

@st.cache_resource
def load_model_ai():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy file mô hình tại thư mục gốc!")
        return None
    try:
        # Load mô hình bằng tf.keras để tránh lỗi cấu trúc Layer trên môi trường mới
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Lỗi nạp mô hình: {str(e)}")
        return None

model = load_model_ai()

# --- XỬ LÝ DỮ LIỆU ---
def preprocess(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

if model:
    tab1, tab2 = st.tabs(["🖼️ Nhận diện Ảnh", "🎥 Nhận diện Video"])

    with tab1:
        file_img = st.file_uploader("Chọn ảnh (JPG, PNG)...", type=["jpg", "png", "jpeg"])
        if file_img:
            img = Image.open(file_img).convert('RGB')
            st.image(img, use_container_width=True)
            if st.button("🚀 Phân tích Ảnh"):
                processed = preprocess(img)
                preds = model.predict(processed)
                label = np.argmax(preds)
                conf = np.max(preds) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Nhãn dự đoán", f"Số {label}")
                col2.metric("Độ tin cậy", f"{conf:.2f}%")

    with tab2:
        file_vid = st.file_uploader("Chọn video (MP4, MOV)...", type=["mp4", "mov"])
        if file_vid:
            st.video(file_vid)
            if st.button("▶️ Phân tích Video"):
                with st.spinner("Đang trích xuất khung hình..."):
                    t_file = tempfile.NamedTemporaryFile(delete=False)
                    t_file.write(file_vid.read())
                    cap = cv2.VideoCapture(t_file.name)
                    cap.set(cv2.CAP_PROP_POS_MSEC, 1000) # Lấy dữ liệu tại giây thứ 1
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed = preprocess(Image.fromarray(frame_rgb))
                        preds = model.predict(processed)
                        st.success(f"Kết quả Video: Nhãn {np.argmax(preds)}")
                    cap.release()
                    os.unlink(t_file.name)

st.divider()
st.caption("Thiết kế bởi Gemini AI - 2026")
