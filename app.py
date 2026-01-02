import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import tempfile

# 1. Cấu hình giao diện
st.set_page_config(page_title="AI Vision Pro", layout="centered")

st.markdown("<h2 style='text-align: center;'>🤖 Nhận diện Ảnh & Video AI</h2>", unsafe_allow_html=True)

# 2. Đường dẫn mô hình (File nằm cùng thư mục với app.py)
MODEL_PATH = "MobileNetV2.keras"

@st.cache_resource
def load_model_ai():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy file {MODEL_PATH} trên GitHub!")
        return None
    try:
        # Sử dụng tf.keras để load (Cách an toàn nhất cho bản 2.15)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Lỗi nạp mô hình: {e}")
        return None

model = load_model_ai()

# 3. Giao diện tải file
uploaded_file = st.file_uploader("Tải Ảnh hoặc Video", type=["jpg", "png", "jpeg", "mp4", "mov"])

if uploaded_file and model:
    # Phân loại file
    is_video = uploaded_file.type.startswith('video')

    if not is_video:
        # --- XỬ LÝ ẢNH ---
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_container_width=True)
        
        if st.button("🔍 Bắt đầu phân tích Ảnh"):
            # Tiền xử lý (224x224 cho MobileNetV2)
            img_prep = np.array(img.resize((224, 224)))
            img_prep = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_prep, axis=0))
            
            preds = model.predict(img_prep)
            st.success(f"### Dự đoán: Nhãn {np.argmax(preds)}")
            st.info(f"Độ tin cậy: {np.max(preds)*100:.2f}%")

    else:
        # --- XỬ LÝ VIDEO ---
        st.video(uploaded_file)
        if st.button("▶️ Phân tích Video"):
            with st.spinner("Đang trích xuất khung hình..."):
                t_file = tempfile.NamedTemporaryFile(delete=False)
                t_file.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(t_file.name)
                # Lấy khung hình tại giây đầu tiên
                cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_prep = np.array(Image.fromarray(frame_rgb).resize((224, 224)))
                    img_prep = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_prep, axis=0))
                    
                    preds = model.predict(img_prep)
                    st.success(f"### Dự đoán Video: Nhãn {np.argmax(preds)}")
                else:
                    st.error("Không thể đọc khung hình video.")
                cap.release()
                os.unlink(t_file.name)
