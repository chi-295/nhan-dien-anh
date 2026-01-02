import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import tempfile

# Giao diện
st.set_page_config(page_title="AI Classifier", layout="centered")
st.title("🤖 Nhận diện Ảnh & Video AI")

# Đường dẫn file mô hình ngay tại thư mục gốc
MODEL_PATH = "MobileNetV2.keras"

@st.cache_resource
def load_model_fixed():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Không tìm thấy file {MODEL_PATH} trên GitHub của bạn!")
        return None
    try:
        # Load mô hình và không biên dịch (compile=False) để tránh lỗi phiên bản
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Lỗi load mô hình: {e}")
        return None

model = load_model_fixed()

# Chức năng dự đoán chung
def predict_logic(img_pil):
    # MobileNetV2 chuẩn: 224x224
    img_resized = img_pil.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_final = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    prediction = model.predict(img_final)
    return np.argmax(prediction), np.max(prediction) * 100

# Giao diện tải file
file = st.file_uploader("Tải lên Ảnh hoặc Video", type=["jpg", "png", "jpeg", "mp4", "mov"])

if file and model:
    is_video = file.type.startswith('video')
    
    if not is_video:
        # XỬ LÝ ẢNH
        image = Image.open(file).convert('RGB')
        st.image(image, use_container_width=True)
        if st.button("🔍 Dự đoán Ảnh"):
            label, conf = predict_logic(image)
            st.success(f"Kết quả: Nhãn {label} (Độ tin cậy: {conf:.2f}%)")
    else:
        # XỬ LÝ VIDEO
        st.video(file)
        if st.button("▶️ Dự đoán Video"):
            with st.spinner("Đang phân tích khung hình..."):
                t_file = tempfile.NamedTemporaryFile(delete=False)
                t_file.write(file.read())
                cap = cv2.VideoCapture(t_file.name)
                cap.set(cv2.CAP_PROP_POS_MSEC, 1000) # Lấy tại giây thứ 1
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    label, conf = predict_logic(Image.fromarray(frame_rgb))
                    st.success(f"Kết quả Video: Nhãn {label}")
                cap.release()
                os.unlink(t_file.name)
