import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import tempfile

# Cấu hình trang
st.set_page_config(page_title="AI Recognition", layout="centered")
st.title("🚀 Phần mềm nhận diện Ảnh & Video AI")

# --- SỬA ĐƯỜNG DẪN: File nằm ngay thư mục gốc ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "MobileNetV2.keras") # Không còn chữ 'model/' ở trước

@st.cache_resource
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy file: {MODEL_PATH}")
        st.write("Các file hiện có trên GitHub của bạn:", os.listdir(BASE_DIR))
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {e}")
        return None

model = load_model_safe()

# --- GIAO DIỆN CHƯƠNG TRÌNH ---
uploaded_file = st.file_uploader("Tải Ảnh hoặc Video vào đây", type=["jpg", "png", "jpeg", "mp4", "mov"])

if uploaded_file and model:
    # Kiểm tra định dạng file
    is_video = uploaded_file.type.startswith('video')
    
    if not is_video:
        # XỬ LÝ ẢNH
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, width=300, caption="Ảnh đã tải lên")
        
        if st.button("🔍 Dự đoán Ảnh"):
            # Tiền xử lý cho MobileNetV2 (224x224)
            img_input = np.array(img.resize((224, 224)))
            img_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_input, axis=0))
            
            pred = model.predict(img_input)
            st.success(f"Kết quả nhãn: **{np.argmax(pred)}** (Tin cậy: {np.max(pred)*100:.2f}%)")

    else:
        # XỬ LÝ VIDEO
        st.video(uploaded_file)
        if st.button("▶️ Phân tích Video"):
            with st.spinner("Đang xử lý..."):
                t_file = tempfile.NamedTemporaryFile(delete=False) 
                t_file.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(t_file.name)
                cap.set(cv2.CAP_PROP_POS_MSEC, 1000) # Lấy khung hình ở giây thứ 1
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_input = np.array(Image.fromarray(frame_rgb).resize((224, 224)))
                    img_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_input, axis=0))
                    
                    pred = model.predict(img_input)
                    st.success(f"Kết quả video (khung hình chính): **{np.argmax(pred)}**")
                else:
                    st.error("Không đọc được video.")
                
                cap.release()
                os.unlink(t_file.name)
