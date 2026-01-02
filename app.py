import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import tempfile

# Cấu hình giao diện chuẩn hiện đại
st.set_page_config(
    page_title="AI Vision Pro", 
    page_icon="🤖", 
    layout="centered"
)

# Custom CSS để giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Hệ thống Nhận diện Ảnh & Video AI")
st.write("Giải pháp phân tích hình ảnh dựa trên kiến trúc **MobileNetV2**.")

# --- XỬ LÝ MÔ HÌNH ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "MobileNetV2.keras")

@st.cache_resource
def load_model_optimized():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy file: {MODEL_PATH}")
        return None
    try:
        # Sử dụng tf.keras để load nhằm khắc phục lỗi xung đột Layer trên Keras 3
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Lỗi cấu trúc mô hình: {e}")
        st.info("💡 Mẹo: Hệ thống đang thử xử lý xung đột phiên bản Keras. Hãy đảm bảo bạn đã Reboot App sau khi sửa requirements.txt.")
        return None

model = load_model_optimized()

# --- KHU VỰC CHỨC NĂNG ---
if model:
    tab1, tab2 = st.tabs(["📸 Phân tích Ảnh", "🎥 Phân tích Video"])

    with tab1:
        uploaded_img = st.file_uploader("Kéo thả ảnh vào đây", type=["jpg", "png", "jpeg"], key="img")
        if uploaded_img:
            col1, col2 = st.columns([1, 1])
            with col1:
                img = Image.open(uploaded_img).convert('RGB')
                st.image(img, caption="Ảnh gốc", use_container_width=True)
            
            with col2:
                if st.button("🚀 Bắt đầu dự đoán", key="btn_img"):
                    with st.spinner("Đang phân tích..."):
                        # Tiền xử lý chuẩn MobileNetV2
                        img_input = np.array(img.resize((224, 224)))
                        img_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_input, axis=0))
                        
                        preds = model.predict(img_input)
                        label = np.argmax(preds)
                        confidence = np.max(preds) * 100
                        
                        st.metric("Nhãn dự đoán", f"Số {label}")
                        st.metric("Độ tin cậy", f"{confidence:.2f}%")
                        if confidence > 80: st.balloons()

    with tab2:
        uploaded_vid = st.file_uploader("Tải video lên", type=["mp4", "mov", "avi"], key="vid")
        if uploaded_vid:
            st.video(uploaded_vid)
            if st.button("▶️ Phân tích Video", key="btn_vid"):
                with st.spinner("Đang trích xuất khung hình..."):
                    t_file = tempfile.NamedTemporaryFile(delete=False)
                    t_file.write(uploaded_vid.read())
                    
                    cap = cv2.VideoCapture(t_file.name)
                    # Lấy khung hình tại 50% thời lượng video
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                    
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_input = np.array(Image.fromarray(frame_rgb).resize((224, 224)))
                        img_input = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_input, axis=0))
                        
                        preds = model.predict(img_input)
                        st.success(f"### Dự đoán: Nhãn {np.argmax(preds)}")
                        st.progress(float(np.max(preds)))
                    else:
                        st.error("Không thể xử lý video.")
                    cap.release()
                    os.unlink(t_file.name)

# --- CHÂN TRANG ---
st.divider()
st.caption("© 2026 AI Vision Pro - Hệ thống vận hành trên nền tảng TensorFlow & Streamlit.")
