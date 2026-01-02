import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import tempfile

# Cấu hình giao diện
st.set_page_config(page_title="AI Recognition Pro", layout="centered")
st.title("🚀 Ứng dụng nhận diện Ảnh & Video AI")
st.write("Mô hình sử dụng: **MobileNetV2**")

# --- XỬ LÝ MÔ HÌNH ---
BASE_DIR = os.path.dirname(__file__)
# Đã sửa đường dẫn theo tên thư mục 'model' của bạn
MODEL_PATH = os.path.join(BASE_DIR, "model", "MobileNetV2.keras")

@st.cache_resource
def load_model_ai():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy mô hình tại: {MODEL_PATH}")
        return None
    try:
        # Load mô hình và tắt compile để tránh lỗi phiên bản thư viện
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Lỗi khi tải file .keras: {e}")
        return None

model = load_model_ai()

# --- HÀM TIỀN XỬ LÝ ---
def prepare_image(img_pil):
    """Chuyển đổi ảnh về định dạng MobileNetV2 yêu cầu (224x224)"""
    img_resized = img_pil.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# --- GIAO DIỆN TẢI FILE ---
uploaded_file = st.file_uploader("Kéo thả Ảnh hoặc Video vào đây", type=["jpg", "png", "jpeg", "mp4", "mov", "avi"])

if uploaded_file and model:
    # Kiểm tra xem là ảnh hay video
    is_video = uploaded_file.type.startswith('video')

    if not is_video:
        # XỬ LÝ ẢNH
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        
        if st.button("🔍 Dự đoán Ảnh"):
            with st.spinner("Đang phân tích..."):
                processed_img = prepare_image(image)
                preds = model.predict(processed_img)
                label = np.argmax(preds)
                score = np.max(preds) * 100
                
                st.divider()
                st.success(f"### Kết quả: Nhãn {label}")
                st.info(f"Độ tin cậy: {score:.2f}%")
    
    else:
        # XỬ LÝ VIDEO
        st.video(uploaded_file)
        if st.button("▶️ Phân tích Video"):
            with st.spinner("Đang xử lý khung hình chính..."):
                # Tạo file tạm vì OpenCV không đọc trực tiếp được file upload từ Streamlit
                t_file = tempfile.NamedTemporaryFile(delete=False)
                t_file.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(t_file.name)
                # Lấy khung hình ở giữa video để có độ chính xác cao nhất
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                
                ret, frame = cap.read()
                if ret:
                    # Chuyển BGR (OpenCV) sang RGB (PIL)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    
                    processed_frame = prepare_image(img_pil)
                    preds = model.predict(processed_frame)
                    label = np.argmax(preds)
                    score = np.max(preds) * 100
                    
                    st.divider()
                    st.success(f"### Kết quả Video: Nhãn {label}")
                    st.info(f"Độ tin cậy (tại khung hình giữa): {score:.2f}%")
                else:
                    st.error("Không thể đọc được video này.")
                
                cap.release()
                os.unlink(t_file.name) # Xóa file tạm để nhẹ bộ nhớ
