import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2 # Thư viện xử lý ảnh và video

# Cấu hình trang
st.set_page_config(page_title="AI Image & Video Classifier", layout="centered")

st.title("🚀 Ứng dụng Nhận diện Ảnh & Video AI")
st.write("Phân tích hình ảnh và video để dự đoán đối tượng.")

# --- PHẦN XỬ LÝ ĐƯỜNG DẪN MÔ HÌNH ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "MobileNetV2.keras")

@st.cache_resource
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Lỗi: Không tìm thấy file mô hình tại đường dẫn: `{MODEL_PATH}`")
        st.info("Mẹo: Kiểm tra thư mục 'model' và tên file 'MobileNetV2.keras' trên GitHub.")
        return None
    try:
        model = tf.keras.model.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Không thể load mô hình: {e}")
        return None

# Load mô hình
model = load_model_safe()
if model is None:
    st.stop() # Dừng ứng dụng nếu mô hình không load được

# --- TIỀN XỬ LÝ ẢNH CHUNG ---
def preprocess_image_for_model(image_array, target_size=(224, 224)):
    """Tiền xử lý mảng ảnh cho MobileNetV2"""
    img_pil = Image.fromarray(image_array) # Chuyển numpy array về PIL Image
    img_resized = img_pil.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # Thêm batch dimension
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# --- KHU VỰC UPLOAD FILE ---
uploaded_file = st.file_uploader(
    "Tải lên ảnh (jpg, png) hoặc video (mp4, mov, avi)",
    type=["jpg", "png", "jpeg", "mp4", "mov", "avi"]
)

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0] # Lấy "image" hoặc "video"
    
    if file_type == "image":
        st.subheader("Phân tích ảnh:")
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Ảnh bạn đã tải lên", use_container_width=True)
        
        # Tiền xử lý và dự đoán
        img_array_np = np.array(img) # Chuyển PIL Image sang numpy array
        processed_img = preprocess_image_for_model(img_array_np)
        
        if st.button("🔍 Dự đoán ảnh"):
            with st.spinner('Đang phân tích ảnh...'):
                prediction = model.predict(processed_img)
                class_idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                st.divider()
                st.subheader("Kết quả dự đoán:")
                st.success(f"**Nhãn dự đoán:** {class_idx}")
                st.info(f"**Độ tin cậy:** {confidence:.2f}%")

    elif file_type == "video":
        st.subheader("Phân tích video:")
        st.video(uploaded_file) # Hiển thị video lên web
        
        # Lưu video tạm thời để OpenCV có thể đọc
        t_file = tempfile.NamedTemporaryFile(delete=False) 
        t_file.write(uploaded_file.read())
        
        if st.button("▶️ Bắt đầu phân tích video (từng khung hình)"):
            with st.spinner("Đang phân tích video... (Quá trình này có thể mất thời gian tùy độ dài video)"):
                cap = cv2.VideoCapture(t_file.name)
                
                predictions_list = []
                frame_count = 0
                
                # Tạo placeholder để cập nhật kết quả liên tục
                prediction_text = st.empty()
                progress_bar = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Chuyển đổi màu từ BGR (OpenCV) sang RGB (TensorFlow/PIL)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Tiền xử lý và dự đoán từng khung hình
                    processed_frame = preprocess_image_for_model(frame_rgb)
                    prediction = model.predict(processed_frame, verbose=0) # verbose=0 để tránh in log nhiều
                    
                    class_idx = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    predictions_list.append((class_idx, confidence))
                    frame_count += 1
                    
                    # Cập nhật thanh tiến trình và kết quả dự đoán
                    if frame_count % 10 == 0: # Cập nhật mỗi 10 frame để không bị quá tải
                        current_pred_idx, current_pred_conf = predictions_list[-1]
                        prediction_text.info(f"Đang xử lý khung hình {frame_count}... Dự đoán hiện tại: Nhãn **{current_pred_idx}** (Độ tin cậy: {current_pred_conf*100:.2f}%)")
                        progress_bar.progress(min(int(frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100), 100))
                
                cap.release()
                os.unlink(t_file.name) # Xóa file tạm thời
                
                st.divider()
                if predictions_list:
                    # Phân tích kết quả tổng thể (ví dụ: nhãn xuất hiện nhiều nhất)
                    from collections import Counter
                    most_common_pred = Counter([p[0] for p in predictions_list]).most_common(1)[0]
                    st.success(f"Phân tích video hoàn tất! Nhãn xuất hiện nhiều nhất: **{most_common_pred[0]}** (số lần: {most_common_pred[1]})")
                    st.info("Để phân tích chi tiết hơn (nhãn thay đổi theo thời gian), bạn cần lưu trữ và hiển thị kết quả phức tạp hơn.")
                else:
                    st.warning("Không có khung hình nào được phân tích từ video.")

# --- CHÚ THÍCH DƯỚI TRANG ---
st.caption("Lưu ý: Nếu kết quả ra nhãn số, bạn cần tạo danh sách tên nhãn để hiển thị chữ.")
