import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Ứng dụng Nhận diện Ảnh AI")

# Đường dẫn đến mô hình duy nhất của bạn
MODEL_PATH = "models/MobileNetV2.keras"

@st.cache_resource
def load_model():
    # compile=False giúp tránh lỗi khác biệt phiên bản thư viện
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

file = st.file_uploader("Tải ảnh lên để AI dự đoán", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300, caption="Ảnh bạn vừa tải lên")
    
    # Tiền xử lý ảnh (MobileNetV2 dùng size 224x224)
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Chuẩn hóa theo chuẩn MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    if st.button("Bấm để dự đoán"):
        pred = model.predict(img_array)
        class_idx = np.argmax(pred)
        confidence = np.max(pred) * 100
        
        st.success(f"Kết quả: Nhãn số {class_idx}")
        st.info(f"Độ tin cậy: {confidence:.2f}%")
