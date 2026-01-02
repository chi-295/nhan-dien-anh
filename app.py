import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("Phần mềm nhận diện 5 mô hình AI")

# Danh sách mô hình
MODELS = {
    "MobileNetV2": "models/MobileNetV2.keras",
    "InceptionV3": "models/InceptionV3.keras",
    "VGG16": "models/VGG16.keras",
    "ResNet50": "models/ResNet50.keras",
    "EfficientNetB0": "models/EfficientNetB0.keras"
}

@st.cache_resource
def load_model_safe(name):
    path = MODELS[name]
    try:
        # Chìa khóa: compile=False để không cần quan tâm phiên bản optimizer
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Lỗi load {name}: {e}")
        return None

selected = st.selectbox("Chọn mô hình:", list(MODELS.keys()))
model = load_model_safe(selected)

file = st.file_uploader("Chọn ảnh từ máy tính", type=["jpg", "png"])

if file and model:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300)
    
    # Tiền xử lý
    size = (299, 299) if selected == "InceptionV3" else (224, 224)
    img_resized = img.resize(size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Dự đoán ngay"):
        pred = model.predict(img_array)
        st.success(f"Kết quả dự đoán nhãn số: {np.argmax(pred)}")