import streamlit as st
import gdown
import joblib
import pandas as pd

# 1. Cấu hình tải model từ Google Drive
file_id = "1LLjj8igfQPTC6ncAv8ybZfJvnAKJRIz8"  # ID từ link bạn vừa chia sẻ
url = f"https://drive.google.com/uc?id={file_id}"
output = "/tmp/randomforest_model.pkl"

@st.cache_resource
def load_model():
    gdown.download(url, output, quiet=False)
    model = joblib.load(output)
    return model

model = load_model()

# 2. Giao diện người dùng
st.set_page_config(page_title="Dự báo giá BĐS", layout="wide")
st.title("🏠 Ứng dụng dự báo giá bất động sản")

dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, step=1.0, value=100.0)
mat_tien = st.number_input("Mặt tiền (m)", min_value=0.5, step=0.5, value=5.0)
duong_vao = st.number_input("Đường vào (m)", min_value=0.5, step=0.5, value=3.0)

loai_bds = st.selectbox("Loại BĐS", ["Nhà phố", "Chung cư", "Đất nền", "Biệt thự"])
giay_to = st.selectbox("Giấy tờ pháp lý", ["Sổ đỏ", "Sổ hồng", "Giấy tờ hợp lệ", "Khác"])
xa_phuong = st.text_input("Xã/Phường", value="Phường 1")
quan_huyen = st.text_input("Quận/Huyện", value="Quận 1")

if st.button("Dự đoán giá"):
    input_df = pd.DataFrame([{
        'Diện tích (m²)': dien_tich,
        'Mặt tiền (m)': mat_tien,
        'Đường vào (m)': duong_vao,
        'Loại BDS': loai_bds,
        'Giấy tờ pháp lý': giay_to,
        'Xã/Phường': xa_phuong,
        'Quận/Huyện': quan_huyen
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Giá dự báo: {prediction:.2f} Tỷ")
