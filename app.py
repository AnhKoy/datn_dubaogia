import streamlit as st
import gdown
import joblib
import pandas as pd

# ==== 1. Tải model .pkl từ Google Drive ====
@st.cache_resource
def load_model():
    file_id = "1LLjj8igfQPTC6ncAv8ybZfJvnAKJRIz8"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "model.pkl"
    gdown.download(url, output, quiet=False)
    model = joblib.load(output)
    return model

model = load_model()

# ==== 2. Đọc dữ liệu CSV từ GitHub ====
@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/AnhKoy/datn_dubaogia/main/bds_dat_clean%20(1).csv"
    df = pd.read_csv(csv_url)
    return df

df_data = load_data()

# ==== 3. Giao diện người dùng ====
st.set_page_config(page_title="Dự báo giá BĐS", layout="wide")
st.title("🏠 Ứng dụng dự báo giá bất động sản")

dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, step=1.0, value=100.0)
mat_tien = st.number_input("Mặt tiền (m)", min_value=0.5, step=0.5, value=5.0)
duong_vao = st.number_input("Đường vào (m)", min_value=0.5, step=0.5, value=3.0)

loai_bds = st.selectbox("Loại BĐS", df_data["Loại BDS"].unique())
giay_to = st.selectbox("Giấy tờ pháp lý", df_data["Giấy tờ pháp lý"].unique())
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
