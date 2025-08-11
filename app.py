import streamlit as st
import pandas as pd
import joblib

# Load mô hình đã train
model = joblib.load("D:\\Downloads\\randomforest_model (1).pkl")

# Load dữ liệu gốc để lấy danh sách giá trị
df = pd.read_csv("E:\\DATA\\DuAn\\BatDongSan\\clean\\bds_dat_clean (1).csv")

st.title("🔍 Dự đoán giá Bất động sản")

st.write("Nhập thông tin bất động sản để dự đoán giá:")

# Nhập dữ liệu số
dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, max_value=2000.0, value=100.0)
mat_tien = st.number_input("Mặt tiền (m)", min_value=0.5, max_value=50.0, value=5.0)
duong_vao = st.number_input("Đường vào (m)", min_value=0.5, max_value=50.0, value=3.0)

# Dropdown tự động từ dữ liệu gốc
loai_bds = st.selectbox("Loại BDS", sorted(df["Loại BDS"].dropna().unique()))
giay_to = st.selectbox("Giấy tờ pháp lý", sorted(df["Giấy tờ pháp lý"].dropna().unique()))
xa_phuong = st.selectbox("Xã/Phường", sorted(df["Xã/Phường"].dropna().unique()))
quan_huyen = st.selectbox("Quận/Huyện", sorted(df["Quận/Huyện"].dropna().unique()))

# Nút dự đoán
if st.button("Dự đoán giá"):
    # Tạo DataFrame giống khi train
    input_df = pd.DataFrame([{
        "Diện tích (m²)": dien_tich,
        "Mặt tiền (m)": mat_tien,
        "Đường vào (m)": duong_vao,
        "Loại BDS": loai_bds,
        "Giấy tờ pháp lý": giay_to,
        "Xã/Phường": xa_phuong,
        "Quận/Huyện": quan_huyen
    }])

    # Dự đoán
    predicted_price = model.predict(input_df)[0]
    st.success(f"💰 Giá dự đoán: {predicted_price:,.2f} tỷ VNĐ")
