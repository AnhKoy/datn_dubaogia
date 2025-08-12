import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np

# Thông tin model trên Drive
MODEL_FILE = "random_forest_with_encoding.pkl"
MODEL_ID = "1ZerzZ9eLY0twOlbUABwJn2MnYp4DKTHJ"

# Tải model từ Drive nếu chưa có
if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# Load model (trong dict nếu cần)
loaded = joblib.load(MODEL_FILE)
if isinstance(loaded, dict) and "model" in loaded:
    model = loaded["model"]
else:
    model = loaded

# Load CSV riêng từ repo
df = pd.read_csv("bds_dat_clean (1).csv")

st.title("🔍 Dự đoán giá Bất động sản")

dien_tich = st.number_input("Diện tích (m²)", 10.0, 2000.0, 100.0)
mat_tien = st.number_input("Mặt tiền (m)", 0.5, 50.0, 5.0)
duong_vao = st.number_input("Đường vào (m)", 0.5, 50.0, 3.0)

loai_bds = st.selectbox("Loại BDS", sorted(df["Loại BDS"].dropna().unique()))
giay_to = st.selectbox("Giấy tờ pháp lý", sorted(df["Giấy tờ pháp lý"].dropna().unique()))
xa_phuong = st.selectbox("Xã/Phường", sorted(df["Xã/Phường"].dropna().unique()))
quan_huyen = st.selectbox("Quận/Huyện", sorted(df["Quận/Huyện"].dropna().unique()))

if st.button("Dự đoán giá"):
    input_df = pd.DataFrame([{
        "Diện tích (m²)": dien_tich,
        "Mặt tiền (m)": mat_tien,
        "Đường vào (m)": duong_vao,
        "Loại BDS": loai_bds,
        "Giấy tờ pháp lý": giay_to,
        "Xã/Phường": xa_phuong,
        "Quận/Huyện": quan_huyen
    }])
    try:
        pred = model.predict(input_df)[0]
        st.success(f"💰 Giá dự đoán: {pred:,.2f} tỷ VNĐ")
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")

