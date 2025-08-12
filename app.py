import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

# ===== 1. Tải model từ Google Drive =====
MODEL_FILE = "random_forest_with_encoding.pkl"
MODEL_ID = "1ZerzZ9eLY0twOlbUABwJn2MnYp4DKTHJ"  # ID của file trên Google Drive

if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

# ===== 2. Load model + scaler + encoding_maps =====
loaded_obj = joblib.load(MODEL_FILE)

if not isinstance(loaded_obj, dict):
    st.error("❌ File pkl không đúng định dạng dict đã lưu.")
    st.stop()

model = loaded_obj.get("model")
scaler = loaded_obj.get("scaler")
encoding_maps = loaded_obj.get("encoding_maps")
mean_target = loaded_obj.get("mean_target")
cat_cols = loaded_obj.get("cat_cols")

# ===== 3. Load CSV để lấy danh sách giá trị cho dropdown =====
DATA_FILE = "bds_dat_clean.csv"
if not os.path.exists(DATA_FILE):
    st.error(f"❌ Không tìm thấy file dữ liệu '{DATA_FILE}'. Hãy upload file CSV hoặc đính kèm khi deploy.")
    st.stop()

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()  # Xóa khoảng trắng thừa

# ===== 4. Giao diện Streamlit =====
st.title("🔍 Dự đoán giá Bất động sản")
st.write("Nhập thông tin bất động sản để dự đoán:")

# Input dạng số
dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, max_value=2000.0, value=100.0)
mat_tien = st.number_input("Mặt tiền (m)", min_value=0.5, max_value=50.0, value=5.0)
duong_vao = st.number_input("Đường vào (m)", min_value=0.5, max_value=50.0, value=3.0)

# Input dạng chọn
loai_bds = st.selectbox("Loại BĐS", sorted(df["Loại BDS"].dropna().unique()))
giay_to = st.selectbox("Giấy tờ pháp lý", sorted(df["Giấy tờ pháp lý"].dropna().unique()))
xa_phuong = st.selectbox("Xã/Phường", sorted(df["Xã/Phường"].dropna().unique()))
quan_huyen = st.selectbox("Quận/Huyện", sorted(df["Quận/Huyện"].dropna().unique()))

# ===== 5. Xử lý khi bấm nút =====
if st.button("Dự đoán giá"):
    try:
        # Tạo DataFrame input ban đầu
        input_df = pd.DataFrame([{
            "Diện tích (m²)": dien_tich,
            "Mặt tiền (m)": mat_tien,
            "Đường vào (m)": duong_vao,
            "Loại BDS": loai_bds,
            "Giấy tờ pháp lý": giay_to,
            "Xã/Phường": xa_phuong,
            "Quận/Huyện": quan_huyen
        }])

        # ===== Feature engineering =====
        input_df["log_area"] = np.log1p(input_df["Diện tích (m²)"])
        input_df["frontage_to_road"] = input_df["Mặt tiền (m)"] / (input_df["Đường vào (m)"] + 0.1)

        # ===== Target encoding =====
        for col in cat_cols:
            input_df[col + "_te"] = input_df[col].map(encoding_maps[col]).fillna(mean_target)
            input_df.drop(columns=[col], inplace=True)

        # Chỉ giữ đúng các cột model cần
        final_X = input_df[
            ["Diện tích (m²)", "Mặt tiền (m)", "Đường vào (m)",
             "log_area", "frontage_to_road"] +
            [c + "_te" for c in cat_cols]
        ]

        # ===== Scaling =====
        final_X_scaled = scaler.transform(final_X)

        # ===== Dự đoán =====
        predicted_price = model.predict(final_X_scaled)[0]
        st.success(f"💰 Giá dự đoán: {predicted_price:,.2f} tỷ VNĐ")

    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")





# import streamlit as st
# import pandas as pd
# import joblib
# import gdown
# import os
# import numpy as np

# # ===== 1. Tải model từ Google Drive =====
# MODEL_FILE = "random_forest_with_encoding.pkl"
# MODEL_ID = "1ZerzZ9eLY0twOlbUABwJn2MnYp4DKTHJ"

# if not os.path.exists(MODEL_FILE):
#     url = f"https://drive.google.com/uc?id={MODEL_ID}"
#     gdown.download(url, MODEL_FILE, quiet=False)

# # ===== 2. Load model & encoder =====
# loaded = joblib.load(MODEL_FILE)

# if isinstance(loaded, dict):
#     model = loaded.get("model", loaded)
#     encoders = loaded.get("encoders", {})  # dict chứa encoder cho từng cột
# else:
#     model = loaded
#     encoders = {}

# # ===== 3. Load CSV để lấy danh sách lựa chọn =====
# df = pd.read_csv("bds_dat_clean (1).csv")

# # ===== 4. Giao diện Streamlit =====
# st.title("🔍 Dự đoán giá Bất động sản")

# dien_tich = st.number_input("Diện tích (m²)", 10.0, 2000.0, 100.0)
# mat_tien = st.number_input("Mặt tiền (m)", 0.5, 50.0, 5.0)
# duong_vao = st.number_input("Đường vào (m)", 0.5, 50.0, 3.0)

# loai_bds = st.selectbox("Loại BDS", sorted(df["Loại BDS"].dropna().unique()))
# giay_to = st.selectbox("Giấy tờ pháp lý", sorted(df["Giấy tờ pháp lý"].dropna().unique()))
# xa_phuong = st.selectbox("Xã/Phường", sorted(df["Xã/Phường"].dropna().unique()))
# quan_huyen = st.selectbox("Quận/Huyện", sorted(df["Quận/Huyện"].dropna().unique()))

# # ===== 5. Xử lý khi bấm nút =====
# if st.button("Dự đoán giá"):
#     try:
#         input_df = pd.DataFrame([{
#             "Diện tích (m²)": dien_tich,
#             "Mặt tiền (m)": mat_tien,
#             "Đường vào (m)": duong_vao,
#             "Loại BDS": loai_bds,
#             "Giấy tờ pháp lý": giay_to,
#             "Xã/Phường": xa_phuong,
#             "Quận/Huyện": quan_huyen
#         }])

#         # Encode dữ liệu nếu có encoders
#         for col, encoder in encoders.items():
#             if col in input_df.columns:
#                 input_df[col] = encoder.transform(input_df[col])

#         # Dự đoán
#         pred = model.predict(input_df)[0]
#         st.success(f"💰 Giá dự đoán: {pred:,.2f} tỷ VNĐ")

#     except Exception as e:
#         st.error(f"❌ Lỗi: {e}")



# import streamlit as st
# import pandas as pd
# import joblib
# import gdown
# import os
# import numpy as np

# # Thông tin model trên Drive
# MODEL_FILE = "random_forest_with_encoding.pkl"
# MODEL_ID = "1ZerzZ9eLY0twOlbUABwJn2MnYp4DKTHJ"

# # Tải model từ Drive nếu chưa có
# if not os.path.exists(MODEL_FILE):
#     url = f"https://drive.google.com/uc?id={MODEL_ID}"
#     gdown.download(url, MODEL_FILE, quiet=False)

# # Load model (trong dict nếu cần)
# loaded = joblib.load(MODEL_FILE)
# if isinstance(loaded, dict) and "model" in loaded:
#     model = loaded["model"]
# else:
#     model = loaded

# # Load CSV riêng từ repo
# df = pd.read_csv("bds_dat_clean (1).csv")

# st.title("🔍 Dự đoán giá Bất động sản")

# dien_tich = st.number_input("Diện tích (m²)", 10.0, 2000.0, 100.0)
# mat_tien = st.number_input("Mặt tiền (m)", 0.5, 50.0, 5.0)
# duong_vao = st.number_input("Đường vào (m)", 0.5, 50.0, 3.0)

# loai_bds = st.selectbox("Loại BDS", sorted(df["Loại BDS"].dropna().unique()))
# giay_to = st.selectbox("Giấy tờ pháp lý", sorted(df["Giấy tờ pháp lý"].dropna().unique()))
# xa_phuong = st.selectbox("Xã/Phường", sorted(df["Xã/Phường"].dropna().unique()))
# quan_huyen = st.selectbox("Quận/Huyện", sorted(df["Quận/Huyện"].dropna().unique()))

# if st.button("Dự đoán giá"):
#     input_df = pd.DataFrame([{
#         "Diện tích (m²)": dien_tich,
#         "Mặt tiền (m)": mat_tien,
#         "Đường vào (m)": duong_vao,
#         "Loại BDS": loai_bds,
#         "Giấy tờ pháp lý": giay_to,
#         "Xã/Phường": xa_phuong,
#         "Quận/Huyện": quan_huyen
#     }])
#     try:
#         pred = model.predict(input_df)[0]
#         st.success(f"💰 Giá dự đoán: {pred:,.2f} tỷ VNĐ")
#     except Exception as e:
#         st.error(f"❌ Lỗi: {e}")







