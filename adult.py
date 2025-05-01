import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load preprocessing pipeline
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="adult_census.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul aplikasi
st.title("Adult Census Income")

# Input pengguna
age = st.slider("Umur", 18, 90, 30)
workclass = st.selectbox("Kelas Kerja", label_encoders['workclass'].classes_)
fnlwgt = st.number_input("FNLWGT", 10000, 1000000, 50000)
education = st.selectbox("Pendidikan", label_encoders['education'].classes_)
education_num = st.slider("Jumlah Pendidikan", 1, 16, 10)
marital_status = st.selectbox("Status Perkawinan", label_encoders['marital.status'].classes_)
occupation = st.selectbox("Pekerjaan", label_encoders['occupation'].classes_)
relationship = st.selectbox("Hubungan", label_encoders['relationship'].classes_)
race = st.selectbox("Ras", label_encoders['race'].classes_)
sex = st.selectbox("Jenis Kelamin", label_encoders['sex'].classes_)
capital_gain = st.number_input("Keuntungan Kapital", 0, 100000, 0)
capital_loss = st.number_input("Kerugian Kapital", 0, 100000, 0)
hours_per_week = st.slider("Jam per Minggu", 1, 99, 40)
native_country = st.selectbox("Asal Negara", label_encoders['native.country'].classes_)

# Buat input dictionary
input_dict = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'education': education,
    'education.num': education_num,
    'marital.status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'sex': sex,
    'capital.gain': capital_gain,
    'capital.loss': capital_loss,
    'hours.per.week': hours_per_week,
    'native.country': native_country
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Encode kolom kategorikal
for col in input_df.columns:
    if col in label_encoders:
        encoder = label_encoders[col]
        input_df[col] = encoder.transform(input_df[col])

# Scaling
input_scaled = scaler.transform(input_df)
input_array = np.array(input_scaled, dtype=np.float32)

# Prediksi dengan model TFLite
if st.button("Prediksi Pendapatan"):
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]
    label = ">50K" if output > 0.5 else "<=50K"

    st.subheader("Hasil Prediksi:")
    st.write(f"Pendapatan diprediksi: **{label}**")
    st.progress(float(min(output, 1.0)))
