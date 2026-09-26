import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Customer Segmentation - K-Means",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation dengan K-Means")
st.write(
    "Aplikasi ini digunakan untuk memprediksi cluster pelanggan berdasarkan "
    "Age, Income, Recency, Total Amount Spent, dan Total Purchases."
)

FEATURES = [
    "Age",
    "Income",
    "Recency",
    "Total_Amount_Spent",
    "Total_Purchases"
]

MODEL_FILE = Path("kmeans_model.joblib")
SCALER_FILE = Path("scaler.joblib")
PROFILE_FILE = Path("cluster_profile.csv")

if not MODEL_FILE.exists() or not SCALER_FILE.exists():
    st.error(
        "File model belum ditemukan. Jalankan notebook terlebih dahulu "
        "sampai bagian Deployment."
    )
    st.stop()

kmeans = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

cluster_profile = None
if PROFILE_FILE.exists():
    cluster_profile = pd.read_csv(PROFILE_FILE, index_col=0)

st.sidebar.header("Input Data Pelanggan")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)

income = st.sidebar.number_input(
    "Income",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

recency = st.sidebar.number_input(
    "Recency",
    min_value=0,
    max_value=365,
    value=30,
    step=1
)

total_spent = st.sidebar.number_input(
    "Total Amount Spent",
    min_value=0.0,
    value=500.0,
    step=50.0
)

total_purchases = st.sidebar.number_input(
    "Total Purchases",
    min_value=0,
    value=10,
    step=1
)

input_df = pd.DataFrame(
    [[age, income, recency, total_spent, total_purchases]],
    columns=FEATURES
)

st.subheader("Data Input")
st.dataframe(input_df, use_container_width=True)

if st.button("Prediksi Cluster", type="primary"):
    input_scaled = scaler.transform(input_df)
    cluster = int(kmeans.predict(input_scaled)[0])

    st.success(f"Pelanggan termasuk ke **Cluster {cluster}**")

    if cluster_profile is not None:
        matching_index = None

        for idx in cluster_profile.index:
            if str(idx) == str(cluster):
                matching_index = idx
                break

        if matching_index is not None:
            profile = cluster_profile.loc[matching_index]

            st.subheader(f"Profil Cluster {cluster}")

            col1, col2, col3 = st.columns(3)

            if "Income" in profile.index:
                col1.metric(
                    "Rata-rata Income",
                    f"{profile['Income']:,.2f}"
                )

            if "Total_Amount_Spent" in profile.index:
                col2.metric(
                    "Rata-rata Total Spent",
                    f"{profile['Total_Amount_Spent']:,.2f}"
                )

            if "Total_Purchases" in profile.index:
                col3.metric(
                    "Rata-rata Purchases",
                    f"{profile['Total_Purchases']:,.2f}"
                )

            st.dataframe(
                profile.to_frame("Rata-rata Cluster"),
                use_container_width=True
            )

st.divider()
st.caption(
    "Model dibuat menggunakan K-Means Clustering pada dataset "
    "Customer Personality Analysis."
)
