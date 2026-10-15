import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Mochi Shop Forecasting", page_icon="🍡", layout="wide")
st.title("🍡 Mochi House")
st.subheader("Daily Sales & Demand Forecasting System")
st.write(
    "Aplikasi analitik data untuk menganalisis penjualan harian berbagai varian mochi "
    "dan memperkirakan permintaan pada hari berikutnya."
)

@st.cache_data
def create_sample_data():
    dates = pd.date_range("2026-09-01", periods=30, freq="D")
    products = {
        "Mochi Original": [35, 38, 34, 40, 42, 45, 50, 37, 39, 36, 43, 46, 49, 53, 40, 42, 38, 45, 48, 52, 56, 43, 45, 41, 48, 51, 55, 59, 46, 50],
        "Mochi Matcha": [25, 27, 24, 29, 31, 34, 38, 26, 28, 25, 32, 34, 37, 41, 29, 31, 27, 34, 36, 40, 43, 31, 33, 30, 36, 39, 42, 46, 34, 37],
        "Mochi Daifuku": [30, 33, 29, 35, 37, 41, 46, 32, 34, 31, 38, 41, 44, 49, 35, 37, 33, 40, 43, 47, 51, 38, 40, 36, 43, 46, 50, 54, 41, 44],
        "Mochi Cokelat": [28, 30, 26, 32, 34, 37, 42, 29, 31, 28, 35, 37, 40, 44, 32, 34, 30, 37, 39, 43, 47, 34, 36, 33, 39, 42, 46, 50, 37, 40],
        "Mochi Stroberi": [22, 24, 21, 26, 28, 31, 35, 23, 25, 22, 29, 31, 34, 38, 26, 28, 24, 31, 33, 37, 40, 29, 31, 27, 33, 36, 39, 43, 31, 34],
        "Mochi Kacang Merah": [20, 22, 19, 24, 26, 29, 33, 21, 23, 20, 27, 29, 32, 36, 24, 26, 22, 29, 31, 35, 38, 27, 29, 25, 31, 34, 37, 41, 29, 32],
    }
    return pd.DataFrame([
        {"Tanggal": date, "Produk": product, "Penjualan Aktual": quantity}
        for product, quantities in products.items()
        for date, quantity in zip(dates, quantities)
    ])

data = create_sample_data()

st.sidebar.header("Pengaturan Forecasting")
selected_product = st.sidebar.selectbox("Pilih varian mochi", data["Produk"].unique())
method = st.sidebar.selectbox("Metode forecasting", [
    "Simple Moving Average", "Weighted Moving Average", "Exponential Smoothing"
])
period = st.sidebar.slider("Jumlah periode (hari)", 2, 7, 3) if method != "Exponential Smoothing" else None
alpha = st.sidebar.slider("Nilai alpha", 0.1, 0.9, 0.3, 0.1) if method == "Exponential Smoothing" else None
safety_stock = st.sidebar.slider("Tambahan stok pengaman (%)", 0, 30, 10, 5)

st.header("1. Data Penjualan Harian")
st.caption("Data ini merupakan data simulasi untuk studi kasus. Kamu bisa mengubah nilai pada kolom Penjualan Aktual.")
edited_data = st.data_editor(
    data, use_container_width=True, hide_index=True, disabled=["Tanggal", "Produk"],
    column_config={
        "Tanggal": st.column_config.DateColumn("Tanggal", format="DD/MM/YYYY"),
        "Produk": st.column_config.TextColumn("Varian Mochi"),
        "Penjualan Aktual": st.column_config.NumberColumn("Penjualan Aktual (pcs)", min_value=0, step=1)
    }, key="sales_editor"
)

product_data = edited_data[edited_data["Produk"] == selected_product].copy()
product_data = product_data.sort_values("Tanggal").reset_index(drop=True)
actual = pd.to_numeric(product_data["Penjualan Aktual"], errors="coerce").to_numpy(dtype=float)

if len(actual) < 2:
    st.error("Minimal diperlukan dua hari data penjualan.")
    st.stop()
if not np.isfinite(actual).all():
    st.error("Data penjualan tidak boleh kosong dan harus berupa angka yang valid.")
    st.stop()
if np.any(actual < 0):
    st.error("Penjualan aktual tidak boleh bernilai negatif.")
    st.stop()

forecast = np.full(len(actual), np.nan)
if method == "Simple Moving Average":
    for i in range(period, len(actual)):
        forecast[i] = np.mean(actual[i-period:i])
    next_forecast = np.mean(actual[-period:])
    explanation = f"Prediksi menggunakan rata-rata penjualan {period} hari sebelumnya."
elif method == "Weighted Moving Average":
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    for i in range(period, len(actual)):
        forecast[i] = np.sum(actual[i-period:i] * weights)
    next_forecast = np.sum(actual[-period:] * weights)
    explanation = f"Prediksi memberi bobot lebih besar pada data terbaru selama {period} hari."
else:
    forecast[0] = actual[0]
    for i in range(1, len(actual)):
        forecast[i] = forecast[i-1] + alpha * (actual[i-1] - forecast[i-1])
    next_forecast = forecast[-1] + alpha * (actual[-1] - forecast[-1])
    explanation = f"Prediksi diperbarui dengan nilai alpha {alpha:.1f}."

next_forecast = max(0, float(next_forecast))
recommended_production = int(np.ceil(next_forecast * (1 + safety_stock / 100)))

st.header("2. Hasil Forecasting")
st.write(f"**Varian:** {selected_product}  \n**Metode:** {method}  \n{explanation}")
c1, c2, c3 = st.columns(3)
c1.metric("Rata-rata penjualan harian", f"{np.mean(actual):.1f} pcs")
c2.metric("Prediksi hari berikutnya", f"{next_forecast:.1f} pcs")
c3.metric("Rekomendasi produksi", f"{recommended_production} pcs")

result_data = product_data[["Tanggal", "Produk", "Penjualan Aktual"]].copy()
result_data["Hasil Forecast"] = np.round(forecast, 2)
st.dataframe(result_data, use_container_width=True, hide_index=True)
st.caption(
    "Rekomendasi produksi = prediksi permintaan + stok pengaman. "
    "Perhitungan ini belum memperhitungkan stok yang sudah tersedia, bahan baku, "
    "kapasitas produksi, atau mochi yang tidak terjual."
)

st.header("3. Visualisasi Penjualan dan Forecast")
chart_data = result_data.melt(
    id_vars=["Tanggal"], value_vars=["Penjualan Aktual", "Hasil Forecast"],
    var_name="Jenis Data", value_name="Jumlah Mochi"
)
fig = px.line(
    chart_data, x="Tanggal", y="Jumlah Mochi", color="Jenis Data", markers=True,
    title=f"Penjualan Aktual vs Forecast - {selected_product}",
    labels={"Tanggal": "Tanggal", "Jumlah Mochi": "Jumlah (pcs)", "Jenis Data": "Keterangan"}
)
st.plotly_chart(fig, use_container_width=True)

st.header("4. Evaluasi Akurasi Forecasting")
valid = ~np.isnan(forecast)
if method == "Exponential Smoothing":
    valid[0] = False
actual_valid, forecast_valid = actual[valid], forecast[valid]

if len(actual_valid) == 0:
    st.info("Belum ada data yang cukup untuk menghitung akurasi.")
else:
    errors = actual_valid - forecast_valid
    mad = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    nonzero = actual_valid != 0
    mape = np.mean(np.abs(errors[nonzero] / actual_valid[nonzero])) * 100 if np.any(nonzero) else None
    c1, c2, c3 = st.columns(3)
    c1.metric("MAD", f"{mad:.2f}")
    c2.metric("MSE", f"{mse:.2f}")
    c3.metric("MAPE", f"{mape:.2f}%" if mape is not None else "Tidak dapat dihitung")
    st.caption(
        "MAD adalah rata-rata kesalahan absolut, MSE rata-rata kuadrat kesalahan, "
        "dan MAPE rata-rata persentase kesalahan. Nilai lebih kecil umumnya menunjukkan prediksi lebih akurat."
    )

st.header("5. Ringkasan Analisis")
st.write(
    f"Berdasarkan data simulasi, rata-rata penjualan {selected_product} adalah "
    f"{np.mean(actual):.1f} pcs per hari. Dengan metode {method}, permintaan hari "
    f"berikutnya diperkirakan {next_forecast:.1f} pcs. Dengan tambahan stok pengaman "
    f"{safety_stock}%, jumlah produksi yang disarankan adalah {recommended_production} pcs."
)
st.info("Catatan: data masih berupa simulasi. Untuk penggunaan nyata, gantilah dengan catatan penjualan toko mochi.")
