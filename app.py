import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Load dataset
@st.cache_resource
def load_data():
    return pd.read_excel('docs/beras.xlsx')

dataset = load_data()

# Sidebar
with st.sidebar:
    st.title('SETOR')
    page = option_menu("Menu", ["Dashboard", "Visualization", "Predict"],
                       icons=["house", "bar-chart", "calculator"])

# Dashboard
if page == 'Dashboard':
    st.title('SETOR (Sembako Predictor)')
    st.write('Aplikasi ini bertujuan untuk memprediksi harga beras berdasarkan tanggal, bulan, tahun, dan jenis beras.')
    st.subheader('Dataset')
    st.write(dataset)

# Visualisasi
elif page == 'Visualization':
    st.title('Visualisasi Perkembangan Harga Beras')

    # Dropdowns for selecting month, year, and rice type
    bulan_dict = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
        'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }
    selected_month = st.selectbox('Pilih Bulan:', options=list(bulan_dict.keys()))
    selected_year = st.selectbox('Pilih Tahun:', dataset['Tahun'].unique())
    selected_rice = st.selectbox('Pilih Jenis Beras:', dataset['Komoditas'].unique())

    # Filter dataset based on selected month, year, and rice type
    filtered_data = dataset[(dataset['Bulan'] == bulan_dict[selected_month]) & (dataset['Tahun'] == selected_year) & (dataset['Komoditas'] == selected_rice)]

    # Group by date and calculate average price
    df_grouped = filtered_data.groupby(['Tanggal'])['Harga'].mean().reset_index()

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped['Tanggal'], df_grouped['Harga'], marker='o')
    plt.title(f'Perkembangan Harga {selected_rice} pada Bulan {selected_month} Tahun {selected_year}')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Rata-rata')
    plt.xticks(df_grouped['Tanggal'].unique(), rotation=45)
    st.pyplot(plt)

    # Dropdown for selecting year to display trend
    selected_trend_year = st.selectbox('Pilih Tahun untuk Tren:', dataset['Tahun'].unique())

    # Plot the trend for all rice types per month for the selected year
    st.subheader(f'Tren Harga Beras per Bulan untuk Semua Jenis Beras Tahun {selected_trend_year}')

    # Group by rice type and month for the selected year, and calculate average price
    df_trend = dataset[dataset['Tahun'] == selected_trend_year].groupby(['Komoditas', 'Bulan'])['Harga'].mean().reset_index()

    # Plot the data
    plt.figure(figsize=(10, 6))
    for rice_type in df_trend['Komoditas'].unique():
        plt.plot(df_trend[df_trend['Komoditas'] == rice_type]['Bulan'], df_trend[df_trend['Komoditas'] == rice_type]['Harga'], label=rice_type, marker='o')

    plt.title(f'Tren Harga Beras per Bulan untuk Semua Jenis Beras Tahun {selected_trend_year}')
    plt.xlabel('Bulan')
    plt.ylabel('Harga Rata-rata')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

# Predict
elif page == 'Predict':
    st.title('Prediksi Harga Beras Baru')

    # Input form
    tanggal = st.number_input('Masukkan Tanggal:', min_value=1, max_value=31)
    bulan_dict = {
        'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
        'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
    }
    bulan = st.selectbox('Masukkan Bulan:', options=list(bulan_dict.keys()))
    tahun = st.selectbox('Masukkan Tahun:', options=range(2020, 2027))
    komoditas_dict = {
        'Beras Bulog': 1, 'Beras Belida': 2, 'Beras Badminton': 3, 'Beras Pandan Wangi': 4,
        'Beras BTN': 5, 'Beras Anak Dara': 6, 'Beras Bumi Ayu': 7
    }
    komoditas_list = list(komoditas_dict.keys())
    selected_komoditas = st.selectbox('Pilih Jenis Beras:', options=komoditas_list)

    # Menghitung fitur baru: Jumlah hari sejak awal dataset
    start_date = dataset[['Tanggal', 'Bulan', 'Tahun']].min()
    dataset['Days'] = (dataset['Tanggal'] - start_date['Tanggal']) + ((dataset['Bulan'] - start_date['Bulan']) * 30) + ((dataset['Tahun'] - start_date['Tahun']) * 365)

    # Train-test split
    X = dataset[['Days', 'Komoditas']]
    y = dataset['Harga']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Menghitung fitur baru untuk data prediksi
    start_date_input = pd.Timestamp(2020, 1, 1)  # Tanggal awal dataset
    days_input = (pd.Timestamp(tahun, bulan_dict[bulan], tanggal) - start_date_input).days
    selected_komoditas_numeric = komoditas_dict[selected_komoditas]

    # Predict
    prediction_input = [[days_input, selected_komoditas_numeric]]  # Menggunakan nilai numerik jenis beras
    predicted_price = model.predict(prediction_input)

    # Extract actual and predicted prices
    actual_prices = y_test.values
    predicted_prices = model.predict(X_test)

    st.subheader('Hasil Prediksi')
    st.write(f'Prediksi harga {selected_komoditas} untuk tanggal {tanggal} {bulan} {tahun}:')
    st.write('Rp {:,.0f}'.format(round(predicted_price[0])))

    # Plotting the comparison between actual and predicted prices
    st.subheader('Perbandingan Antara Harga Sebenarnya dan Harga Prediksi')
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, label='Harga Sebenarnya', marker='o', color='blue')
    plt.plot(predicted_prices, label='Harga Prediksi', marker='o', color='red')
    plt.plot(np.arange(len(actual_prices)), actual_prices, 'b--', label='Ketika Prediksi Sempurna', alpha=0.3)
    plt.xlabel('Sampel Data Uji')
    plt.ylabel('Harga')
    plt.title('Perbandingan Antara Harga Sebenarnya dan Harga Prediksi')
    plt.legend()
    st.pyplot(plt)
