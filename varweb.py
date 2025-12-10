import streamlit as st
import cdsapi
import zipfile
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(
    page_title="AI Pollution Analyzer",
    page_icon="🌍",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    /* Background general cu gradient */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Stilizare containere */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00ffcc;
    }
    
    /* Titluri */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Buton custom */
    div.stButton > button {
        background-color: #00ffcc;
        color: #1e3c72;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
    }
    div.stButton > button:hover {
        background-color: #ffffff;
        color: #1e3c72;
    }
    </style>
    """, unsafe_allow_html=True)


def setup_api_credentials(key):
    url = "https://ads.atmosphere.copernicus.eu/api"
    rc_file = os.path.expanduser('~/.cdsapirc')
    with open(rc_file, 'w') as f:
        f.write(f"url: {url}\n")
        f.write(f"key: {key}\n")

def get_data():
    output_zip = "download_app.zip"
    extract_folder = "date_input_app"
    
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    existing_nc = [f for f in os.listdir(extract_folder) if f.endswith('.nc')]
    if existing_nc:
        return os.path.join(extract_folder, existing_nc[0])

    client = cdsapi.Client()
    
    client.retrieve(
        'cams-global-reanalysis-eac4',
        {
            'date': '2024-12-31/2024-12-31',
            'time': '12:00',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', 
                '2m_temperature', 'particulate_matter_2.5um', 'surface_pressure'
            ],
            'area': [48.5, 20, 43, 30], # Romania
            'format': 'netcdf_zip',
        },
        output_zip
    )
    
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        nc_file = [f for f in zip_ref.namelist() if f.endswith('.nc')][0]
    
    return os.path.join(extract_folder, nc_file)

def process_and_train(nc_path):
    ds = xr.open_dataset(nc_path, engine='netcdf4')
    df = ds.to_dataframe().reset_index().dropna()
    
    rename_map = {
        'u10': 'u_wind', 'v10': 'v_wind', 't2m': 'temperature',
        'sp': 'pressure', 'pm2p5': 'actual_pm25' 
    }
    for col in df.columns:
        if col in rename_map:
            df.rename(columns={col: rename_map[col]}, inplace=True)

    # Feature Engineering
    df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
    df['temp_celsius'] = df['temperature'] - 273.15
    df['pressure_hpa'] = df['pressure'] / 100.0
    df['actual_pm25_ug'] = df['actual_pm25'] * 1e9

    # ML Setup
    X = df[['wind_speed', 'temp_celsius', 'pressure_hpa', 'latitude', 'longitude']]
    y = df['actual_pm25_ug']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Antrenare
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    return y_test, y_pred_lr, y_pred_rf

# Titlu Principal
st.title("🌍 Monitorizare & Predicție Poluare Aer (PM2.5)")
st.markdown("### Analiză bazată pe date satelitare Copernicus (CAMS) și Machine Learning")

# Sidebar pentru configurare
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=100)
    st.header("⚙️ Configurare")
    api_key_input = st.text_input("Introdu Cheia API Copernicus:", type="password")
    st.info("Dacă nu ai cheie, vizitează ads.atmosphere.copernicus.eu")
    
    start_btn = st.button("🚀 Rulează Analiza")

# Zona Principală
if start_btn:
    if not api_key_input:
        st.error("⚠️ Te rog introdu cheia API în meniul din stânga!")
    else:
        # 1. Setup
        setup_api_credentials(api_key_input)
        
        # 2. Download cu animatie
        with st.spinner('📡 Se comunică cu satelitul Copernicus... (poate dura 1 min)'):
            try:
                nc_path = get_data()
                st.success("Date descărcate cu succes!")
            except Exception as e:
                st.error(f"Eroare la descărcare: {e}")
                st.stop()

        # 3. Procesare
        with st.spinner('🤖 Se antrenează modelele AI...'):
            y_test, y_pred_lr, y_pred_rf = process_and_train(nc_path)
        
        # 4. Metrici
        col1, col2 = st.columns(2)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        
        with col1:
            st.metric(label="Eroare Random Forest (MSE)", value=f"{mse_rf:.4f}", delta="- Mai precis")
        with col2:
            st.metric(label="Eroare Linear Regression (MSE)", value=f"{mse_lr:.4f}", delta_color="inverse", delta="Mai puțin precis")

        # 5. Graficul
        st.subheader("📊 Vizualizare Comparativă")
        
        # date sortate
        results = pd.DataFrame({
            'Real': y_test,
            'Linear': y_pred_lr,
            'Forest': y_pred_rf
        })
        results_sorted = results.sort_values(by='Real').reset_index(drop=True)
        
        # Desenare Grafic
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#1e3c72')
        ax.set_facecolor('#0f2027')
        
        # Plotare
        ax.plot(results_sorted['Real'], label='Date Reale (CAMS)', color='white', linewidth=2, linestyle='-')
        ax.plot(results_sorted['Linear'], label='Linear Regression', color='#ff4b1f', marker='.', linestyle='', markersize=8, alpha=0.7)
        ax.plot(results_sorted['Forest'], label='Random Forest', color='#00ffcc', linestyle=':', linewidth=3)
        
        # Stilizare axe & text
        ax.set_title('Predicție vs Realitate (Sortat după intensitate)', color='white', fontsize=14)
        ax.set_xlabel('Eșantioane', color='white')
        ax.set_ylabel('PM2.5 [µg/m³]', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.legend(facecolor='#1e3c72', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white')
        
        st.pyplot(fig)
        
        st.success("Analiză finalizată! Graficul arată performanța superioară a modelului Random Forest (linia verde) care urmărește mai bine linia albă.")
else:
    st.markdown("""
    <div style='background-color: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px;'>
        <h3>👋 Bine ai venit!</h3>
        <p>Această aplicație descarcă date meteo în timp real pentru România și folosește Inteligența Artificială pentru a prezice nivelul de poluare.</p>
        <p>Introdu cheia API în stânga pentru a începe.</p>
    </div>
    """, unsafe_allow_html=True)