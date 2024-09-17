import warnings
import numpy as np
import pickle as pk
import streamlit as st


@st.cache_data
def load_models():
    try:
        with open('./Model/dataframe.pkl', 'rb') as dataframe:
            df = pk.load(dataframe)
        with open('./Model/pipeline.pkl', 'rb') as pipeline:
            pipe = pk.load(pipeline)
        return df, pipe

    except (FileNotFoundError, pk.UnpicklingError) as e:
        st.error(f'Error loading models: {e}')
        st.stop()


df, pipe = load_models()

st.title('Amazon Laptop Price Predictor')

company = st.selectbox('Brand', df['Company'].unique())
laptop_type = st.selectbox('Type', df['TypeName'].unique())
cpu = st.selectbox('CPU', df['CPU Brand'].unique())
gpu = st.selectbox('GPU', df['GPU Brand'].unique())
os = st.selectbox('Operating System', df['OS'].unique())

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])

ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
screen_size = st.slider('Screen size (inches)', 10.0, 18.0, 13.0)

resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
weight = st.number_input('Weight of the Laptop (kg)', min_value=0.0, step=0.1)

if st.button('Predict Price'):
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = np.array([
        company, laptop_type, ram, weight, touchscreen,
        ips, ppi, cpu, hdd, ssd, gpu, os
    ])
    query = query.reshape(1, -1)

    try:
        warnings.filterwarnings(
            'ignore', message='X does not have valid feature names'
        )
        predicted_price = np.exp(pipe.predict(query)[0])
        st.success(
            f'The predicted price of this configuration is ₹{int(predicted_price):,}'
        )

    except Exception as e:
        st.error(f'Error during prediction: {e}')
