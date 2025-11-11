import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title("EV Charging Prediction & Analysis")

uploaded_file = st.file_uploader("Upload cleaned_ev_dataset.csv", type=["csv"]) 

model = None
try:
    model = joblib.load('models/ev_energy_model.pkl')
except:
    st.info('Model not found. Run train_model.py to generate models/ev_energy_model.pkl')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
    df = df.dropna(subset=['StartTime'])
    df['Hour'] = df['StartTime'].dt.hour

    st.subheader('Dataset Preview')
    st.dataframe(df.head())

    st.subheader('Peak Charging Hours')
    hour_counts = df['Hour'].value_counts().sort_index()
    fig, ax = plt.subplots()
    hour_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.subheader('Predict Energy Consumption')
    duration = st.number_input('Charging Duration (hours)', min_value=0.0, max_value=24.0, value=1.0)
    hour = st.number_input('Charging Start Hour (0-23)', min_value=0, max_value=23, value=18)

    if st.button('Predict'):
        if model is None:
            st.error('Model not available. Please run training script first.')
        else:
            pred = model.predict([[hour, duration]])[0]
            st.success(f'Predicted Energy Consumption: {pred:.2f} kWh')

    st.subheader('Download Predictions')
    if 'EnergyConsumption' in df.columns and model is not None:
        X = df[['Hour']]
        if 'ChargingDuration' in df.columns:
            X['ChargingDuration'] = df['ChargingDuration']
        else:
            X['ChargingDuration'] = df['EnergyConsumption'] / 1.0
        preds = model.predict(X)
        df['PredictedEnergy'] = preds
        st.download_button('Download predictions CSV', df.to_csv(index=False), file_name='predictions.csv')
else:
    st.info('Upload the cleaned dataset to proceed.')
