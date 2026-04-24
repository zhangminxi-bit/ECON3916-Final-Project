### 6.2 app.py Template

import streamlit as st
import numpy as np
import joblib

st.title('Breakout Success Predictor')
st.markdown('Enter breakout characteristics to predict whether the signal will succeed.')

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Sidebar inputs
bk_strength  = st.sidebar.slider('Breakout Strength (%)', -2.0, 10.0, 1.0) / 100
vol_ratio    = st.sidebar.slider('Volume Ratio (vs 20d avg)', 0.5, 5.0, 1.5)
ret_5d       = st.sidebar.slider('Prior 5-day Return (%)', -10.0, 15.0, 2.0) / 100
ret_20d      = st.sidebar.slider('Prior 20-day Return (%)', -15.0, 30.0, 5.0) / 100
volatility   = st.sidebar.slider('20-day Volatility', 0.005, 0.05, 0.015)
atr_ratio    = st.sidebar.slider('ATR Ratio', 0.005, 0.04, 0.015)

X_input = scaler.transform([[bk_strength, vol_ratio, ret_5d,
                              ret_20d, volatility, atr_ratio,
                              0, 0, 0, 0, 0, 0, 0, 0, 0]])  # 6 features + 9 ticker dummies
prob = model.predict_proba(X_input)[0][1]

st.metric('Predicted Success Probability', f'{prob:.1%}')
if prob >= 0.55:
    st.success('Signal quality: STRONG — consider entering position')
elif prob >= 0.40:
    st.warning('Signal quality: MODERATE — wait for confirmation')
else:
    st.error('Signal quality: WEAK — avoid this breakout')
