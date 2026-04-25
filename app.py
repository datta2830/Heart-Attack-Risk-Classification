import pickle
import streamlit as st
import pandas as pd
import numpy as np

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open('rf_model.pkl', 'rb'))


# =========================
# TITLE
# =========================
st.title("Heart Attack Risk Classification App ❤️")

st.write("Enter patient details below to predict risk of heart attack.")

# =========================
# INPUT FIELDS
# =========================
age = st.number_input('Age', min_value=20, max_value=100, value=25)
restingbp = st.number_input('RestingBP', min_value=0, max_value=300, value=100)
cholesterol = st.number_input('Cholesterol', min_value=0, max_value=700, value=140)
fastingbs = st.selectbox('FastingBS', (0, 1))
maxhr = st.number_input('MaxHR', min_value=60, max_value=250, value=140)
oldpeak = st.number_input('Oldpeak', min_value=-3.0, max_value=6.6, value=1.0)

gender = st.selectbox('Gender', ('M', 'F'))
chestpaintype = st.selectbox('ChestPainType', ('ATA', 'NAP', 'ASY', 'TA'))
restingecg = st.selectbox('RestingECG', ('Normal', 'ST', 'LVH'))
exerciseangina = st.selectbox('ExerciseAngina', ('N', 'Y'))
st_slope = st.selectbox('ST_Slope', ('Up', 'Flat', 'Down'))

# =========================
# ENCODING
# =========================
Exercise_Angina = 1 if exerciseangina == 'Y' else 0

Sex_F = 1 if gender == 'F' else 0
Sex_M = 1 if gender == 'M' else 0

Chest_PainType_dict = {'ASY': 3, 'NAP': 2, 'ATA': 1, 'TA': 0}
Chest_PainType = Chest_PainType_dict[chestpaintype]

Resting_ECG_dict = {'Normal': 0, 'LVH': 1, 'ST': 2}
Resting_ECG = Resting_ECG_dict[restingecg]

ST_Slope_dict = {'Down': 0, 'Up': 1, 'Flat': 2}
ST_Slope = ST_Slope_dict[st_slope]

# =========================
# CREATE DATAFRAME
# =========================
input_features = pd.DataFrame([[
    age,
    restingbp,
    cholesterol,
    fastingbs,
    maxhr,
    oldpeak,
    Exercise_Angina,
    Sex_F,
    Sex_M,
    Chest_PainType,
    Resting_ECG,
    ST_Slope
]], columns=[
    'Age',
    'RestingBP',
    'Cholesterol',
    'FastingBS',
    'MaxHR',
    'Oldpeak',
    'Exercise_Angina',
    'Sex_F',
    'Sex_M',
    'Chest_PainType',
    'Resting_ECG',
    'st_Slope'
])

# =========================
# PREDICTION
# =========================
if st.button('Predict'):
    prediction = model.predict(input_features)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Attack!")
    else:
        st.success("✅ Low Risk of Heart Attack")
