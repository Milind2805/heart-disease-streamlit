import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
st.title("🫀 Heart Disease Prediction")
st.write("Enter the patient details below to predict heart disease risk.")
# Example for Age
age = st.slider("Age", min_value=20, max_value=100, value=50)

# Example for Sex
sex = st.selectbox("Sex", options=[0, 1], 
                   format_func=lambda x: "Female" if x==0 else "Male")
chp=st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
bp=st.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol=st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
FBS=st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
EKG=st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
maxHR=st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina=st.selectbox("Exercise Induced Angina", options=[0, 1])
st_depression=st.slider("ST Depression Induced by Exercise Relative to Rest", min_value = 0.0, max_value=10.0, value=1.0, step=0.1)
slope=st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
num_vessels=st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal=st.selectbox("Thalassemia", options=[0, 1, 2, 3]) 
if st.button("prediction"):
    input_data=np.array([[age,sex,chp,bp,chol,FBS,EKG,maxHR,exercise_angina,st_depression,slope,num_vessels,thal]])
    input_scaled=scaler.transform(input_data)
    prediction=model.predict(input_scaled)
    if prediction[0]==1:
        st.error("The patient is at risk of heart disease.")
    else:
        st.success("The patient is not at risk of heart disease.")
            