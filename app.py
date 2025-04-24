import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction App")

# Input form
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
sex_male = st.selectbox("Gender", ["Female", "Male"]) == "Male"
embarked = st.selectbox("Embarked Port", ["S", "Q", "C"])
embarked_Q = embarked == "Q"
embarked_S = embarked == "S"

# Predict
if st.button("Predict Survival"):
    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Sex_male': int(sex_male),
        'Embarked_Q': int(embarked_Q),
        'Embarked_S': int(embarked_S)
    }])
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {'Survived' if prediction else 'Did Not Survive'}")
