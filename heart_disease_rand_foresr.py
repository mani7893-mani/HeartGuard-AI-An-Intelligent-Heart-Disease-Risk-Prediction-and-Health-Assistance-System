import streamlit as st
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
data=pd.read_csv("framingham (1).csv")
data.drop(["education","cigsPerDay","BMI"],axis=1,inplace=True)
x=data[['male', 'age', 'currentSmoker', 'BPMeds', 'prevalentStroke',
       'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'heartRate',
       'glucose']]
y=data["TenYearCHD"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

pred=model.predict(x_test)
accuracy=accuracy_score(y_test,pred)

page=st.sidebar.radio("Navigation",["prediction","AI helpe"])
if page=="prediction":
    st.title("Coronary Heart Disease (CHD) within 10 years")
    male = st.selectbox("Gender", ["Male", "Female"])
    male = 1 if male == "Male" else 0

    age = st.number_input("Enter your age", max_value=120, min_value=1)

    currentSmoker = st.selectbox("Current smoker", ["Yes", "No"])
    currentSmoker = 1 if currentSmoker == "Yes" else 0

    BPMeds = st.selectbox("Taking blood pressure medication?", ["Yes", "No"])
    BPMeds = 1 if BPMeds == "Yes" else 0

    prevalentStroke = st.selectbox("Have you had a stroke previously?", ["Yes", "No"])
    prevalentStroke = 1 if prevalentStroke == "Yes" else 0

    prevalentHyp = st.selectbox("Do you have hypertension?", ["Yes", "No"])
    prevalentHyp = 1 if prevalentHyp == "Yes" else 0

    diabetes = st.selectbox("Do you have diabetes?", ["Yes", "No"])
    diabetes = 1 if diabetes == "Yes" else 0

    totChol = st.number_input("Enter total cholesterol level", min_value=0.0)

    sysBP = st.number_input("Enter systolic blood pressure", min_value=0.0)

    diaBP = st.number_input("Enter diastolic blood pressure", min_value=0.0)

    heartRate = st.number_input("Enter heart rate", min_value=0.0)

    glucose = st.number_input("Enter blood glucose level", min_value=0.0)

    st.markdown(f"**our model accuracy is {accuracy *100}%**")
    button=st.button("predict")
    if button:
        pre=model.predict([[male, age, currentSmoker, BPMeds, prevalentStroke,
        prevalentHyp, diabetes, totChol, sysBP, diaBP, heartRate,
        glucose]])
        if pre[0] == 1: st.error("The model predicts a higher risk of CHD.")
        else: st.success("The model predicts a lower risk of CHD.")
        st.markdown(f"if u want any AI suggesions our AI model is in next winow")


client=genai.Client(api_key=os.getenv("API"))

if page=="AI helpe":
    st.title("AI health suggestions")
    input=st.text_area("enter your quary ")
    b=st.button("Aske AI")
    if b:
        response=client.models.generate_content(model="gemini-3.1-flash-lite",contents=input)
        st.write(response.text)
    