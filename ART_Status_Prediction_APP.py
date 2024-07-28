import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

st.write("""
# ART Intake Status Prediction App

This app predicts the **ART Intake Status**!
""")
st.write('---')

st.image(r'C:\Users\hp\Downloads\Current-State-of-HIV-and-AIDS-Treatment_pv.jpg',
          caption='HIV/AIDS ART Intake Prediction', use_column_width=True)

# Load the ART Dataset
ART_Data = pd.read_csv(r'C:\Users\hp\Downloads\ART_data.xls')
X = ART_Data.drop(columns='CurrentARTStatus')
Y = ART_Data['CurrentARTStatus'] 

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    current_Age = st.sidebar.slider('current_Age', float(X.current_Age.min()), float(X.current_Age.max()), float(X.current_Age.mean()))
    Age_At_Start = st.sidebar.slider('Age_At_Start', float(X.Age_At_Start.min()), float(X.Age_At_Start.max()), float(X.Age_At_Start.mean()))
    DaysOfARVRefill = st.sidebar.slider('DaysOfARVRefill', float(X.DaysOfARVRefill.min()), float(X.DaysOfARVRefill.max()), float(X.DaysOfARVRefill.mean()))
    RegimenAtARTStart = st.sidebar.slider('RegimenAtARTStart', float(X.RegimenAtARTStart.min()), float(X.RegimenAtARTStart.max()), float(X.RegimenAtARTStart.mean()))
    CurrentARTRegimen = st.sidebar.slider('CurrentARTRegimen', float(X.CurrentARTRegimen.min()), float(X.CurrentARTRegimen.max()), float(X.CurrentARTRegimen.mean()))
    CurrentWeight_Kg = st.sidebar.slider('CurrentWeight_Kg', float(X.CurrentWeight_Kg.min()), float(X.CurrentWeight_Kg.max()), float(X.CurrentWeight_Kg.mean()))
    CurrentHeight_cm = st.sidebar.slider('CurrentHeight_cm', float(X.CurrentHeight_cm.min()), float(X.CurrentHeight_cm.max()), float(X.CurrentHeight_cm.mean()))
    Sex_Female = st.sidebar.slider('Sex_Female', float(X.Sex_Female.min()), float(X.Sex_Female.max()), float(X.Sex_Female.mean()))
    Sex_Male = st.sidebar.slider('Sex_Male', float(X.Sex_Male.min()), float(X.Sex_Male.max()), float(X.Sex_Male.mean()))
    Biometric_Status_No = st.sidebar.slider('Biometric_Status_No', float(X.Biometric_Status_No.min()), float(X.Biometric_Status_No.max()), float(X.Biometric_Status_No.mean()))
    Biometric_Status_Yes = st.sidebar.slider('Biometric_Status_Yes', float(X.Biometric_Status_Yes.min()), float(X.Biometric_Status_Yes.max()), float(X.Biometric_Status_Yes.mean()))
    ViralLoadIndication_Initial = st.sidebar.slider('ViralLoadIndication_Initial', float(X.ViralLoadIndication_Initial.min()), float(X.ViralLoadIndication_Initial.max()), float(X.ViralLoadIndication_Initial.mean()))
    ViralLoadIndication_Normal_priority_status = st.sidebar.slider('ViralLoadIndication_Normal_priority_status', float(X['ViralLoadIndication_Normal_priority_status'].min()), float(X['ViralLoadIndication_Normal_priority_status'].max()), float(X['ViralLoadIndication_Normal_priority_status'].mean()))
    Appointment_Status_Active_With_Drugs = st.sidebar.slider('Appointment_Status_Active_With_Drugs', float(X['Appointment_Status_Active_With_Drugs'].min()), float(X['Appointment_Status_Active_With_Drugs'].max()), float(X['Appointment_Status_Active_With_Drugs'].mean()))
    Appointment_Status_LTFU = st.sidebar.slider('Appointment_Status_LTFU', float(X.Appointment_Status_LTFU.min()), float(X.Appointment_Status_LTFU.max()), float(X.Appointment_Status_LTFU.mean()))
    Appointment_Status_Missed_Appointment= st.sidebar.slider('Appointment_Status_Missed_Appointment', float(X['Appointment_Status_Missed_Appointment'].min()), float(X['Appointment_Status_Missed_Appointment'].max()), float(X['Appointment_Status_Missed_Appointment'].mean()))
    data = {'current_Age': current_Age,
            'Age_At_Start': Age_At_Start,
            'DaysOfARVRefill': DaysOfARVRefill,
            'RegimenAtARTStart': RegimenAtARTStart,
            'CurrentARTRegimen': CurrentARTRegimen,
            'CurrentWeight_Kg': CurrentWeight_Kg,
            'CurrentHeight_cm': CurrentHeight_cm,
            'Sex_Female': Sex_Female,
            'Sex_Male': Sex_Male,
            'Biometric_Status_No': Biometric_Status_No,
            'Biometric_Status_Yes': Biometric_Status_Yes,
            'ViralLoadIndication_Initial': ViralLoadIndication_Initial,
            'ViralLoadIndication_Normal_priority_status': ViralLoadIndication_Normal_priority_status,
            'Appointment_Status_Active_With_Drugs': Appointment_Status_Active_With_Drugs,
            'Appointment_Status_LTFU': Appointment_Status_LTFU,
            'Appointment_Status_Missed_Appointment': Appointment_Status_Missed_Appointment}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Load the pre-trained model
with open('ART_RF_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Apply Model to Make Prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.header('Prediction of ART Intake Status')
st.write(prediction)
st.write('---')

# Show prediction probabilities for each class
st.header('Prediction Probabilities')
st.write(pd.DataFrame(prediction_proba, columns=model.classes_))
st.write('---')