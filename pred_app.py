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
ART_Data = pd.read_csv(r'C:\Users\hp\Downloads\ART Project\ART-Status-Intake-Prediction-Analysis\new_ART_data.csv')
X = ART_Data.drop(columns='CurrentARTStatus')
Y = ART_Data['CurrentARTStatus'] 

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    # Continuous Features
    Days_To_Schedule = st.sidebar.slider('Days_To_Schedule', int(X.Days_To_Schedule.min()), int(X.Days_To_Schedule.max()), int(X.Days_To_Schedule.mean()))
    CurrentViralLoad = st.sidebar.slider('CurrentViralLoad', float(X.CurrentViralLoad.min()), float(X.CurrentViralLoad.max()), float(X.CurrentViralLoad.mean()))
    DaysOfARVRefill = st.sidebar.slider('DaysOfARVRefill', float(X.DaysOfARVRefill.min()), float(X.DaysOfARVRefill.max()), float(X.DaysOfARVRefill.mean()))
    CurrentHeight_cm = st.sidebar.slider('CurrentHeight_cm', float(X.CurrentHeight_cm.min()), float(X.CurrentHeight_cm.max()), float(X.CurrentHeight_cm.mean()))
    CurrentWeight_Kg = st.sidebar.slider('CurrentWeight_Kg', float(X.CurrentWeight_Kg.min()), float(X.CurrentWeight_Kg.max()), float(X.CurrentWeight_Kg.mean()))
    Age_At_Start = st.sidebar.slider('Age_At_Start', int(X.Age_At_Start.min()), int(X.Age_At_Start.max()), int(X.Age_At_Start.mean()))
    current_Age = st.sidebar.slider('current_Age', int(X.current_Age.min()), int(X.current_Age.max()), int(X.current_Age.mean()))

    # Categorical Features (using selectbox for unique categorical options)
    Appointment_Status_LTFU = st.sidebar.selectbox('Appointment_Status_LTFU', X.Appointment_Status_LTFU.unique())
    Appointment_Status_Active_With_Drugs = st.sidebar.selectbox('Appointment_Status_Active_With_Drugs', X.Appointment_Status_Active_With_Drugs.unique())
    Biometric_Status_No = st.sidebar.selectbox('Biometric_Status_No', X.Biometric_Status_No.unique())
    CurrentARTRegimen = st.sidebar.selectbox('CurrentARTRegimen', X.CurrentARTRegimen.unique())
    Biometric_Status_Yes = st.sidebar.selectbox('Biometric_Status_Yes', X.Biometric_Status_Yes.unique())
    RegimenAtARTStart = st.sidebar.selectbox('RegimenAtARTStart', X.RegimenAtARTStart.unique())
    ViralLoadIndication_Initial = st.sidebar.selectbox('ViralLoadIndication_Initial', X.ViralLoadIndication_Initial.unique())
    ViralLoadIndication_Normal_priority_status = st.sidebar.selectbox('ViralLoadIndication_Normal_priority_status', X.ViralLoadIndication_Normal_priority_status.unique())
    Appointment_Status_Missed_Appointment = st.sidebar.selectbox('Appointment_Status_Missed_Appointment', X.Appointment_Status_Missed_Appointment.unique())

    # Categorical Features - Gender
    Sex_Female = st.sidebar.selectbox('Sex_Female', X.Sex_Female.unique())
    Sex_Male = st.sidebar.selectbox('Sex_Male', X.Sex_Male.unique())

    # Create a data dictionary with the features in the specified order
    data = {
        'Days_To_Schedule': Days_To_Schedule,
        'Appointment_Status_LTFU': Appointment_Status_LTFU,
        'CurrentViralLoad': CurrentViralLoad,
        'Appointment_Status_Active_With_Drugs': Appointment_Status_Active_With_Drugs,
        'Biometric_Status_No': Biometric_Status_No,
        'DaysOfARVRefill': DaysOfARVRefill,
        'CurrentHeight_cm': CurrentHeight_cm,
        'CurrentARTRegimen': CurrentARTRegimen,
        'Biometric_Status_Yes': Biometric_Status_Yes,
        'CurrentWeight_Kg': CurrentWeight_Kg,
        'RegimenAtARTStart': RegimenAtARTStart,
        'Age_At_Start': Age_At_Start,
        'current_Age': current_Age,
        'ViralLoadIndication_Initial': ViralLoadIndication_Initial,
        'ViralLoadIndication_Normal_priority_status': ViralLoadIndication_Normal_priority_status,
        'Appointment_Status_Missed_Appointment': Appointment_Status_Missed_Appointment,
        'Sex_Female': Sex_Female,
        'Sex_Male': Sex_Male
    }

    # Convert to dataframe
    features = pd.DataFrame(data, index=[0])
    return features




df = user_input_features()

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Load the pre-trained model
with open('new_ART_RF_Model1.pkl', 'rb') as f:
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