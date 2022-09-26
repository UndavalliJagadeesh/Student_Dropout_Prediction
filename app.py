# from html.entities import html5
# from pydoc import html
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import joblib

from code import interact
import streamlit as st
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

with st.container():
    st.title("Student Dropout Predictor")
    st.write("Prediction of a student whether he/she drops out from the education based on various factors")

with open('app.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


model = joblib.load('model.pkl')
le = joblib.load('encoder.pkl')

input_names = {'school' : 'Select School type', 'gender' : 'Gender', 'age' : "Age", 'address' : "Address",'famsize':"Family Size", 'Pstatus' : "Pstatus", 'Medu' : "Mother Education",'Fedu' :"Father Education",'Mjob' : "Mother Job", 'Fjob' : "Father Job",'reason' : "Reason",'guardian' :"Guardian",'traveltime' : "Travel Time(hrs)",'studytime' : "Study Time(hrs)",'failures' :"Failures",'schoolsup' : "School Support",'famsup' : "Family Support",'paid' : "Fee paid",'activities' : "Activities",'nursery':"Nursery",'higher' : "Higher Education??",'internet' : "Internet",'romantic' : "Romantic",'famrel' : "Family relatives",'freetime' :"Free Time(hrs)",'goout' : "Vacation/Go out Time(hrs)",'Dalc' : "Dalc",'Walc' : "Walc",'health' : "Health",'absences' : "Days absent"}
input_type = {'school' : ['',"MS","GP"], 'gender' :['',"M","F"], 'address' : ['',"R","U"],'famsize':['',"GT3","LE3"], 'Pstatus' : ['',"T","A"], 'Medu' : ['',0,1,2,3,4],'Fedu' : ['',0,1,2,3,4],'Mjob' : ['',"Teacher","at home","health","services","other"],'Fjob' : ['',"Teacher","at home","health","services","other"],'reason' : ['',"Reputation","Course","Home","other"],'guardian' : ['',"Father","Mother","Other"],'traveltime' : ['',1,2,3,4],'studytime' : ['',1,2,3,4],'failures' : ['',0,1,2,3],'schoolsup' : ['',"Yes","No"],'famsup' : ['',"Yes","No"],'paid' : ['',"Yes","No"],'activities' : ['',"Yes","No"],'nursery' : ['',"Yes","No"],'higher' : ['',"Yes","No"],'internet' : ['',"Yes","No"],'romantic' : ['',"Yes","No"],'famrel' : ['',1,2,3,4],'freetime' : ['',1,2,3,4],'goout' : ['',1,2,3,4],'Dalc' : ['',1,2,3,4],'Walc' : ['',1,2,3,4],'health' : ['',1,2,3,4]}
input_lst=[]

with st.form(key="my_form", clear_on_submit=True):
    selected_features_list = model.feature_names
    for i in selected_features_list:
        if i=='age':
            ele = st.slider("Age",15,22)
        elif i!='age' and i!='absences':
            ele = st.selectbox(input_names[i], input_type[i])
        elif i=="absences":
            ele = st.slider("Days absent",0,100)
        input_lst.append(ele)
    submitted = st.form_submit_button("Test")
# reload_btn = st.button('Test another')
if submitted:

    X_test_input_cols = list(model.feature_names)
    default_dict = {}
    for i in range(len(X_test_input_cols)):
        default_dict[X_test_input_cols[i]] = input_lst[i]

    X_input_test = pd.DataFrame(default_dict,index=[0])

    for name in X_test_input_cols:
        if X_input_test[name].dtype =='object':
            X_input_test[name] = le.fit_transform(X_input_test[name])

    y_input_pred = model.predict(X_input_test)
    if input_lst.count('')>0:
        st.error('Some inputs are missing')
    elif y_input_pred[0]==0:
        st.success('The Student will not dropout 😆😆😆')
    else:
        st.error('The Student will dropout 😭😭😭')
    
    # if reload_btn:
    #     st.experimental_rerun()


st.write('')
st.write('')
st.write("The source code can be found here 👉[🔗](https://www.github.com/UndavalliJagadeesh/Student_Dropout_Prediction)")
