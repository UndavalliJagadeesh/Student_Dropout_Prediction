from html.entities import html5
from pydoc import html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from code import interact
import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")
# components.html('<html><body><div class="header">Student Dropout Prediction</div></body></html>')


with open('app.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# input_dit = {'school' : 'st.selectbox("Select School type",("MS","GP"))', 'gender' : st.selectbox("Gender",("M","F")), 'age' : st.slider("Age",15,22), 'address' : st.selectbox("Address",("R","U")),'famsize':st.selectbox("Family Size",("GT3","LE3")), 'Pstatus' : st.selectbox("Pstatus",("T","A")), 'Medu' : st.selectbox("Mother Education",(0,1,2,3,4)),'Fedu' : st.selectbox("Father Education",(0,1,2,3,4)),'Mjob' : st.selectbox("Mother JOb",("Teacher","at home","health","services","other")),'Fjob' : st.selectbox("Father JOb",("Teacher","at home","health","services","other")),'reason' : st.selectbox("Reason",("Reputation","Course","Home","other")),'gaurdian' : st.selectbox("Gaurdian",("Father","Mother","Other")),'traveltime' : st.selectbox("Travel Time(hrs)",(1,2,3,4)),'studytime' : st.selectbox("Study Time(hrs)",(1,2,3,4)),'failures' : st.selectbox("failures",(0,1,2,3)),'schoolsup' : st.selectbox("School Support",("Yes","No")),'famsup' : st.selectbox("Family Support",("Yes","No")),'paid' : st.selectbox("Fee paid",("Yes","No")),'activities' : st.selectbox("Activities",("Yes","No")),'nursery' : st.selectbox("Nursery",("Yes","No")),'higher' : st.selectbox("Higher Education??",("Yes","No")),'internet' : st.selectbox("Internet",("Yes","No")),'romantic' : st.selectbox("Romantic",("Yes","No")),'famrel' : st.selectbox("Family relatives",(1,2,3,4)),'freetime' : st.selectbox("Free Time(hrs)",(1,2,3,4)),'goout' : st.selectbox("Vacation/Go out Time(hrs)",(1,2,3,4)),'Dalc' : st.selectbox("Dalc",(1,2,3,4)),'Walc' : st.selectbox("Walc",(1,2,3,4)),'health' : st.selectbox("Health",(1,2,3,4)),'absences' : st.slider("Days absent",0,100)}
input_names = {'school' : 'Select School type', 'gender' : 'Gender', 'age' : "Age", 'address' : "Address",'famsize':"Family Size", 'Pstatus' : "Pstatus", 'Medu' : "Mother Education",'Fedu' :"Father Education",'Mjob' : "Mother Job", 'Fjob' : "Father Job",'reason' : "Reason",'guardian' :"Guardian",'traveltime' : "Travel Time(hrs)",'studytime' : "Study Time(hrs)",'failures' :"failures",'schoolsup' : "School Support",'famsup' : "Family Support",'paid' : "Fee paid",'activities' : "Activities",'nursery':"Nursery",'higher' : "Higher Education??",'internet' : "Internet",'romantic' : "Romantic",'famrel' : "Family relatives",'freetime' :"Free Time(hrs)",'goout' : "Vacation/Go out Time(hrs)",'Dalc' : "Dalc",'Walc' : "Walc",'health' : "Health",'absences' : "Days absent"}
input_type = {'school' : ['',"MS","GP"], 'gender' :['',"M","F"], 'address' : ['',"R","U"],'famsize':['',"GT3","LE3"], 'Pstatus' : ['',"T","A"], 'Medu' : ['',0,1,2,3,4],'Fedu' : ['',0,1,2,3,4],'Mjob' : ['',"Teacher","at home","health","services","other"],'Fjob' : ['',"Teacher","at home","health","services","other"],'reason' : ['',"Reputation","Course","Home","other"],'guardian' : ['',"Father","Mother","Other"],'traveltime' : ['',1,2,3,4],'studytime' : ['',1,2,3,4],'failures' : ['',0,1,2,3],'schoolsup' : ['',"Yes","No"],'famsup' : ['',"Yes","No"],'paid' : ['',"Yes","No"],'activities' : ['',"Yes","No"],'nursery' : ['',"Yes","No"],'higher' : ['',"Yes","No"],'internet' : ['',"Yes","No"],'romantic' : ['',"Yes","No"],'famrel' : ['',1,2,3,4],'freetime' : ['',1,2,3,4],'goout' : ['',1,2,3,4],'Dalc' : ['',1,2,3,4],'Walc' : ['',1,2,3,4],'health' : ['',1,2,3,4]}
input_lst=[]
# with st.container():
#     school = st.selectbox("Select School type",(lst))
#     gender = st.selectbox("Gender",("M","F"))
#     age = st.slider("Age",15,22)
#     address = st.selectbox("Address",("R","U"))
#     famsize = st.selectbox("Family Size",("GT3","LE3"))
#     Pstatus = st.selectbox("Pstatus",("T","A"))
#     Medu = st.selectbox("Mother Education",(0,1,2,3,4))
#     Fedu = st.selectbox("Father Education",(0,1,2,3,4))
#     Mjob = st.selectbox("Mother JOb",("Teacher","at home","health","services","other"))
#     Fjob = st.selectbox("Father JOb",("Teacher","at home","health","services","other"))
#     reason = st.selectbox("Reason",("Reputation","Course","Home","other"))
#     gaurdian = st.selectbox("Gaurdian",("Father","Mother","Other"))
#     traveltime = st.selectbox("Travel Time(hrs)",(1,2,3,4))
#     studytime = st.selectbox("Study Time(hrs)",(1,2,3,4))
#     failures = st.selectbox("failures",(0,1,2,3))
#     schoolsup = st.selectbox("School Support",("Yes","No"))
#     famsup = st.selectbox("Family Support",("Yes","No"))
#     paid = st.selectbox("Fee paid",("Yes","No"))
#     activities = st.selectbox("Activities",("Yes","No"))
#     nursery = st.selectbox("Nursery",("Yes","No"))
#     higher = st.selectbox("Higher Education??",("Yes","No"))
#     internet = st.selectbox("Internet",("Yes","No"))
#     romantic = st.selectbox("Romantic",("Yes","No"))
#     famrel = st.selectbox("Family relatives",(1,2,3,4))
#     freetime = st.selectbox("Free Time(hrs)",(1,2,3,4))
#     goout = st.selectbox("Vacation/Go out Time(hrs)",(1,2,3,4))
#     Dalc = st.selectbox("Dalc",(1,2,3,4))
#     Walc = st.selectbox("Walc",(1,2,3,4))
#     health = st.selectbox("Health",(1,2,3,4))
#     absences = st.slider("Days absent",0,100)


data = pd.read_csv('dropout.csv')

le = LabelEncoder()
feature_names = data.columns.values

for name in feature_names:
  if data[name].dtype =='object':
    data[name] = le.fit_transform(data[name])

X = data.drop('dropout', axis=1)
y= data.dropout

# selecting the most important features

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = SelectFromModel(DecisionTreeClassifier())
model.fit(X_train, y_train)

model.get_support()
selected_feat= X_train.columns[(model.get_support())]

# st.write(selected_feat.values)
with st.container():
    st.title("Student Dropout Predictor")
    # st.subheader("By team CODE BUDDIES")
    st.write("Prediction of a student whether he/she drops out from the education based on various factors")
    
    
    for i in selected_feat.values:
        if i=='age':
            input_lst.append(st.slider("Age",15,22))
        elif i!='age' and i!='absences':
            input_lst.append(st.selectbox(input_names[i], input_type[i]))
        elif i=="absences":
            input_lst.append(st.slider("Days absent",0,100))

# st.write(input_lst)

df = pd.DataFrame(data=data, columns=selected_feat)
df_target = data['dropout']
# df1 = pd.concat([df, df_target], ignore_index=True, sort=False)
df = df.join(df_target, lsuffix='_caller', rsuffix='_other')

X = df.drop('dropout',axis=1)
y = df.dropout

#dealing with unbalanced data

# from imblearn.over_sampling import RandomOverSampler

# ros = RandomOverSampler()
# X, y = ros.fit_resample(X, y)


# #splitting the dataset


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# """> Training the models and Evaluating their performance

# *Since the target feature is categorical, the Machine learning models that are used to train and predict on this dataset should be of type Classification.*

# ### Random Forest Classifier

# *`Random Forest` is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.*
# """

model = RandomForestClassifier(n_estimators=13, criterion='gini',max_depth=10, max_features='auto')
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


# st.write("Random Forest Classifier\'s Accuracy :",round(accuracy_score(y_test, y_pred),4))
# st.write("Random Forest Classifier\'s F1 Score :",round(f1_score(y_test, y_pred),4))

X_test_input_cols = list(X.columns)
default_dict = {}
for i in range(len(X_test_input_cols)):
    default_dict[X_test_input_cols[i]] = input_lst[i] 

X_input_test = pd.DataFrame(default_dict,index=[0])

for name in X_test_input_cols:
  if X_input_test[name].dtype =='object':
    X_input_test[name] = le.fit_transform(X_input_test[name])

y_input_pred = model.predict(X_input_test)
if y_input_pred[0]==0:
    st.success('The Student will not dropout 😆😆😆')
else:
    st.error('The Student will dropout 😭😭😭')


st.write('')
st.write('')
st.write("The source code can be found here 👉[🔗](https://www.github.com/UndavalliJagadeesh/ADS_HACKATHON)")

