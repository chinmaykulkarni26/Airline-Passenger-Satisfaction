#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.model_selection import train_test_split

st.title('Model Deployment: SVM')

st.header('User Input Parameters')


type_dict = {'Business travel':0, 'Personal Travel':1}
class_dict = {'Business':0, 'Eco':1, 'Eco Plus':2}
satisfaction_dict = {'neutral or dissatisfied':0, 'satisfied':1}

type_list = ['Business travel', 'Personal Travel']
class_list = ['Business', 'Eco', 'Eco Plus']
satisfaction_list = ['neutral or dissatisfied', 'satisfied']


travels = st.selectbox(label = 'Select your type of travel:', options = type_list)
type_of_travel = type_dict[travels]

cl = st.selectbox(label = 'Select your prefence of the class of travel:', options = class_list)
class_ = class_dict[cl]

Flight_distance = st.number_input(' Insert the distance of the aircraft in KMs:')

Inflight_wifi_service = st.slider(' Select the rating of the inflight wifi service:', 0, 5)

Online_boarding = st.slider(' Select the the rating of the online boarding service:', 0, 5)

Seat_comfort = st.slider(' Select the rating of the seat comfort:', 0, 5)

Inflight_entertainment = st.slider(' Select the inflight entertainment service rating:', 0, 5)

On_board_service = st.slider(' Select the on-board service rating:', 0, 5)

Leg_room_service = st.slider(' Select the leg-room service rating:', 0, 5)


data = pd.read_csv('train.csv')
df = pd.DataFrame(data[['Type of Travel','Class','Flight Distance','Inflight wifi service','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','satisfaction']])
df = df.dropna()
c = ['Type of Travel', 'Class', 'satisfaction']
df[c] = df[c].apply(LabelEncoder().fit_transform)

X = df.iloc[:, 0:9]
y = df.iloc[:, 9]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=101)

svm = SVC()
model = svm.fit(X_train.values, y_train.values)

def satisfaction_prediction_svm(type_of_travel, class_, Flight_distance, Inflight_wifi_service, Online_boarding, Seat_comfort, Inflight_entertainment, On_board_service, Leg_room_service):
    prediction = model.predict([[type_of_travel, class_, Flight_distance, Inflight_wifi_service, Online_boarding, Seat_comfort, Inflight_entertainment, On_board_service, Leg_room_service]])
    
    return prediction

result = 0
st.button('Prediction')
result = satisfaction_prediction_svm(type_of_travel, class_, Flight_distance, Inflight_wifi_service, Online_boarding, Seat_comfort, Inflight_entertainment, On_board_service, Leg_room_service)

if result < 0.7:
    st.success('The customer is neutral or dissatisfied!')
else:
    result = 1
    st.success('The customer is satisfied!')

    
        

