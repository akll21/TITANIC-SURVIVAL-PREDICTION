#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[11]:


# Create a Streamlit app
st.title("Titanic Survival Prediction")


# In[13]:


model = pickle.load(open('log_deploy.pkl','rb'))


# In[14]:


# Set up user inputs
st.subheader("Enter Passenger Details:")
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.number_input("Age", min_value=0, max_value=100)
SibSp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=10)
Parch = st.number_input("Number of Parents/Children", min_value=0, max_value=10)
Fare = st.number_input("Fare", min_value=0, max_value=1000)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])


# In[15]:


# Convert user inputs to a DataFrame
user_input = pd.DataFrame({
    "Pclass": [Pclass],
    "Sex": [0 if Sex == "Male" else 1],
    "Age": [Age],
    "SibSp": [SibSp],
    "Parch": [Parch],
    "Fare": [Fare],
    "Embarked": [0 if Embarked == "S" else 1 if Embarked == "C" else 2]
})


# In[16]:


# Make predictions
prediction = model.predict(user_input)

# Display the prediction
st.subheader("Prediction:")
if prediction[0] == 0:
    st.write("The passenger is unlikely to survive.")
else:
    st.write("The passenger is likely to survive.")

# Display the probability of survival
probability = model.predict_proba(user_input)[0][1]
st.subheader("Probability of Survival:")
st.write(f"The probability of survival is {probability:.2f}")


# In[ ]:





# In[ ]:





# In[ ]:




