import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title of the app
st.title('Tennis Play Predictor')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("The first 5 values of data are:")
    st.write(data.head())
    
    # Extract Train data and Train output
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    
    st.write("The first 5 values of train data are:")
    st.write(X.head())
    
    st.write("The first 5 values of train output are:")
    st.write(y.head())
    
    # Convert categorical features to numerical
    le_outlook = LabelEncoder()
    X['Outlook'] = le_outlook.fit_transform(X['Outlook'])
    
    le_Temperature = LabelEncoder()
    X['Temperature'] = le_Temperature.fit_transform(X['Temperature'])
    
    le_Humidity = LabelEncoder()
    X['Humidity'] = le_Humidity.fit_transform(X['Humidity'])
    
    le_Windy = LabelEncoder()
    X['Windy'] = le_Windy.fit_transform(X['Windy'])
    
    st.write("Now the Train data is:")
    st.write(X.head())
    
    le_PlayTennis = LabelEncoder()
    y = le_PlayTennis.fit_transform(y)
    
    st.write("Now the Train output is:")
    st.write(y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    # Train the Gaussian Naive Bayes model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"Accuracy is: {accuracy:.2f}")

# To run the Streamlit app, you can use the command in the terminal:
# streamlit run app.py
