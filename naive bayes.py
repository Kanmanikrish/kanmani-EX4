import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title of the app
st.title('Tennis Play Predictor')

# File uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("The first 5 values of the data are:")
    st.write(data.head())
    
    # Extract Train data (features) and Train output (target)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    st.write("The first 5 values of the training data (features) are:")
    st.write(X.head())
    
    st.write("The first 5 values of the training output (target) are:")
    st.write(y.head())
    
    # Convert categorical features to numerical values
    label_encoders = {}
    
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    st.write("The transformed training data is:")
    st.write(X.head())
    
    # Convert the target column to numerical values
    le_play_tennis = LabelEncoder()
    y = le_play_tennis.fit_transform(y)
    
    st.write("The transformed training output is:")
    st.write(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Train the Gaussian Naive Bayes model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predict on the test set and calculate accuracy
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"Accuracy is: {accuracy:.2f}")

# Instructions to run the Streamlit app in the terminal:
# streamlit run app.py
