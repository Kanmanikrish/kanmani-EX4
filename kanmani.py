import streamlit as st
import pandas as pd
from scikit-learn.preprocessing import LabelEncoder
from scikit-learn.naive_bayes import GaussianNB
from scikit-learn.model_selection import train_test_split
from scikit-learn.metrics import accuracy_score

# Title of the app
st.title('Tennis Play Predictor')

# File uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("The first 5 values of the data are:")
    st.write(data.head())

    # Extract features (training data) and target variable (training output)
    features = data.iloc[:, :-1]  # Assuming the last column is the target
    target = data.iloc[:, -1]

    st.write("The first 5 values of the features are:")
    st.write(features.head())

    st.write("The first 5 values of the target variable are:")
    st.write(target.head())

    # Convert categorical features to numerical values
    label_encoders = {}

    for col in features.columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        label_encoders[col] = le  # Store encoders for later use (optional)

    st.write("The transformed features are:")
    st.write(features.head())

    # Convert the target variable to numerical values
    le_play_tennis = LabelEncoder()
    target = le_play_tennis.fit_transform(target)

    st.write("The transformed target variable is:")
    st.write(target)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=42)

    # Train the Gaussian Naive Bayes model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predict on the test set and calculate accuracy
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Accuracy is: {accuracy:.2f}")

# Instructions to run the Streamlit app in the terminal:
# streamlit run app.py

   
