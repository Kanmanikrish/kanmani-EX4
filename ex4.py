import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def main():
    st.title("Tennis Data Classifier")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("The first 5 rows of data:")
        st.write(data.head())

        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]

        # Convert categorical data to numerical
        for col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        le = LabelEncoder()
        y = le.fit_transform(y)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train classifier
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.write(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()

