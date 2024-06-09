import streamlit as st
import numpy as np
import pandas as pd
from streamlit.datasets import load_iris
from atreamlit.model_selection import train_test_split

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k=1):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], x_test)
        distances.append((distance, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    votes = [neighbor[1] for neighbor in neighbors]
    prediction = max(set(votes), key=votes.count)
    return prediction

def main():
    st.title("BYTES BRIGADE")
    st.title("K-Nearest Neighbors Classifier on Iris Dataset")

    # Load Iris dataset
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)
    
    # Display the dataset
    st.subheader("Iris Dataset")
    iris_data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    iris_data['target'] = dataset.target
    iris_data['target_name'] = iris_data['target'].apply(lambda x: dataset.target_names[x])
    st.write(iris_data)
    
    # Predict and display results
    st.subheader("Predictions on Test Data")
    results = []
    for i in range(len(X_test)):
        x = X_test[i]
        prediction = knn_predict(X_train, y_train, x, k=1)
        results.append({
            "Target": y_test[i],
            "Target Name": dataset["target_names"][y_test[i]],
            "Predicted": prediction,
            "Predicted Name": dataset["target_names"][prediction]
        })
    
    results_df = pd.DataFrame(results)
    st.write(results_df)
    
    # Calculate and display model accuracy
    st.subheader("Model Accuracy")
    correct_predictions = sum(results_df["Target"] == results_df["Predicted"])
    accuracy = correct_predictions / len(results_df)
    st.write(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
