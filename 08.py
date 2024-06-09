import streamlit as st
import numpy as np
import pandas as pd

def load_iris_data():
    # Load the Iris dataset from a CSV file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    iris_data = pd.read_csv(url, header=None, names=columns)
    iris_data["target"] = iris_data["target"].astype('category').cat.codes
    return iris_data

def split_train_test(data, test_size=0.25, random_state=0):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k=1):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train.iloc[i], x_test)
        distances.append((distance, y_train.iloc[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    votes = [neighbor[1] for neighbor in neighbors]
    prediction = max(set(votes), key=votes.count)
    return prediction

def main():
    st.title("BYTES BRIGADE")
    st.title("K-Nearest Neighbors Classifier on Iris Dataset")

    # Load Iris dataset
    iris_data = load_iris_data()
    train_data, test_data = split_train_test(iris_data)
    X_train, y_train = train_data.drop("target", axis=1), train_data["target"]
    X_test, y_test = test_data.drop("target", axis=1), test_data["target"]
    
    # Display the dataset
    st.subheader("Iris Dataset")
    iris_data['target_name'] = iris_data['target'].apply(lambda x: ["setosa", "versicolor", "virginica"][x])
    st.write(iris_data)
    
    # Predict and display results
    st.subheader("Predictions on Test Data")
    results = []
    for i in range(len(X_test)):
        x = X_test.iloc[i]
        prediction = knn_predict(X_train, y_train, x, k=1)
        results.append({
            "Target": y_test.iloc[i],
            "Target Name": ["setosa", "versicolor", "virginica"][y_test.iloc[i]],
            "Predicted": prediction,
            "Predicted Name": ["setosa", "versicolor", "virginica"][prediction]
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
