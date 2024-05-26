import streamlit as st
import pandas as pd
import numpy as np
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.vocabulary = set()
    
    def fit(self, X, y):
        for text, label in zip(X, y):
            words = preprocess_text(text)
            for word in words:
                self.class_word_counts[label][word] += 1
                self.class_counts[label] += 1
                self.vocabulary.add(word)
    
    def predict(self, X):
        predictions = []
        for text in X:
            words = preprocess_text(text)
            pos_score = np.log(self.class_counts[1] / sum(self.class_counts.values()))
            neg_score = np.log(self.class_counts[0] / sum(self.class_counts.values()))
            
            for word in words:
                if word in self.vocabulary:
                    pos_score += np.log((self.class_word_counts[1][word] + 1) / (self.class_counts[1] + len(self.vocabulary)))
                    neg_score += np.log((self.class_word_counts[0][word] + 1) / (self.class_counts[0] + len(self.vocabulary)))
            
            if pos_score > neg_score:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return np.array(predictions)

def main():
    st.title("Text Classification using Naive Bayes")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, names=['message', 'label'])
        st.write("The first 5 rows of data:")
        st.write(data.head())

        msg = data.message
        y = data.label.map({'pos': 1, 'neg': 0})

        # Split data into train and test sets
        split_ratio = 0.8
        indices = np.random.permutation(len(msg))
        train_size = int(len(msg) * split_ratio)
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        Xtrain, Xtest = msg.iloc[train_idx], msg.iloc[test_idx]
        ytrain, ytest = y.iloc[train_idx], y.iloc[test_idx]

        # Train classifier
        classifier = NaiveBayesClassifier()
        classifier.fit(Xtrain, ytrain)

        # Predict and evaluate
        y_pred = classifier.predict(Xtest)
        accuracy = np.mean(y_pred == ytest)

        st.write(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
