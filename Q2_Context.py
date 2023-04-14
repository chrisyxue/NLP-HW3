"""
Use Contexted Features Encoded by BERT
"""
import random
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Tokenize and remove stopwords
    words = [word for word in nltk.word_tokenize(text.lower()) if word.isalnum() and word not in stop_words]
    
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

# Load the data
dataset = load_dataset("imdb")
train_data, test_data = dataset["train"], dataset["test"]

# Preprocess the data
train_texts = [preprocess_text(text) for text in train_data["text"]]
train_labels = train_data["label"]
test_texts = [preprocess_text(text) for text in test_data["text"]]
test_labels = test_data["label"]

# Initialize BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
model = AutoModel.from_pretrained(bert_model_name)

def get_bert_embeddings(texts):
    embeddings = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].detach().numpy().flatten())

    return np.array(embeddings)

# Get BERT embeddings for the text data
train_vectors = get_bert_embeddings(train_texts)
test_vectors = get_bert_embeddings(test_texts)

classifiers = [
    ("SVM", LinearSVC()),
    ("GBDT", GradientBoostingClassifier()),
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Random Forest", RandomForestClassifier()),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Decision Tree", DecisionTreeClassifier())
]

for name, classifier in classifiers:
    # Train the classifier
    classifier.fit(train_vectors, train_labels)

    # Test the classifier
    predictions = classifier.predict(test_vectors)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"{name} accuracy: {accuracy:.2f}")