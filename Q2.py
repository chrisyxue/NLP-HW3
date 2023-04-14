"""
Compare SVM with BERT
"""
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the data
dataset = load_dataset("imdb")
# dataset = load_dataset("yelp_review_full")
train_data, test_data = dataset["train"], dataset["test"]

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# Tokenize the data
train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
test_data = test_data.map(tokenize, batched=True, batch_size=len(test_data))

# Set the format for PyTorch
train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=300,
    save_steps=300,
    evaluation_strategy="steps",
)
# Initialize the model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=lambda pred: {"accuracy": accuracy_score(pred.label_ids, np.argmax(pred.predictions, axis=1))},
)

# Train the model
trainer.train()


# Now, let's use a non-neural network model, e.g., SVM
train_texts, train_labels = dataset["train"]["text"], dataset["train"]["label"]
test_texts, test_labels = dataset["test"]["text"], dataset["test"]["label"]

# Vectorize the text data
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Train the SVM
svm = LinearSVC()
svm.fit(train_vectors, train_labels)

# Test the SVM
svm_predictions = svm.predict(test_vectors)
svm_accuracy = accuracy_score(test_labels, svm_predictions)

print(f"SVM accuracy: {svm_accuracy:.2f}")

# Compare the results
bert_accuracy = trainer.evaluate()["eval_accuracy"]
print(f"BERT accuracy: {bert_accuracy:.2f}")


"""
SVM with more Feature Engineering (FE)
"""
import random
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier

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

# Vectorize the text data using bi-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Train the SVM
svm = LinearSVC()
svm.fit(train_vectors, train_labels)

# Test the SVM
svm_predictions = svm.predict(test_vectors)
svm_accuracy = accuracy_score(test_labels, svm_predictions)

print(f"SVM with FE accuracy: {svm_accuracy:.2f}")


