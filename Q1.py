"""
Tutorial for Hugging Face (Replace DistilBertTokenizerFast) 
"""

import random
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

# Load Yelp dataset
dataset = load_dataset("yelp_review_full")

# Select 1000 samples for training and 1000 samples for testing
train_data = dataset["train"].shuffle(seed=42).select(range(1000))
test_data = dataset["test"].shuffle(seed=42).select(range(1000))

# Initialize tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Tokenize the text data
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
test_data = test_data.map(tokenize, batched=True, batch_size=len(test_data))
train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
trainer.evaluate()


# from datasets import load_dataset
# import torch
# dataset = load_dataset("yelp_review_full")
# # dataset["train"][100]

# # AutoTokenizer
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # dataset
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# # Load Model
# from transformers import AutoModelForSequenceClassification
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model.to(device)


# from transformers import TrainingArguments
# training_args = TrainingArguments(output_dir="test_trainer")

# # evaluate
# import numpy as np
# import evaluate
# metric = evaluate.load("accuracy")
# from sklearn.metrics import accuracy_score

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


# # Training
# from transformers import TrainingArguments, Trainer
# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",
#                                 num_train_epochs=2,
#                                 per_device_train_batch_size=4,
#                                 per_device_eval_batch_size=4,
#                                 logging_dir="./logs")

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=lambda pred: {"accuracy": accuracy_score(pred.label_ids, np.argmax(pred.predictions, axis=1))}
# )

# trainer.train()