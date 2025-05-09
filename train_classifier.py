from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Load your dataset (replace with your own CSV path)
dataset = load_dataset(
    "csv",
    data_files={
        "train": "Data/classifier_train.csv",
        "test": "Data/classifier_test.csv",
    },
)


# -------------------------------------------------------------------------------
"""Apply tokenization function to every function in dataset"""
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# Tokenize function, with padding (all seq in batch same length) and truncation to maximum input length
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# apply Tokenize function to by batches in dataset
dataset = dataset.map(tokenize, batched=True)


# Model
# faster and smaller BERT model trained on lowercase only.
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)


# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

model.save_pretrained("Models/eminem_classifier_model")
tokenizer.save_pretrained("Models/eminem_classifier_model")
