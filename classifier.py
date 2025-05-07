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
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


dataset = dataset.map(tokenize, batched=True)

# Model
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
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
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


def predict_eminem_style(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    return probs[:, 1].tolist()  # Eminem class probability


generated_lyrics = [
    "Snap back to reality, oh there goes gravity...",
    "I’m just a product of Slick Rick and Onyx, told 'em lick the balls...",
]

scores = predict_eminem_style(generated_lyrics)
for line, score in zip(generated_lyrics, scores):
    print(f"{score:.2f} – {line}")
