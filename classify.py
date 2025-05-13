from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("Models/eminem_classifier_model")
model = AutoModelForSequenceClassification.from_pretrained(
    "Models/eminem_classifier_model"
)
model = model.to("cuda")


def predict_eminem_style(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        "cuda"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    return probs[:, 1].tolist()  # Eminem class probability


generated_lyrics = [
    "Snap back to reality, oh there goes gravity...",
    "I’m just a product of Slick Rick and Onyx, told 'em lick the balls...",
    "i just lay here and watch tv or sit and listen to just about nothing i put out the whitest school in town put out the worst",
    "whether it's on the steps getting to pepp's garden hose stuck in the studio sittin' back with nate",
]

# compute scores
scores = predict_eminem_style(generated_lyrics)

for line, score in zip(generated_lyrics, scores):
    print(f"{score:.2f} – {line}")
