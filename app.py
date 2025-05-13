from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the model and tokenizer (replace with the path to your custom model)
model_name = "Models/custom_model"  # Replace with your model path
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


@app.route("/generate", methods=["POST"])
def generate_lyrics():
    data = request.get_json()
    prompt = data.get("prompt", "")

    # Tokenize and generate text
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"lyrics": generated_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
