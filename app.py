from flask import Flask, request, jsonify
import torch
import helper_functions
import tiktoken
from gpt2 import GPT, GPTConfig

app = Flask(__name__)

# ------------------------------------------------------------------------------
"""Generate Parameters"""
num_return_sequences = 1
max_length = 100
device = "cpu"
# ------------------------------------------------------------------------------
"""Load Model and Tokenizer"""
model = helper_functions.load_model(
    GPT, GPTConfig, "Models", "Finetuned_Eminem_GPT2_v2"
)
enc = tiktoken.get_encoding("gpt2")
model.eval()
model.to(device)


# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return """
        <form action="/generate" method="post" onsubmit="event.preventDefault(); generate();">
            <input type="text" id="prompt" placeholder="Enter prompt..." />
            <input type="number" id="temperature" step="0.1" value="0.9" />
            <input type="number" id="p" step="0.05" value="0.9" />
            <button type="submit">Generate</button>
        </form>
        <pre id="result"></pre>
        <script>
        async function generate() {
            const prompt = document.getElementById("prompt").value;
            const temperature = parseFloat(document.getElementById("temperature").value);
            const p = parseFloat(document.getElementById("p").value);
            const res = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt, temperature, p })
            });
            const data = await res.json();
            document.getElementById("result").innerText = data.lyrics;
        }
        </script>
    """


@app.route("/generate", methods=["POST"])
def generate_lyrics():
    data = request.get_json()
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.9))
    top_p = float(data.get("p", 0.9))

    tokens = enc.encode(prompt)
    tokens = (
        torch.tensor(tokens, dtype=torch.long)
        .unsqueeze(0)
        .repeat(num_return_sequences, 1)
        .to(device)
    )

    outputs = model.generate(
        tokens,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        p=top_p,
    )

    generated_text = enc.decode(outputs[0])
    return jsonify({"lyrics": generated_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
