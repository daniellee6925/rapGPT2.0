import torch
import helper_functions
import tiktoken
from torch.nn import functional as F
from gpt2 import GPT, GPTConfig
from safetensors.torch import load_file
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------
"""Load Pretrained Model"""
# model = helper_functions.load_model(GPT, GPTConfig, "Models", "pretrained_gpt2_v1")
# model.eval()
# model.to(device)
# ------------------------------------------------------------------------------
"""Load Finetuned Model"""
model = GPT(GPTConfig)
state_dict = load_file("Models/Finetuned_Eminem_GPT2.pth")
model.load_state_dict(state_dict, strict=False)
model.lm_head.weight = model.transformer.wte.weight  # weight tying
model.to(device=device)
model.eval()
# ------------------------------------------------------------------------------
"""Get Tokenizer"""
enc = tiktoken.get_encoding("gpt2")
# ------------------------------------------------------------------------------
"""Function to Calculate Perplexity"""


def compute_perplexity(texts):
    total_loss = 0.0
    total_length = 0.0

    for text in texts:
        tokens = enc.encode(text)
        input = torch.tensor(tokens, dtype=torch.long)
        input = input.unsqueeze(0)  # add batch dim
        x = input.to(device)
        with torch.no_grad():
            logits, loss = model(idx=x, targets=x, use_cache=False)

        loss = loss.item()
        total_loss += loss * x.size(1)
        total_length += x.size(1)
        print(f"Loss for this input: {loss}")

    print(f"Total tokens: {total_length}")
    avg_nll = total_loss / total_length
    return math.exp(avg_nll)


# ------------------------------------------------------------------------------
"""Load Actual Eminem lyrics"""
with open("Data/small_data.txt", "r", encoding="utf-8") as f:
    eminem_lyrics = [line.strip() for line in f if line.strip()]

"""
# Load model-generated lyrics
with open("Data/eminem_generated.txt", "r", encoding="utf-8") as f:
    generated_lyrics = [line.strip() for line in f if line.strip()]
"""

# Calculate perplexity
real_ppl = compute_perplexity(eminem_lyrics)
# gen_ppl = compute_perplexity(generated_lyrics)

print(f"Perplexity on real Eminem lyrics: {real_ppl:.2f}")
# print(f"Perplexity on generated lyrics: {gen_ppl:.2f}")
