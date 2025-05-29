# generate
import torch
import helper_functions
import tiktoken
from gpt2 import GPT, GPTConfig
from safetensors.torch import load_file

# ------------------------------------------------------------------------------
"""Generate Parameters"""
num_return_sequences = 1
max_length = 100
prompt = "It feels so empty without me"
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------------------
"""Load Pretrained Model"""
model = helper_functions.load_model(GPT, GPTConfig, "Models", "GPT2_final")
model.eval()
model.to(device)
# ------------------------------------------------------------------------------
"""Load Finetuned Model"""
# model = GPT(GPTConfig)
# state_dict = load_file("Models/Finetuned_Eminem_GPT2.pth")
# model.load_state_dict(state_dict, strict=False)
# model.lm_head.weight = model.transformer.wte.weight  # weight tying
# model.to(device=device)
# model.eval()
# ------------------------------------------------------------------------------
"""Get Tokenizer"""
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

# helper_functions.set_seeds(9090)

generated_tokens = model.generate(
    x, max_length=100, num_return_sequences=1, temperature=0.9, p=0.9
)

decoded = enc.decode(generated_tokens[0])

print("Generated text:", decoded)
