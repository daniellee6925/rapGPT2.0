# generate
import torch
import helper_functions
import tiktoken
from torch.nn import functional as F
from gpt2 import GPT, GPTConfig

# ------------------------------------------------------------------------------
"""Generate Parameters"""
num_return_sequences = 5
max_length = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------------------
model = helper_functions.load_model(GPT, GPTConfig, "Models", "pretrained_gpt2")
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("cute boy")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

helper_functions.set_seeds(1337)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)  # (B, T, vocab_size)
        # take the logits at the last location
        logits = logits[:, -1, :]  # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (5, 50)
        # select a token from the top k probs
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to sequence
        x = torch.cat((x, xcol), dim=1)  # (B, i+1)
        # will have (5, 30) at the end of while loop

# print the results
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
