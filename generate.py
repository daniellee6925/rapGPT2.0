# generate
import torch
import helper_functions
import tiktoken
from torch.nn import functional as F
from gpt2 import GPT, GPTConfig
from safetensors.torch import load_file

# ------------------------------------------------------------------------------
"""Generate Parameters"""
num_return_sequences = 5
max_length = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------------------
"""Load Pretrained Model"""
# model = helper_functions.load_model(GPT, GPTConfig, "Models", "pretrained_gpt2_v2")
# model.eval()
# model.to(device)
# ------------------------------------------------------------------------------
"""Load Finetuned Model"""
model = GPT(GPTConfig)
state_dict = load_file("Models/Finetuned_by_artists_GPT2.pth")
model.load_state_dict(state_dict, strict=False)
model.lm_head.weight = model.transformer.wte.weight  # weight tying
model.to(device=device)
model.eval()

prompt = "It feels so empty"
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

helper_functions.set_seeds(1337)


past_kv = None  # initialize cache

while x.size(1) < max_length:
    with torch.no_grad():
        if past_kv is None:
            # First forward pass: input the full prompt
            position_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
            logits, _ = model(x, position_ids=position_ids, use_cache=False)
            # logits, _, past_kv = model(x, position_ids=position_ids, use_cache=True)
        else:
            # Subsequent passes: input only the last generated token
            position_ids = torch.tensor([[x.size(1)]], device=x.device)
            logits, _ = model(
                x[:, -1:], position_ids=position_ids, past_kv=past_kv, use_cache=False
            )

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
