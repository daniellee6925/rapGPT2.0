# generate
import torch
import helper_functions
import tiktoken
from torch.nn import functional as F
from gpt2 import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

model = helper_functions.load_model(GPT, GPTConfig, "Models", "pretrained_gpt2_v2")
model.eval()
model.to(device)


model.eval()
torch.set_grad_enabled(False)

# Initial input tokens (e.g. "cute boy")
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("cute boy")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
tokens = tokens.unsqueeze(0).repeat(1, 1)  # (5, 8)
x = tokens.to(device)
T = x.size(1)

# Initialize past_kv
past_kv = None

print("Verifying KV cache...")

# Loop over each position in the prompt + generation
for i in range(10):  # check 10 steps max
    # Input token: either full (uncached) or last (cached)
    x_cached = x[:, -1:] if i > 0 else x
    position_ids_cached = (
        torch.tensor([[x.size(1) - 1]], device=x.device)
        if i > 0
        else torch.arange(0, T, device=x.device).unsqueeze(0)
    )

    with torch.no_grad():
        # With cache
        logits_cached, _, past_kv = model(
            x_cached, position_ids=position_ids_cached, past_kv=past_kv, use_cache=True
        )

        # Without cache: pass full input every time
        position_ids_full = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        logits_full, _ = model(x, position_ids=position_ids_full, use_cache=False)

    # Get logits at last position
    logit_c = logits_cached[0, -1]
    logit_f = logits_full[0, -1]

    # Compare outputs
    diff = torch.abs(logit_c - logit_f).max().item()
    print(f"Step {i}: max logit diff = {diff:.6f}")

    if diff > 1e-4:
        print("Logits diverge — KV cache implementation has an issue!")
        break

    # Generate next token (sampling or argmax — doesn't matter here)
    next_token = torch.argmax(logit_f, dim=-1).unsqueeze(0).unsqueeze(0)  # shape (1,1)
    x = torch.cat((x, next_token), dim=1)

else:
    print("KV caching verified! Outputs match without caching.")
