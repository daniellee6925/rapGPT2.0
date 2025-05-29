import torch
import helper_functions
import tiktoken
from torch.nn import functional as F
from gpt2 import GPT, GPTConfig
from safetensors.torch import load_file
import torch.quantization
import os
import time

# ------------------------------------------------------------------------------
"""Generate Parameters"""
num_return_sequences = 1
max_length = 100
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------------------
"""Load Models"""
original_model = helper_functions.load_model(
    GPT, GPTConfig, "Models", "Finetuned_Eminem_GPT2_v2"
)
quantized_model = helper_functions.load_model(
    GPT, GPTConfig, "Models", "Quantized_Finetuned_Eminem_GPT2_v2", strict=False
)

# ------------------------------------------------------------------------------
"""Qunatize and Save model"""
# quantized_model = torch.quantization.quantize_dynamic(
#     original_model,  # your model
#     {torch.nn.Linear},  # which layers to quantize
#     dtype=torch.float16,  # quantization type
# )
# helper_functions.save_model(
#     quantized_model, "Models", "Quantized_Finetuned_Eminem_GPT2_v2"
# )


# # ------------------------------------------------------------------------------
def get_model_size(path):
    """Measure Model Size"""
    # Save the model to a temporary file
    model_size = os.path.getsize(path) / (1024 * 1024)  # size in MB
    return model_size


def measure_inference_time(model, prompt, device, num_runs=10):
    """Measure inference time"""
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    model.to(device)
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.generate(
                tokens, max_length=100, num_return_sequences=1, temperature=0.9, p=0.9
            )
    avg_inference_time = (
        time.time() - start_time
    ) / num_runs  # time per inference in seconds
    return avg_inference_time


# ------------------------------------------------------------------------------
"""Compare Models"""
# Set device (CPU in this case)
device = torch.device("cpu")

# Sample prompt for inference
prompt = "It feels so empty without me"

# Measure model sizes
original_size = get_model_size("Models/Finetuned_Eminem_GPT2_v2.pth")
quantized_size = get_model_size("Models/Quantized_Finetuned_Eminem_GPT2_v2.pth")

# Measure inference times
original_inference_time = measure_inference_time(original_model, prompt, device)
quantized_inference_time = measure_inference_time(quantized_model, prompt, device)

# Print the comparison
print(f"Original model size: {original_size:.2f} MB")
print(f"Quantized model size: {quantized_size:.2f} MB")
print(f"Original inference time: {original_inference_time:.4f} seconds")
print(f"Quantized inference time: {quantized_inference_time:.4f} seconds")
