import math
import torch
import os
from torch.distributed import init_process_group
from pathlib import Path


def get_lr(it: int, max_lr: float, warmup_steps: int, max_steps: int):
    """Adjust learning rate with Warm Up and Cosine Decay"""
    min_lr = max_lr * 0.1  # 10% of max lr
    # linear warmup for warmup steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # if it > lr_decay_itrs, return min lr
    if it > max_steps:
        return min_lr
    # if in between, use cosine decay down to min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # coeff starts at 1 and goes to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup_distributed():
    """Sets up distributed data parallel (DDP) configuration and device assignment.

    Returns:
        ddp (bool): Whether the run is using Distributed Data Parallel (DDP).
        ddp_rank (int): Global rank of the current process.
        ddp_local_rank (int): Local rank of the current process on the node.
        ddp_world_size (int): Total number of processes in the DDP run.
        master_process (bool): Whether the current process is the master (rank 0).
        device (str): Device identifier to be used for torch operations.
    """
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a DDP run?

    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend="nccl")

        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # rank 0 does logging/checkpointing

    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        # Auto-detect device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves a PyTorch model to a target directory

    Args:
        model: PyTorch model to save
        traget_dir: directory for saving the model
        model_name: name of model
    """
    # create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # set file name
    model_filename = model_name + ".pth"
    config_filename = model_name + "_config.pth"

    model_save_path = target_dir_path / model_filename
    config_save_path = target_dir_path / config_filename

    # save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
    torch.save(obj=model.config, f=config_save_path)


def load_model(
    model_class: torch.nn.Module, config_class, target_dir: str, model_name: str
):
    """
    Loads a PyTorch model and its configuration from a target directory.

    Args:
        model_class: The model class to instantiate (e.g., GPT)
        config_class: The config class used to construct the model (e.g., GPTConfig)
        target_dir: Directory where the model and config are saved
        model_name: Base name of the model files (without .pth extension)

    Returns:
        model: The PyTorch model with loaded weights
    """
    target_dir_path = Path(target_dir)
    model_path = target_dir_path / f"{model_name}.pth"
    config_path = target_dir_path / f"{model_name}_config.pth"

    # Mark config_class as safe for torch.load
    torch.serialization.add_safe_globals([config_class])

    # Load the config object
    config = torch.load(config_path, weights_only=False)

    # Instantiate the model with the loaded config
    model = model_class(config)

    # Load model weights
    model.load_state_dict(torch.load(model_path))

    print(f"[INFO] Loaded model from: {model_path}")
    return model


"""
# check if weights are loaded correctly
from transformers import GPT2LMHeadModel
hf_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
hf_gpt2.to("cuda")
sd_hf = model.state_dict()
print(sd_hf["transformer.h.0.attn.c_attn.weight"].shape)

transposed = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]
# Assume `model` is your model and `gpt2_model` is the pretrained GPT-2
for (name1, param1), (name2, param2) in zip(
    model.named_parameters(), hf_gpt2.named_parameters()
):
    if any(name1.endswith(w) for w in transposed):
        param1_to_compare = param1.t()  # Transpose the parameter if it matches
    else:
        param1_to_compare = param1  # Use original parameter
    if not torch.allclose(param1_to_compare, param2, atol=1e-6):
        print(f"🚨 Mismatch in {name1}")
    else:
        print(f"✅ {name1} matches!")
"""
