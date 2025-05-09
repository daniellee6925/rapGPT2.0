import torch
import helper_functions
import tiktoken
import time
from gpt2 import GPT, GPTConfig
from data_loader import DataloaderLite
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from lora import LoraModel, LoraConfig

# ------------------------------------------------------------------------------
"""Setup DDP if poosible"""
ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = (
    helper_functions.setup_distributed()
)

# ------------------------------------------------------------------------------
"""Hyperparameters"""
total_batch_size = 524288  # 2 ** 19 ~0.5M in number of tokens
B = 8
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, (
    "make sure total_batch_size is divisible by B*T*ddp_world_size"
)

# learning rates
max_lr = 2e-4
min_lr = max_lr * 0.1  # 10% of max lr
weight_decay = 0.01
learning_rate = 2e-4

# training steps
max_steps = 100
warmup_steps = int(max_steps * 0.10)  # 10% of max_steps
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

# LoRA Config
target_modules = ["c_attn", "c_proj", "c_fc"]
exclude_modules = ["ln_f"]
rank = 8
lora_alpha = 8
use_rslora = True
bias = "lora_only"
lora_dropout = 0.1
# ------------------------------------------------------------------------------
"""Set Seed"""
helper_functions.set_seeds(1337)
# ------------------------------------------------------------------------------
"""Load and tokenize dataset"""
train_loader = DataloaderLite(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train",
    file_path="Data/Eminem_lyrics.txt",
)
val_loader = DataloaderLite(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="val",
    file_path="Data/Eminem_lyrics.txt",
)
# ------------------------------------------------------------------------------
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulated steps: {grad_accum_steps}")
# use TF 32
torch.set_float32_matmul_precision("high")
# ------------------------------------------------------------------------------
"""Create Model"""
# Load pretrained GPT2 model and tokenizer
model = helper_functions.load_model(GPT, GPTConfig, "Models", "pretrained_gpt2_v1")
model = model.to(device)
# enc = tiktoken.get_encoding("gpt2")
enc = tiktoken.get_encoding("gpt2")

# user LoRA Model
lora_config = LoraConfig(
    rank=rank,
    target_modules=target_modules,
    exclude_modules=exclude_modules,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias=bias,
    use_rslora=use_rslora,
)

model = LoraModel(model, lora_config).to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model
# ------------------------------------------------------------------------------
"""Set Optimizer"""
optimizer = raw_model.configure_optimizers(
    weight_decay=weight_decay, learning_rate=learning_rate, device=device
)
# ------------------------------------------------------------------------------
"""Start Training Loop"""
for step in range(max_steps):
    t0 = time.time()

    # evaluate validation loss
    model.eval()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation Loss: {val_loss_accum.item():.4f}")

    # training loop
    model.train()  # model was set to eval above
    optimizer.zero_grad()
    loss_accum = 0.0

    # gradient accumulation to reduce GPU memory usage and stabilizes training
    # simulate larger batch size before applying update
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # use BF16 using autocast (mixed precision - matrix multiplies)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            # we have to scale the loss for gradient accumulation
            # addition of loss is SUM of the objective
            # we want MEAN. scale the loss by dividing my accum_steps
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            # process of sharing and averaging gradients across multiple GPUs
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()

    # all ranks with have average loss accum
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # Apply gradient clipping before optimizer step (prevent exploding gradients)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = helper_functions.get_lr(step, max_lr, warmup_steps, max_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()  # wait for GPU to finish work
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(
            f"step {step} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | lr: {lr:4f} | dt: {dt:.2f}ms | tokens_per_sec {tokens_per_sec:.2f}"
        )


# ------------------------------------------------------------------------------``
"""If Using DDP, destroy process group"""
if ddp:
    destroy_process_group()

# ------------------------------------------------------------------------------
"""Save model"""
(model).save_model(path="Models/Finetuned_Eminem_GPT2.pth", merge_weights=True)
