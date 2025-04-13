from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
# -----------------------------------------


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # Output is 3 * n_embd because it concatenates Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Linear layer to project the concatenated attention output back to the embedding size
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Create a causal mask (lower triangular matrix)
        # Shape: (1, 1, block_size, block_size) so it can broadcast during attention computation
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        # Save config values for number of heads and embedding size
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # B: batch size, T: sequence length, C: embedding dimension
        B, T, C = x.size()  # (B, T, 3 * C)

        # Project the input x to get concatenated Q, K, V tensors
        qkv = self.c_attn(x)

        # Split the last dimension into separate Q, K, V
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each has shape (B, T, C)

        # New shape: (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # FLASH ATTENTION
        # Apply scaled dot-product attention with causal masking
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # contiguous creates new memory in pytorch
        # transpose do not change the underlying memory.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Project back to original embedding size
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Expands the embedding size by 4x
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU Activation with tanh" approximation
        self.gelu = nn.GELU(approximate="tanh")
        # Project back to original embedding size
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)  # expand
        x = self.gelu(x)  # apply non-linear activation
        x = self.c_proj(x)  # shrink back
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # layer norm -> attn -> + residual connection
        # layer norm -> mlp -> + residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings: map token indices to embedding vectors
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Positional embeddings: learnable embeddings for each position in a sequence
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # List of transformer blocks (attention + MLP layers)
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # Final LayerNorm applied to the output of the last transformer block
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # projects the final hidden state to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and final projection
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """initialize model weights with standard GPT2 initialiation (normal, std=0.02)"""
        # If the module is a linear (fully connected) layer
        if isinstance(module, nn.Linear):
            # Standard GPT-style initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # If the module is an embedding layer
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """forward pass"""
        # idx has shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Cannot forward sequence lenght of {T}, block size is {self.config.block_size}"
        )

        # forward the token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape T
        pos_emb = self.transformer.wpe(pos)  # (T, embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, embd)
        x = pos_emb + tok_emb  # (B, T, embd)

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:  # (B*T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """loads pretrained GPT-2 model weights from hugging face"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading wieghts from pretrained gpt: %s" % model_type)

        # n_layer, n_embd, n_head are determined from model type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        # create from scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        # ignore the buffers
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # initialized hugging face transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        # ignore buffers and masks
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        # manually transpose the weights if it's from tensorflow
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )

        # copy the pretrained weights
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """configure optimizers by implementing weight decay and using fused Adam (if possible)"""
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters in 2D or above will be weight decayed
        # all tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # create AdamW optimizer and use the fused version if it is available
        # enables optimized CUDA kernel implementations: reduce overhead and memory access, and increasing throughput.
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"Using fused AdamW: {use_fused}")

        # beta1: momentum, beta2: RMS scaling:
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8
        )
        return optimizer
