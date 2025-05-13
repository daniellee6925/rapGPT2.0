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
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size()

        # Linear projection → (B, T, 3C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape and transpose for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # Save new present for caching
        present = (k, v) if use_cache else None
        L = k.size(2)  # total length of K/V after cache
        S = q.size(2)  # current input length

        # Create custom causal mask for query length S and key length L
        mask = torch.full((S, L), float("-inf"), device=x.device)
        mask.triu_(1)  # upper triangular
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, L)
        mask = mask.expand(B, self.n_head, S, L)  # match q/k/v shape

        # Scaled dot-product attention (causal)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return (y, present) if use_cache else y


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

    def forward(self, x, past_kv=None, use_cache=False):
        if use_cache:
            attn_out, present = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln_1(x))
            present = None

        x = x + self.mlp(self.ln_2(x))
        return (x, present) if use_cache else x


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

    def forward(
        self, idx, targets=None, position_ids=None, past_kv=None, use_cache=False
    ):
        """forward pass"""
        # idx has shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Cannot forward sequence lenght of {T}, block size is {self.config.block_size}"
        )

        # forward the token and positional embeddings
        if position_ids is None:
            past_len = 0 if past_kv is None else past_kv[0][0].size(2)
            position_ids = (
                torch.arange(
                    past_len, past_len + T, dtype=torch.long, device=idx.device
                )
                .unsqueeze(0)
                .expand(B, -1)
            )  # (B, T)

        tok_emb = self.transformer.wte(idx)  # (B, T, embd)
        pos_emb = self.transformer.wpe(position_ids)  # (B, T, embd)
        x = tok_emb + pos_emb

        presents = []
        if past_kv is None:
            past_kv = [None] * len(self.transformer.h)

        for i, block in enumerate(self.transformer.h):
            out = block(x, past_kv=past_kv[i], use_cache=use_cache)
            if use_cache:
                x, present = out
                presents.append(present)
            else:
                x = out

        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:  # (B*T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if use_cache:
            return logits, loss, presents
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

    def resize_token_embeddings(self, new_vocab_size):
        old_emb = self.transformer.wte
        old_vocab_size, emb_dim = old_emb.weight.size()

        if new_vocab_size == old_vocab_size:
            return  # no change needed

        # Create new embedding layer
        new_emb = torch.nn.Embedding(new_vocab_size, emb_dim)
        new_emb.weight.data[:old_vocab_size] = old_emb.weight.data

        self.transformer.wte = new_emb
        self.lm_head = torch.nn.Linear(emb_dim, new_vocab_size, bias=False)

        # Tie weights if needed
        self.lm_head.weight = self.transformer.wte.weight

    def generate(self, x, max_length=100, num_return_sequences=1):
        past_kv = None  # initialize cache

        while x.size(1) < max_length:
            with torch.no_grad():
                if past_kv is None:
                    # First forward pass: input the full prompt
                    position_ids = torch.arange(
                        0, x.size(1), device=x.device
                    ).unsqueeze(0)
                    logits, _ = self.forward(
                        x, position_ids=position_ids, use_cache=False
                    )
                    # logits, _, past_kv = model(x, position_ids=position_ids, use_cache=True)
                else:
                    # Subsequent passes: input only the last generated token
                    position_ids = torch.tensor([[x.size(1)]], device=x.device)
                    logits, _ = self.forward(
                        x[:, -1:],
                        position_ids=position_ids,
                        past_kv=past_kv,
                        use_cache=False,
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

        generated_texts = []
        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            generated_texts.append(tokens)
        return generated_texts
