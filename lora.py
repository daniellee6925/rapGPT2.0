import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Literal, Union
from safetensors.torch import save_file
from safetensors.torch import load_file
import inspect
from gpt2 import GPT, GPTConfig
import helper_functions


class LoRALayerBase:
    def __init__(self, rank=8, lora_alpha=8, lora_dropout=0.0, use_rslora=True):
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = (
            nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x
        )
        self.use_rslora = use_rslora

        self.scaling = (
            self.lora_alpha / self.rank**0.5
            if use_rslora
            else self.lora_alpha / self.rank
        )

    def _load_pretrained_wiehgts(self, state_dict):
        self.weight_data = state_dict["weight"]
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bias"]


class LoRALinear(nn.Linear, LoRALayerBase):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        use_rslora=True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)

        LoRALayerBase.__init__(
            self,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
        )

        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))  # for testing

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5)) # for testing

    def _merge_weights(self):
        merged_weights = self.weight.data + self.scaling * (self.lora_A @ self.lora_B).T

        state_dict = {"weight": merged_weights}
        if self.bias is not None:
            state_dict["bias"] = self.bias.detach().clone()

        merged_linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=True if self.bias is not None else False,
        )

        merged_linear.load_state_dict(state_dict)

        return merged_linear

    def forward(self, x):
        orig_layer_out = F.linear(x, self.weight, bias=self.bias)  # original W matrix

        lora_mult = (self.lora_A @ self.lora_B) * self.scaling  # delta W matrix

        low_rank_out = self.lora_dropout(x) @ lora_mult

        output = orig_layer_out + low_rank_out  # W + delta W

        return output


class LoRAEmbedding(nn.Embedding, LoRALayerBase):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        use_rslora=True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)

        LoRALayerBase.__init__(
            self,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
        )

        self.weight.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(self.num_embeddings, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.embedding_dim))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):
        merged_weights = self.weight.data + self.scaling * (self.lora_A @ self.lora_B)

        state_dict = {"weight": merged_weights}

        merged_emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        merged_emb.load_state_dict(state_dict)

        return merged_emb

    def forward(self, x):
        orig_layer_out = F.embedding(
            input=x,
            weight=self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        low_rank_A_output = F.embedding(
            input=x,
            weight=self.lora_A,
            padding_idx=self.padding_idx,
            max_norm=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        low_rank_output = (low_rank_A_output @ self.lora_B) * self.scaling

        output = orig_layer_out + low_rank_output

        return output


@dataclass
class LoraConfig:
    rank: int = 8
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True


class LoraModel(nn.Module):
    def __init__(self, model, config):
        super(LoraModel, self).__init__()

        self.lora_model = model
        self.config = config

        if self.config.target_modules is None:
            self.config.target_modules = []
        elif isinstance(self.config.target_modules, str):
            self.config.target_modules = [self.config.target_modules]

        if self.config.exclude_modules is None:
            self.config.exclude_modules = []
        elif isinstance(self.config.exclude_modules, str):
            self.config.exclude_modules = [self.config.exclude_modules]

        orig_trainable_parameters = self._compute_trainable_parameters()

        self._disable_all_grads()

        # print(orig_trainable_parameters)
        # print(self._compute_trainable_parameters())
        self._apply_lora(self.lora_model)

        self._toggle_bias_grad()

        lora_trainable_parameters = self._compute_trainable_parameters()

        print(f"Initial Parameters: {orig_trainable_parameters}")
        print(f"LoRA Parameters: {lora_trainable_parameters}")
        print(
            f"Trainable Portion: {round(lora_trainable_parameters * 100 / orig_trainable_parameters, 2)}%"
        )

    def forward(self, *inputs, **kwargs):
        return self.lora_model(*inputs, **kwargs)

    def _exclude_module_name_check(self, name):
        return any([ex in name for ex in self.config.exclude_modules])

    def _target_module_name_check(self, name):
        return any([tgt in name for tgt in self.config.target_modules])

    def _apply_lora(self, module):
        for name, child in module.named_children():
            if self._target_module_name_check(name):
                if isinstance(child, nn.Linear):
                    new_layer = LoRALinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=True if child.bias is not None else False,
                        rank=self.config.rank,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        use_rslora=self.config.use_rslora,
                    )

                    new_layer._load_pretrained_wiehgts(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Embedding):
                    new_layer = LoRAEmbedding(
                        num_embeddings=child.num_embeddings,
                        embedding_dim=child.embedding_dim,
                        rank=self.config.rank,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        use_rslora=self.config.use_rslora,
                    )
                    new_layer._load_pretrained_wiehgts(child.state_dict())
                    setattr(module, name, new_layer)

            if (
                len(list(child.children())) > 0
            ) and not self._exclude_module_name_check(name):
                self._apply_lora(child)

    def _compute_trainable_parameters(self):
        total_learnable_parameters = 0
        for param in self.lora_model.parameters():
            if param.requires_grad:
                total_learnable_parameters += param.numel()
        return total_learnable_parameters

    def _disable_all_grads(self):
        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                param.requires_grad = False

    def _toggle_bias_grad(self):
        for name, param in self.named_parameters():
            if not self._exclude_module_name_check(name):
                if ".bias" in name:
                    if self.config.bias == "none":
                        param.requires_grad = False
                    elif self.config.bias == "all":
                        param.requires_grad = True
                    elif (self.config.bias == "lora only") and (
                        self._target_module_name_check(name)
                    ):
                        param.requires_grad = True

    def _merge_weights(self, module):
        for name, child in module.named_children():
            if isinstance(child, (LoRALinear, LoRAEmbedding)):
                merged_layer = child._merge_weights()

                setattr(module, name, merged_layer)

            if len(list(child.children())) > 0:
                self._merge_weights(child)

    def save_model(self, path, merge_weights=False):
        def _detach_cpu(param):
            return param.detach().cpu()

        if merge_weights:
            self._merge_weights(self.lora_model)

            state_dict = {
                name.replace("lora_model.", ""): _detach_cpu(param)
                for (name, param) in self.named_parameters()
            }

        else:
            state_dict = {
                name: _detach_cpu(param)
                for (name, param) in self.named_parameters()
                if param.requires_grad
            }

        # for k in state_dict.keys():
        #     print(k)

        save_file(state_dict, path)

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


# if __name__ == "__main__":
#     model = helper_functions.load_model(GPT, GPTConfig, "Models", "pretrained_gpt2_v2")

#     target_modules = ["c_attn", "c_proj", "c_fc"]
#     exclude_modules = ["ln_f"]

#     config = LoraConfig(
#         target_modules=target_modules, exclude_modules=exclude_modules, bias="none"
#     )

#     # for name, param in model.named_parameters():
#     #     print(name)

#     print("------------------------------------")
#     lora_model = LoraModel(model, config)
#     lora_model.save_model("path", merge_weights=True)
#     model = GPT(GPTConfig)
#     state_dict = load_file("path")
#     model.load_state_dict(state_dict)
