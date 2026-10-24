"""用 PyTorch 基础模块实现一个小型 GPT。"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 64
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class LayerNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dimension))
        self.shift = nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        return self.scale * (x - mean) * torch.rsqrt(variance + self.eps) + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x):
        batch, length, dimension = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # 将特征维度拆成多个头：(batch, heads, sequence, head_dim)。
        q = q.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 当前位置只能关注自己及之前的位置。
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        context = (weights @ v).transpose(1, 2).contiguous().view(batch, length, dimension)
        return self.output_dropout(self.projection(context))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.num_layers)))
        self.final_norm = LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        # 输入嵌入和输出分类器共享权重。
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids, targets=None):
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device)
        x = self.dropout(self.token_embedding(token_ids) + self.position_embedding(positions))
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=100):
        # 调用前使用 model.eval() 关闭 dropout。
        for _ in range(max_new_tokens):
            logits, _ = self(token_ids[:, -self.config.context_length :])
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
