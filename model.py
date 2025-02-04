import torch
import torch.nn as nn
import math
from typing import Optional
import torch.nn.functional as F

# This llama model is based on the paper: https://arxiv.org/pdf/2302.13971.pdf
# Model Architecturte: static/llamaModel.jpg
# It is a transformer model with rotary position embeddings (RoPE) and SwiGLU 
# activation function. It uses RMSNorm for normalization.
# Other Good reads: https://pub.towardsai.net/llama-explained-a70e71e706e9

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def precompute_rotary_emb(dim: int, max_seq_len: int, base: int = 10000):
    # Create position indices
    position = torch.arange(max_seq_len).unsqueeze(1)
    # Create dimension indices
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(base) / dim))
    # Compute sin and cos
    sin = torch.sin(position * div_term)
    cos = torch.cos(position * div_term)
    return sin, cos

def apply_rotary_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    x_rot, x_pass = x[..., :dim//2], x[..., dim//2:]
    # Reshape for broadcasting
    x_rot = x_rot.view(*x_rot.shape[:-1], -1, 2)
    rot_matrix = torch.stack([cos, -sin], dim=-1)
    x_rotated = torch.sum(x_rot * rot_matrix, dim=-1)
    return torch.cat([x_rotated, x_pass], dim=-1)

class LlamaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_position_embeddings=2048):
        # dim or nn_embed is the embedding dimensions
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        # Precompute sin n cos for all the positions in the sequence (based on max sequence length)
        self.sin, self.cos = precompute_rotary_emb(self.head_dim, max_position_embeddings)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # BatchSize, Sequence Length, Embedding Dimensions
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape (B, T, D) -> (B, T, H, D/H)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Use the precomputed sin and cos for the input sequence length
        sin = self.sin[:seq_len].to(x.device)
        cos = self.cos[:seq_len].to(x.device)
        # Apply rotary embeddings on q & k 
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)
        
        # Reshape for attention computation
        q = q.transpose(1, 2) # (B, H, T, D/H)
        k = k.transpose(1, 2) # (B, H, T, D/H)
        v = v.transpose(1, 2) # (B, H, T, D/H)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1) # (B, H, T, T)
        
        # Compute output
        out = torch.matmul(attn, v) # (B, H, T, T) * (B, H, T, D/H) -> (B, H, T, D/H)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # (B, H, T, D/H) -> (B, T, D)
        return self.o_proj(out) # (B, T, D) -> (B, T, D)

class LlamaFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.act_fn = nn.SiLU() # SwiGLU activation function
    
    def forward(self, x):
        return self.down(self.act_fn(self.gate(x)) * self.up(x))

class LlamaBlock(nn.Module):
    def __init__(self, config):
        # nn_embed or dim is the dimension of the input to the block
        super().__init__()
        self.attention = LlamaAttention(config.nn_embed, config.num_attention_heads, config.max_sequence_len)
        self.feed_forward = LlamaFFN(config.nn_embed, config.ffn_intermediate_size)
        self.attention_norm = RMSNorm(config.nn_embed)
        self.ffn_norm = RMSNorm(config.nn_embed)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class SmolLM2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Normal Embedding (position embedding will be part of Attention layer)
        self.embedding = nn.Embedding(config.vocab_size, config.nn_embed)
        
        # total num_hidden_layers Blocks (Each block has attention and feedforward layer)
        self.layers = nn.ModuleList([
            LlamaBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.nn_embed)
        # final layer returning the logits of size (batch_size, vocab_size)
        self.lm_head = nn.Linear(config.nn_embed, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.lm_head(x)