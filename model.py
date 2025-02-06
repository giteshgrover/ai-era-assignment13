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

def precompute_rotary_emb(dim: int, max_seq_len: int, base: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the rotary position embeddings
    Args:
        dim: Dimension of the embeddings
        max_seq_len: Maximum sequence length
        base: Base for the angle calculations
    Returns:
        Tuple of (sin, cos) tensors of shape (max_seq_len, dim//2)
    """
    # Create position indices tensor
    position = torch.arange(max_seq_len).unsqueeze(1)  # (seq_len, 1)
    # Create dimension indices tensor
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(base) / dim))  # (dim//2)
    # Compute angles
    angles = position * div_term  # (seq_len, dim//2)
    # Return sin and cos
    return torch.sin(angles), torch.cos(angles)

def apply_rotary_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to the input tensor
    Args:
        x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
        sin: Sine tensor of shape (seq_len, head_dim//2)
        cos: Cosine tensor of shape (seq_len, head_dim//2)
    Returns:
        Tensor with rotary position embeddings applied
    """
    # Reshape x to split last dimension in half
    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)
    # Extract even and odd dimensions
    x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
    
    # Reshape sin and cos for broadcasting
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])  # (1, seq_len, 1, dim//2)
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])  # (1, seq_len, 1, dim//2)
    
    # Apply rotation using the rotation matrix multiplication
    result = torch.stack([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)
    
    return result.flatten(-2)  # Flatten last 2 dimensions

class LlamaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: Optional[int] = None, max_position_embeddings=2048):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # self.q_proj = nn.Linear(dim, dim, bias=False)
        # self.k_proj = nn.Linear(dim, dim, bias=False)
        # self.v_proj = nn.Linear(dim, dim, bias=False)
        # Adjust projections for GQA
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False) # (B, T, D) -> (B, T, D) or (B, T, H * D/H)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False) # (B, T, D) -> (B, T, H_kv * D/H)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False) # (B, T, D) -> (B, T, H_kv * D/H)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        # Cache attributes
        self.k_cache = None
        self.v_cache = None
        self.cache_seq_len = 0
        
        # Precompute sin and cos for all positions
        self.sin, self.cos = precompute_rotary_emb(self.head_dim, max_position_embeddings)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache: bool = False):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Get rotary embeddings for the new tokens
        # sin = self.sin[self.cache_seq_len:self.cache_seq_len + seq_len].to(x.device)
        # cos = self.cos[self.cache_seq_len:self.cache_seq_len + seq_len].to(x.device)
        sin = self.sin[:seq_len].to(x.device)
        cos = self.cos[:seq_len].to(x.device)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)

        # Handle KV caching
        # if use_cache:
        #     if self.k_cache is None:
        #         # Initialize cache if empty
        #         self.k_cache = k
        #         self.v_cache = v
        #     else:
        #         # Concatenate new KV with cached KV
        #         self.k_cache = torch.cat([self.k_cache, k], dim=1)
        #         self.v_cache = torch.cat([self.v_cache, v], dim=1)
            
        #     # Use concatenated KV pairs
        #     k = self.k_cache
        #     v = self.v_cache
            
        #     # Update cache sequence length
        #     self.cache_seq_len += seq_len

        # Reshape for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Handle GQA (Grouped Query Attention)
        if self.num_queries_per_kv > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1)
            v = v.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1)
            k = k.reshape(batch_size, self.num_heads, -1, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, -1, self.head_dim)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(out)

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.cache_seq_len = 0

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
        self.attention = LlamaAttention(
            config.nn_embed, 
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_sequence_len
        )
        self.feed_forward = LlamaFFN(config.nn_embed, config.ffn_intermediate_size)
        self.attention_norm = nn.RMSNorm(config.nn_embed, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.nn_embed, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache: bool = False):
        x = x + self.attention(self.attention_norm(x), mask, use_cache)
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
        self.norm = nn.RMSNorm(config.nn_embed, eps=config.rms_norm_eps)
        # final layer returning the logits of size (batch_size, vocab_size)
        self.lm_head = nn.Linear(config.nn_embed, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache: bool = False):
        if (mask is None):
            mask = self.create_causal_mask(x.shape[1], device=x.device)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask, use_cache)
        x = self.norm(x)
        return self.lm_head(x)

    def clear_cache(self):
        """Clear KV cache in all attention layers"""
        for layer in self.layers:
            layer.attention.clear_cache()
    
    def create_causal_mask(self, seq_len, device):
        """Creates a causal attention mask where each position can only attend to previous positions"""
        # Create lower triangular matrix (including diagonal)
        # mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # mask = torch.triu(torch.ones(1, 1, seq_len, seq_len), diagonal=1).bool()
        # # Invert and convert to float
        # return (~mask).float()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len).to(device)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, 
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """
        Generate text using the model
        Args:
            input_ids: Starting token ids (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Controls randomness (1.0 = neutral, <1.0 = more deterministic, >1.0 = more random)
            top_k: Number of highest probability tokens to consider for sampling
        Returns:
            Generated token ids (B, T+max_new_tokens)
        """
        batch_size, seq_len = input_ids.shape
        
        # clear existing KV caching
        self.clear_cache()
        
        # Create a new tensor to store the generated tokens
        input_ids = torch.cat([input_ids, torch.zeros((batch_size, max_new_tokens), 
                            dtype=torch.long, device=input_ids.device)], dim=1)
        
        # Generate tokens one at a time
        for idx in range(max_new_tokens):
            # print(f"Generating token {idx+1} of {max_new_tokens}")
            
            # Get the current sequence length including cached tokens
            current_seq_len = seq_len + idx

            next_mask = self.create_causal_mask(current_seq_len, device=input_ids.device)
            
            # Create mask that includes both the current input and cached tokens
            # if idx == 0:
            #     # First iteration - create mask for the full input sequence
            #     next_mask = self.create_causal_mask(current_seq_len, device=input_ids.device)
            # else:
            #     # Subsequent iterations - create mask for the new token attending to all previous tokens
            #     next_mask = torch.ones((1, 1, 1, current_seq_len), device=input_ids.device)

            # Process including the new tokens
            logits = self(input_ids[:, :current_seq_len], next_mask, use_cache=False)
            
            # Get the last token's logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from the filtered distribution
            next_token = top_k_indices[
                torch.arange(batch_size, device=input_ids.device),
                torch.multinomial(probs, num_samples=1).squeeze(1)
            ]
            
            # Update input_ids with the new token
            input_ids[:, current_seq_len] = next_token
        
        return input_ids