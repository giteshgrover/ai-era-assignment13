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
        # BatchSize, Sequence Length, Embedding Dimensions
        batch_size, seq_len, _ = x.shape
        print(x.shape)
        # print("batch_size", batch_size)
        # print("seq_len", seq_len)
        # print("num_heads", self.num_heads)
        # print("num_kv_heads", self.num_kv_heads)
        # print("num_queries_per_kv", self.num_queries_per_kv)
        # print("head_dim", self.head_dim)
        
        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)     # (B, T, H * D/H) -> (B, T, H, D/H)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)  # (B, T, H_kv * D/H) -> (B, T, H_kv, D/H)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)  # (B, T, H_kv * D/H) -> (B, T, H_kv, D/H)

        # Use the precomputed sin and cos for the input sequence length
        # sin = self.sin[:seq_len].to(x.device)
        # cos = self.cos[:seq_len].to(x.device)
        sin = self.sin[self.cache_seq_len:self.cache_seq_len + seq_len].to(x.device)
        cos = self.cos[self.cache_seq_len:self.cache_seq_len + seq_len].to(x.device)

        # Apply rotary embeddings on q & k
        q = apply_rotary_emb(q, sin, cos)
        k = apply_rotary_emb(k, sin, cos)
        
        # Handle KV caching
        if use_cache:
            if self.k_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=1)
                self.v_cache = torch.cat([self.v_cache, v], dim=1)
            k, v = self.k_cache, self.v_cache
            self.cache_seq_len += seq_len
        
        # Reshape for attention computation
        q = q.transpose(1, 2)  # (B, T, H, D/H) -> (B, H, T, D/H)
        k = k.transpose(1, 2)  # (B, T', H_kv, D/H) -> (B, H_kv, T', D/H)
        v = v.transpose(1, 2)  # (B, T', H_kv, D/H) -> (B, H_kv, T', D/H)
        
        # Repeat k and v for each query head in the group
        if self.num_queries_per_kv > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1) # (B, H_kv, T', D/H_kv) -> (B, H_kv, 1, T', D/H) -> (B, H_kv, N_q, T', D/H)
            v = v.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1)
            k = k.reshape(batch_size, self.num_heads, -1, self.head_dim)   # (B, H_kv, N_q, T', D/H) -> (B, H_kv*N_q, T, D/H) i.e (B, H, T, D/H)
            v = v.reshape(batch_size, self.num_heads, -1, self.head_dim) 
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (B, H, T, D/H) * (B, H, D/H, T) -> (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        # Compute output
        # out = torch.matmul(attn, v) # (B, H, T, T) * (B, H, T, D/H) -> (B, H, T, D/H)
        # out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # (B, H, T, D/H) -> (B, T, D)
        # return self.o_proj(out) # (B, T, D) -> (B, T, D)
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
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask, use_cache)
        x = self.norm(x)
        return self.lm_head(x)

    def clear_cache(self):
        """Clear KV cache in all attention layers"""
        for layer in self.layers:
            layer.attention.clear_cache()

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
        # Create attention mask for input sequence
        batch_size, seq_len = input_ids.shape
        # TODO where are the 0s?
        input_mask = torch.ones((batch_size, 1, seq_len, seq_len), device=input_ids.device) # (B, 1, T, T) of 1s
        
        # clear existing KV caching
        self.clear_cache()
        
        # Process the initial input sequence to generate the first token
        logits = self(input_ids, input_mask, use_cache=True) # (B, T, V)
        # Create a new tensor of size (B, T+max_new_tokens) to store the generated tokens (inital value of these new tokes is 0)
        input_ids = torch.cat([input_ids, torch.zeros((batch_size, max_new_tokens), 
                            dtype=torch.long, device=input_ids.device)], dim=1)
        
        # Generate tokens one at a time
        for idx in range(max_new_tokens):
            # Get the last token's logits
            next_token_logits = logits[:, -1, :] / temperature # (B, V)
            
            # Apply top-k filtering on the last token's logits
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1) # probability distribution over the top k tokens (B, K)
            
            # Sample from the filtered distribution i.e. get the top token for every batch
            next_token = top_k_indices[
                torch.arange(batch_size, device=input_ids.device), # (0,1,2....)
                torch.multinomial(probs, num_samples=1).squeeze(1) # (1,0,5,.. )..Indices of top sample of every batch
            ] # (B, 1)
            
            # Update input_ids with the new token
            input_ids[:, seq_len + idx] = next_token
            
            # Create mask for the next token TODO where are the 0s?
            next_mask = torch.ones((batch_size, 1, 1, seq_len + idx + 1), device=input_ids.device) # (B, 1, 1, T+1) of 1s
            
            # Process only the new token (B, 1)
            logits = self(input_ids[:, seq_len + idx:seq_len + idx + 1], 
                         next_mask, use_cache=True)
        
        return input_ids