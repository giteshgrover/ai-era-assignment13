# ai-era-assignment13
Assignment 13 for SmolLM2-135 model

# Llama 2 Architecture
![Llama 2 Architecture](./static/llamaModel.jpg)
Read https://pub.towardsai.net/llama-explained-a70e71e706e9 for more details.

# Compare Custom SmolLM2-135 with HuggingFaceTB/SmolLM2-135M
 HuggingFaceTB/SmolLM2-135M
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
```

Custom SmolLM2-135
```
SmolLM2(
  (embedding): Embedding(49152, 576)
  (layers): ModuleList(
    (0-29): 30 x LlamaBlock(
      (attention): LlamaAttention(
        (q_proj): Linear(in_features=576, out_features=576, bias=False)
        (k_proj): Linear(in_features=576, out_features=576, bias=False)
        (v_proj): Linear(in_features=576, out_features=576, bias=False)
        (o_proj): Linear(in_features=576, out_features=576, bias=False)
      )
      (feed_forward): LlamaFFN(
        (gate): Linear(in_features=576, out_features=1536, bias=False)
        (up): Linear(in_features=576, out_features=1536, bias=False)
        (down): Linear(in_features=1536, out_features=576, bias=False)
        (act_fn): SiLU()
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)

```