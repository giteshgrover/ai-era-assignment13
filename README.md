# ai-era-assignment13
Assignment 13 for SmolLM2-135 model

# Llama 2 Architecture
![Llama 2 Architecture](./static/llamaModel.jpg)
Read https://pub.towardsai.net/llama-explained-a70e71e706e9 for more details.

# Compare Custom SmolLM2-135 with HuggingFaceTB/SmolLM2-135M
 HuggingFaceTB/SmolLM2-135M
```bash
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
```bash 
SmolLM2(
  (embedding): Embedding(49152, 576)
  (layers): ModuleList(
    (0-29): 30 x LlamaBlock(
      (attention): LlamaAttention(
        (q_proj): Linear(in_features=576, out_features=576, bias=False)
        (k_proj): Linear(in_features=576, out_features=192, bias=False)
        (v_proj): Linear(in_features=576, out_features=192, bias=False)
        (o_proj): Linear(in_features=576, out_features=576, bias=False)
      )
      (feed_forward): LlamaFFN(
        (gate): Linear(in_features=576, out_features=1536, bias=False)
        (up): Linear(in_features=576, out_features=1536, bias=False)
        (down): Linear(in_features=1536, out_features=576, bias=False)
        (act_fn): SiLU()
      )
      (attention_norm): RMSNorm((576,), eps=1e-05, elementwise_affine=True)
      (ffn_norm): RMSNorm((576,), eps=1e-05, elementwise_affine=True)
    )
  )
  (norm): RMSNorm((576,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)

```

# Training Logs
```bash
(venv) gitesh.grover@Giteshs-MacBook-Pro ai-era-assignment13 % python train.py


Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 720.56it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 562123.22it/s]
Epoch: 0, Step: 0, Batch: 0, Loss: 10.9101, Time: 1.44s, Token/s: 2842.75
Saved checkpoint at step 0
What is Gravity? thymopenedi something aneur checklist fertiliserlete hiding Watching [[GuardinnamonGuard thym thym something multilinguali runway astronlighten runwayinnamon nastylighten disadvant snout plumquest
Epoch: 0, Step: 1, Batch: 1, Loss: 10.6729, Time: 2.00s, Token/s: 2044.98
Epoch: 0, Step: 2, Batch: 2, Loss: 9.2034, Time: 1.16s, Token/s: 3517.56
Epoch: 0, Step: 3, Batch: 3, Loss: 8.5723, Time: 1.09s, Token/s: 3766.14
Epoch: 0, Step: 4, Batch: 4, Loss: 8.1478, Time: 1.07s, Token/s: 3845.85
Epoch: 0, Step: 5, Batch: 5, Loss: 8.1278, Time: 1.15s, Token/s: 3567.04
Epoch: 0, Step: 6, Batch: 6, Loss: 8.2777, Time: 1.06s, Token/s: 3855.25
Epoch: 0, Step: 7, Batch: 7, Loss: 8.1852, Time: 1.08s, Token/s: 3810.08
Epoch: 0, Step: 8, Batch: 8, Loss: 8.3953, Time: 1.09s, Token/s: 3764.08
Epoch: 0, Step: 9, Batch: 9, Loss: 8.3247, Time: 1.08s, Token/s: 3791.72
Epoch: 0, Step: 10, Batch: 10, Loss: 8.3027, Time: 1.06s, Token/s: 3859.40
Epoch: 0, Step: 11, Batch: 11, Loss: 8.2021, Time: 1.10s, Token/s: 3713.23
Epoch: 0, Step: 12, Batch: 12, Loss: 8.0664, Time: 1.06s, Token/s: 3857.36
Epoch: 0, Step: 13, Batch: 13, Loss: 7.9695, Time: 1.08s, Token/s: 3782.96
Epoch: 0, Step: 14, Batch: 14, Loss: 8.2516, Time: 1.16s, Token/s: 3523.41
Epoch: 0, Step: 15, Batch: 15, Loss: 8.0935, Time: 1.16s, Token/s: 3541.74
Epoch: 0, Step: 16, Batch: 16, Loss: 8.0569, Time: 1.07s, Token/s: 3818.43
Epoch: 0, Step: 17, Batch: 17, Loss: 7.8988, Time: 1.08s, Token/s: 3787.21
Epoch: 0, Step: 18, Batch: 18, Loss: 7.9157, Time: 1.08s, Token/s: 3802.87
Epoch: 0, Step: 19, Batch: 19, Loss: 7.9390, Time: 1.08s, Token/s: 3790.54
Epoch: 0, Step: 20, Batch: 20, Loss: 8.1111, Time: 1.07s, Token/s: 3818.68
Epoch: 0, Step: 21, Batch: 21, Loss: 7.9728, Time: 1.12s, Token/s: 3657.56
Epoch: 0, Step: 22, Batch: 22, Loss: 7.9211, Time: 1.08s, Token/s: 3808.09
Epoch: 0, Step: 23, Batch: 23, Loss: 7.9855, Time: 1.09s, Token/s: 3762.07
Epoch: 0, Step: 24, Batch: 24, Loss: 8.4774, Time: 1.08s, Token/s: 3794.95
Epoch: 0, Step: 25, Batch: 25, Loss: 8.1493, Time: 1.07s, Token/s: 3816.37
Epoch: 0, Step: 26, Batch: 26, Loss: 8.1052, Time: 1.08s, Token/s: 3792.31
Epoch: 0, Step: 27, Batch: 27, Loss: 8.2873, Time: 1.15s, Token/s: 3570.67
Epoch: 0, Step: 28, Batch: 28, Loss: 7.9622, Time: 1.11s, Token/s: 3701.80
```