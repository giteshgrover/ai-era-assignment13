---
title: SmolLM2 135M Text Generation Demo
emoji: 📚
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# SmolLM2 Text Generation Demo

This is a simple text generation demo using the SmolLM2 language model with a Gradio interface.

## Description

This application provides a web interface for text generation using the SmolLM2 language model. Users can input a prompt and adjust various generation parameters to control the output.

## Features

- Interactive web interface built with Gradio
- Adjustable generation parameters:
  - Maximum new tokens (1-150)
  - Temperature (0.1-2.0)
  - Top-K sampling (1-100)
- Real-time text generation

## Usage

1. Enter your prompt in the text input field
2. Adjust the generation parameters (optional):
   - **Max New Tokens**: Controls the length of the generated text
   - **Temperature**: Controls randomness (higher = more creative, lower = more focused)
   - **Top-K**: Controls diversity of word choices
3. Click submit to generate text

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
 ## Run the application:
   ```bash
   python app.py
   ```
   The interface will be available at `http://localhost:7860`


## Train the model:
```bash
python train.py
```


# Model details
SmolLM2 is a language model designed for [add your model's specific details here]. The model uses the [specify tokenizer] tokenizer from Hugging Face's transformers library.

## Llama 2 Architecture

![Llama 2 Architecture](./static/llamaModel.jpg)
Read https://pub.towardsai.net/llama-explained-a70e71e706e9 for more details.

# Compare Custom SmolLM2-135 with HuggingFaceTB/SmolLM2-135M
 HuggingFaceTB/SmolLM2-135M
```bash
Model 1 - HuggingFaceTB/SmolLM2-135M:
Model: LlamaForCausalLM(
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
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
LlamaForCausalLM                              --                        --
├─LlamaModel: 1-1                             --                        --
│    └─Embedding: 2-1                         [64, 64, 576]             28,311,552
│    └─LlamaRotaryEmbedding: 2-2              [1, 64, 64]               --
│    └─ModuleList: 2-3                        --                        --
│    │    └─LlamaDecoderLayer: 3-1            [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-2            [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-3            [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-4            [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-5            [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-6            [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-7            [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-8            [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-9            [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-10           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-11           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-12           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-13           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-14           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-15           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-16           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-17           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-18           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-19           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-20           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-21           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-22           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-23           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-24           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-25           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-26           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-27           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-28           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-29           [64, 64, 576]             3,540,096
│    │    └─LlamaDecoderLayer: 3-30           [64, 64, 576]             3,540,096
│    └─LlamaRMSNorm: 2-4                      [64, 64, 576]             576
├─Linear: 1-2                                 [64, 64, 49152]           28,311,552
===============================================================================================
Total params: 162,826,560
Trainable params: 162,826,560
Non-trainable params: 0
Total mult-adds (G): 10.42
===============================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 7876.90
Params size (MB): 651.31
Estimated Total Size (MB): 8528.24
===============================================================================================
```

Custom SmolLM2-135
```bash 
Model 2 - Custom SmolLM2-135M Model :
Model: SmolLM2(
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
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
SmolLM2                                  [64, 64, 49152]           --
├─Embedding: 1-1                         [64, 64, 576]             28,311,552
├─ModuleList: 1-2                        --                        --
│    └─LlamaBlock: 2-1                   [64, 64, 576]             --
│    │    └─RMSNorm: 3-1                 [64, 64, 576]             576
│    │    └─LlamaAttention: 3-2          [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-3                 [64, 64, 576]             576
│    │    └─LlamaFFN: 3-4                [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-2                   [64, 64, 576]             --
│    │    └─RMSNorm: 3-5                 [64, 64, 576]             576
│    │    └─LlamaAttention: 3-6          [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-7                 [64, 64, 576]             576
│    │    └─LlamaFFN: 3-8                [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-3                   [64, 64, 576]             --
│    │    └─RMSNorm: 3-9                 [64, 64, 576]             576
│    │    └─LlamaAttention: 3-10         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-11                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-12               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-4                   [64, 64, 576]             --
│    │    └─RMSNorm: 3-13                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-14         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-15                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-16               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-5                   [64, 64, 576]             --
│    │    └─RMSNorm: 3-17                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-18         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-19                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-20               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-6                   [64, 64, 576]             --
│    │    └─RMSNorm: 3-21                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-22         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-23                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-24               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-7                   [64, 64, 576]             --
│    │    └─RMSNorm: 3-25                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-26         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-27                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-28               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-8                   [64, 64, 576]             --
│    │    └─RMSNorm: 3-29                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-30         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-31                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-32               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-9                   [64, 64, 576]             --
│    │    └─RMSNorm: 3-33                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-34         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-35                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-36               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-10                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-37                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-38         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-39                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-40               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-11                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-41                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-42         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-43                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-44               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-12                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-45                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-46         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-47                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-48               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-13                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-49                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-50         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-51                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-52               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-14                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-53                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-54         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-55                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-56               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-15                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-57                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-58         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-59                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-60               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-16                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-61                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-62         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-63                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-64               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-17                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-65                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-66         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-67                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-68               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-18                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-69                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-70         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-71                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-72               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-19                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-73                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-74         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-75                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-76               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-20                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-77                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-78         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-79                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-80               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-21                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-81                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-82         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-83                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-84               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-22                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-85                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-86         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-87                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-88               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-23                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-89                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-90         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-91                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-92               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-24                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-93                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-94         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-95                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-96               [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-25                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-97                [64, 64, 576]             576
│    │    └─LlamaAttention: 3-98         [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-99                [64, 64, 576]             576
│    │    └─LlamaFFN: 3-100              [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-26                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-101               [64, 64, 576]             576
│    │    └─LlamaAttention: 3-102        [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-103               [64, 64, 576]             576
│    │    └─LlamaFFN: 3-104              [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-27                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-105               [64, 64, 576]             576
│    │    └─LlamaAttention: 3-106        [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-107               [64, 64, 576]             576
│    │    └─LlamaFFN: 3-108              [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-28                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-109               [64, 64, 576]             576
│    │    └─LlamaAttention: 3-110        [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-111               [64, 64, 576]             576
│    │    └─LlamaFFN: 3-112              [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-29                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-113               [64, 64, 576]             576
│    │    └─LlamaAttention: 3-114        [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-115               [64, 64, 576]             576
│    │    └─LlamaFFN: 3-116              [64, 64, 576]             2,654,208
│    └─LlamaBlock: 2-30                  [64, 64, 576]             --
│    │    └─RMSNorm: 3-117               [64, 64, 576]             576
│    │    └─LlamaAttention: 3-118        [64, 64, 576]             884,736
│    │    └─RMSNorm: 3-119               [64, 64, 576]             576
│    │    └─LlamaFFN: 3-120              [64, 64, 576]             2,654,208
├─RMSNorm: 1-3                           [64, 64, 576]             576
├─Linear: 1-4                            [64, 64, 49152]           28,311,552
==========================================================================================
Total params: 162,826,560
Trainable params: 162,826,560
Non-trainable params: 0
Total mult-adds (G): 10.42
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 7876.90
Params size (MB): 651.31
Estimated Total Size (MB): 8528.24
==========================================================================================

```

# Training Logs
## Training with 5000 steps (Starting from step 0)
```bash
(venv) gitesh.grover@Giteshs-MacBook-Pro ai-era-assignment13 % python train.py


Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 720.56it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 562123.22it/s]
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 336.18it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 708129.25it/s]
Epoch: 0, Step: 0, Batch(micro): 0, Loss: 11.3113, Time: 1.45s, Token/s: 2824.73
Saved checkpoint at step 0
What is Gravity?
 JunMon
 observation,,,, observation,affin,,,, Treating,Seququestion,,,.,,,,,,
Epoch: 0, Step: 1, Batch(micro): 1, Loss: 10.2534, Time: 2.58s, Token/s: 1586.06
Epoch: 0, Step: 2, Batch(micro): 2, Loss: 10.3703, Time: 1.23s, Token/s: 3323.19
Epoch: 0, Step: 3, Batch(micro): 3, Loss: 9.3798, Time: 1.09s, Token/s: 3769.05
Epoch: 0, Step: 4, Batch(micro): 4, Loss: 8.9930, Time: 1.09s, Token/s: 3741.50
Epoch: 0, Step: 5, Batch(micro): 5, Loss: 8.7043, Time: 1.12s, Token/s: 3652.90
Epoch: 0, Step: 6, Batch(micro): 6, Loss: 8.5976, Time: 1.08s, Token/s: 3776.95
:
:
Epoch: 0, Step: 497, Batch(micro): 497, Loss: 6.1369, Time: 1.17s, Token/s: 3493.98
Epoch: 0, Step: 498, Batch(micro): 498, Loss: 5.6010, Time: 1.12s, Token/s: 3647.57
Epoch: 0, Step: 499, Batch(micro): 499, Loss: 5.8359, Time: 1.10s, Token/s: 3716.22
Epoch: 0, Step: 500, Batch(micro): 500, Loss: 5.7775, Time: 1.08s, Token/s: 3777.33
Saved checkpoint at step 500
What is Gravity? These to his device is the end of your fingers and its people. That's all about the body you can make, and give you ever, some
Epoch: 0, Step: 501, Batch(micro): 501, Loss: 5.8698, Time: 2.16s, Token/s: 1897.06
Epoch: 0, Step: 502, Batch(micro): 502, Loss: 6.0635, Time: 1.13s, Token/s: 3631.70
Epoch: 0, Step: 503, Batch(micro): 503, Loss: 6.5260, Time: 1.11s, Token/s: 3694.65
:
:
Epoch: 0, Step: 998, Batch(micro): 998, Loss: 5.7383, Time: 1.07s, Token/s: 3812.48
Epoch: 0, Step: 999, Batch(micro): 999, Loss: 5.8485, Time: 1.09s, Token/s: 3753.90
Epoch: 0, Step: 1000, Batch(micro): 1000, Loss: 6.2793, Time: 1.10s, Token/s: 3718.65
Saved checkpoint at step 1000
What is Gravity? They then, consider the class, many people might work together. After they, remember their feet, I loved you, what happens to think how they
Epoch: 0, Step: 1001, Batch(micro): 1001, Loss: 5.7521, Time: 2.86s, Token/s: 1431.97
:
:
Epoch: 0, Step: 1500, Batch(micro): 1500, Loss: 5.8363, Time: 1.06s, Token/s: 3868.92
Saved checkpoint at step 1500
What is Gravity?

Imagine being something difficult to be doing so incredible artists to keep't enough their emotions. Imagine walking by others have all this person has never always
Epoch: 0, Step: 1501, Batch(micro): 1501, Loss: 6.0452, Time: 1.49s, Token/s: 2740.78

:
:
:
:

Epoch: 0, Step: 3499, Batch(micro): 3499, Loss: 5.7676, Time: 1.06s, Token/s: 3853.71
Epoch: 0, Step: 3500, Batch(micro): 3500, Loss: 6.0451, Time: 1.06s, Token/s: 3856.43
Saved checkpoint at step 3500
What is Gravity? Well does you do your own body, a small way of your friend, or maybe we can work to create the surface.
7. **G
Epoch: 0, Step: 3501, Batch(micro): 3501, Loss: 5.7938, Time: 1.38s, Token/s: 2959.03
Epoch: 0, Step: 3502, Batch(micro): 3502, Loss: 6.3508, Time: 1.08s, Token/s: 3785.58

:
:
Epoch: 0, Step: 4498, Batch(micro): 4498, Loss: 5.6528, Time: 1.06s, Token/s: 3870.78
Epoch: 0, Step: 4499, Batch(micro): 4499, Loss: 6.1692, Time: 1.06s, Token/s: 3850.22
Epoch: 0, Step: 4500, Batch(micro): 4500, Loss: 5.6509, Time: 1.08s, Token/s: 3784.61
Saved checkpoint at step 4500
What is Gravity?

Have you ever seen how they go at home as the bustling forest. How do you really your way, what makes it would affect! Well
Epoch: 0, Step: 4501, Batch(micro): 4501, Loss: 5.7961, Time: 1.49s, Token/s: 2746.98
Epoch: 0, Step: 4502, Batch(micro): 4502, Loss: 5.8517, Time: 1.08s, Token/s: 3804.95
Epoch: 0, Step: 4503, Batch(micro): 4503, Loss: 6.3001, Time: 1.06s, Token/s: 3872.11
:
:
Epoch: 0, Step: 4999, Batch(micro): 4999, Loss: 6.1256, Time: 1.06s, Token/s: 3853.00
Epoch: 0, Step: 5000, Batch(micro): 5000, Loss: 6.0105, Time: 1.07s, Token/s: 3833.68
Saved checkpoint at step 5000
What is Gravity?
Now that you know that there were many people around you might have to be to ask from us. By understanding more efficiently, we've witnessed some
Saved final checkpoint
What is Gravity? Well, I know that, they are plenty of people! This can learn that everyone's own own opinions –, I promise to learn these essential ones
Saved the trained model
Training complete


```

## Training with Additional 50 steps (Starting from checkpoint )
```bash
Last Saved epoch 0 and step 5000 with loss 6.010481357574463
Resuming from epoch 0 and next step 5001 with loss 6.010481357574463
Resolving data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 442.22it/s]
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 334258.71it/s]
Epoch: 0, Step: 5001, Batch(micro): 5001, Loss: 5.8380, Time: 3.07s, Token/s: 1334.43
Epoch: 0, Step: 5002, Batch(micro): 5002, Loss: 6.2662, Time: 1.10s, Token/s: 3736.21
Epoch: 0, Step: 5003, Batch(micro): 5003, Loss: 6.0323, Time: 1.09s, Token/s: 3762.44
Epoch: 0, Step: 5004, Batch(micro): 5004, Loss: 6.1844, Time: 1.20s, Token/s: 3407.99
Epoch: 0, Step: 5005, Batch(micro): 5005, Loss: 5.8835, Time: 1.09s, Token/s: 3743.08
Epoch: 0, Step: 5006, Batch(micro): 5006, Loss: 5.6408, Time: 1.09s, Token/s: 3751.67
Epoch: 0, Step: 5007, Batch(micro): 5007, Loss: 5.7871, Time: 1.08s, Token/s: 3782.06
:
:
:
Epoch: 0, Step: 5047, Batch(micro): 5047, Loss: 6.0360, Time: 1.11s, Token/s: 3702.12
Epoch: 0, Step: 5048, Batch(micro): 5048, Loss: 5.8714, Time: 1.09s, Token/s: 3755.10
Epoch: 0, Step: 5049, Batch(micro): 5049, Loss: 5.8596, Time: 1.09s, Token/s: 3769.04
Epoch: 0, Step: 5050, Batch(micro): 5050, Loss: 5.8586, Time: 1.09s, Token/s: 3746.08
Saved final checkpoint
What is Gravity? Well for sharing, we're going to understand this fascinating adventure, and it means of our beliefs: the world of a special. Now that our planet
Saved the trained model
Training complete

```
