from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 49
    vocab_size: int = 49152 # it should match the vocab size of the tokenizer
    num_hidden_layers: int = 30 # number of layers
    num_attention_heads: int = 9 # number of heads
    num_key_value_heads: int = 3 # number of key and value heads
    nn_embed: int = 576 # embedding dimension or hidden_size
    max_sequence_len: int = 2048 # max token sequence length (for pos embedding) # Block size
    ffn_intermediate_size: int = 1536
    rms_norm_eps: float = 1.0e-05
    nn_top_k: int = 50 # top k for the model
    nn_temperature: float = 1.0 # temperature for the model
    tokenizer_name_or_path: str = "HuggingFaceTB/cosmo2-tokenizer"
    checkpoints_path = "checkpoints"
    init_method_std = 0.041666666666666664
    nn_train_tok_seq: int = 1024 # 2048 64 Actual training token sequence block size 64 + 1 as we are shifting the targets by 1
    # nn_mlp_expansion: int = 4 # Expansion in the MLP layer 
    micro_batch_size: int = 2
    intended_batch_size: int = 8
    optimizer_learning_rate_scheduler_learning_rate: float = 0.003
    optimizer_learning_rate_scheduler_lr_decay_starting_step: int = 1600000
    optimizer_learning_rate_scheduler_lr_decay_steps: int = 400000
    optimizer_learning_rate_scheduler_lr_decay_style: str = "linear"
    optimizer_learning_rate_scheduler_lr_warmup_steps: int = 2000
    optimizer_learning_rate_scheduler_lr_warmup_style: str = "linear"
    optimizer_learning_rate_scheduler_min_decay_lr: float = 0
    optimizer_factory_adam_beta1: float = 0.9
    optimizer_factory_adam_beta2: float = 0.95
    optimizer_factory_adam_eps: float = 1.0e-08
    optimizer_factory_name: str = "adamW"
    optimizer_factory_torch_adam_is_fused: bool = True
    optimizer_weight_decay: float = 0.01
    optimizer_zero_stage: int = 0
    optimizer_clip_grad: float = 1.0