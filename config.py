from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 49152 
    num_hidden_layers: int = 30 # number of layers
    num_attention_heads: int = 9 # number of heads
    nn_embed: int = 576 # embedding dimension or hidden_size
    max_sequence_len: int = 2048 # max token sequence length (for pos embedding) # Block size
    ffn_intermediate_size: int = 1536
    rms_norm_eps: float = 1.0e-05
    # checkpoint_interval: int = 2000
    # checkpoints_path = "checkpoints"
    # init_method_std: 0.041666666666666664
    # nn_train_tok_seq: int = 32 # Actual training token sequence
    # nn_mlp_expansion: int = 4 # Expansion in the MLP layer 
    # batch_size: int = 256
    # train_tok_size: int = 32
    # saved_model_path = 'data/model_tf.pth'
    # train_input_file = 'data/input.txt'