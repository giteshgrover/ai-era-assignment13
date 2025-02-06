from dataclasses import dataclass

@dataclass
class Config:
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
    # checkpoint_interval: int = 2000
    # checkpoints_path = "checkpoints"
    # init_method_std: 0.041666666666666664
    # nn_train_tok_seq: int = 32 # Actual training token sequence
    # nn_mlp_expansion: int = 4 # Expansion in the MLP layer 
    # batch_size: int = 256
    # train_tok_size: int = 32
    # saved_model_path = 'data/model_tf.pth'
    # train_input_file = 'data/input.txt'