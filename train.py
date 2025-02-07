import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import torch.nn.functional as F
from torchsummary import summary
from model import SmolLM2
from utils import get_device
from config import Config
import os
import time

class StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, block_size=512):
        # self.dataset = load_dataset("smollm-ai/smollm-corpus", streaming=True)["train"]
        self.dataset =  load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True)["train"]
        
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        iterator = iter(self.dataset)
        buffer = []
        
        for item in iterator:
            tokens = self.tokenizer.encode(item['text'])
            buffer.extend(tokens)
            
            while len(buffer) >= self.block_size:
                yield torch.tensor(buffer[:self.block_size])
                buffer = buffer[self.block_size:]



def get_pretrained_tokenizer_n_model():
    checkpoint = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return model, tokenizer

def get_custom_tokenizer_n_model(config):
    model = SmolLM2(config=config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)  # You can use a different tokenizer
    return model, tokenizer


    # return mask.to(device)

def compareModels(device):
    model1, tokenizer1 = get_pretrained_tokenizer_n_model()
    model1.to(device)
    model2, tokenizer2 = get_custom_tokenizer_n_model(Config())
    model2.to(device)

    print("Model 1 - HuggingFaceTB/SmolLM2-135M:")
    print(model1)
    print("Model 2 - Custom SmolLM2-135M Model :")
    print(model2)

def test(model, tokenizer, device, config):
    inputs = tokenizer.encode("What is Gravity?", return_tensors="pt").to(device)
    B, T = inputs.size()

    outputs = model.generate(inputs, max_new_tokens=30, temperature=config.nn_temperature, top_k=config.nn_top_k)
    print(tokenizer.decode(outputs[0]))

def load_checkpoint(model, optimizer, checkpoint_path):
    # Use weights_only=True for safer loading
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    steps = checkpoint['steps']
    return start_epoch, steps, checkpoint['loss']

def save_checkpoint(model, optimizer, epoch, steps, loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps': steps,
        'loss': loss
    }, checkpoint_path)


def train_model():
    SEED = 49
    device = get_device(seed=SEED)

    # Compare Actual HuggingFaceTB/SmolLM2-135M with my model for parameters and layers
    compareModels(device)

    # Speed up
    torch.set_float32_matmul_precision('high')

    # Initialize model
    # model, tokenizer = get_pretrained_tokenizer_n_model()
    config = Config()
    model, tokenizer = get_custom_tokenizer_n_model(config)
    model.to(device)
    torch.compile(model) # As per the class, torch.compile doesn't work for Windows or Mac, but it appears to be working for Mac M4Pro
    vocab_size = tokenizer.vocab_size
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # TODO LR update

    # Try to load checkpoint if it exists
    start_epoch = 0
    steps = 0
    checkpoint_path = config.checkpoints_path + '/checkpoint_final.pt'  # or specify a specific checkpoint like 'checkpoint_step_500.pt'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, steps, loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming from epoch {start_epoch} at step {steps} with loss {loss}")

    # print("Testing the model before training...")
    # test(model, tokenizer, device, config)

    # Initialize dataset and dataloader
    dataset = StreamingDataset(tokenizer, block_size=config.nn_train_tok_seq)
    dataloader = DataLoader(dataset, batch_size=config.batch_size) 

    # Training loop
    model.train()
    # for epoch in range(start_epoch, 10):  # For this assignment we are not going to go over 1 epoch
    epoch = start_epoch

    max_steps = 20 # TODO change it to 5000
    if steps > 0:
        max_steps = steps + 10 # TODO change it to 500

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        start_time = time.time()
        
        # Create targets (shifted by 1 position)
        targets = batch[:, 1:].contiguous()
        inputs = batch[:, :-1].contiguous()

        optimizer.zero_grad()

        # Forward pass
        # Speed up - Auto Cast (Forward pass)
         # Modified autocast section to handle MPS properly
        if device.type == 'mps':
            # MPS doesn't fully support autocast yet, run without it
            outputs, loss = model(inputs, targets=targets)
        else:
            # Use autocast for CUDA and CPU
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs, loss = model(inputs, targets=targets)
        # Backward pass
        loss.backward()
        optimizer.step()

        end_time = time.time()
        token_per_second = (inputs.shape[1] * config.batch_size) / (end_time - start_time)
        print(f"Epoch: {epoch}, Step: {steps}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f}s, Token/s: {token_per_second:.2f}")
        if batch_idx % 9 == 0: #TODO change it to 500
            test(model, tokenizer, device, config)
        
        # Save checkpoint every 500 or config.checkpoint_interval  TODO  
        if steps % 5 == 0:
            save_checkpoint(model, optimizer, epoch, steps, loss, f'{config.checkpoints_path}/checkpoint_step_{steps}.pt')
            print(f"Saved checkpoint at step {steps}")
        
        steps += 1
        if (steps >= max_steps):
            test(model, tokenizer, device, config)
            #   Save final checkpoint
            save_checkpoint(model, optimizer, epoch, steps, loss, f'{config.checkpoints_path}/checkpoint_final.pt')
            print("Saved final checkpoint")
            break

    print("Training complete")

if __name__ == "__main__":
    train_model()