import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import torch.nn.functional as F
from torchinfo import summary
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

def compareModels(device, config):
    model1, tokenizer1 = get_pretrained_tokenizer_n_model()
    model1.to(device)
    model2, tokenizer2 = get_custom_tokenizer_n_model(Config())
    model2.to(device)

    print("Model 1 - HuggingFaceTB/SmolLM2-135M:")
    printModelSummary(model1, config)
    print("Model 2 - Custom SmolLM2-135M Model :")
    printModelSummary(model2, config)

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
    
def printModelSummary(model, config):
    print(f"Model: {model}")
    summary(model, 
            input_size=(config.micro_batch_size, config.nn_train_tok_seq),
             dtypes=[torch.long])

def train_model():
    config = Config()
    SEED = config.seed
    device = get_device(seed=SEED)

    # Compare Actual HuggingFaceTB/SmolLM2-135M with my model for parameters and layers
    compareModels(device, config)

    # Speed up
    torch.set_float32_matmul_precision('high')

    # Initialize model
    # model, tokenizer = get_pretrained_tokenizer_n_model()
    model, tokenizer = get_custom_tokenizer_n_model(config)
    model.to(device)
    # printModelSummary(model, config)
    torch.compile(model) # As per the class, torch.compile doesn't work for Windows or Mac, but it appears to be working for Mac M4Pro
    vocab_size = tokenizer.vocab_size
    
    # Optimizer AdamW with weightDecay
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=config.optimizer_learning_rate_scheduler_learning_rate,
                                 weight_decay=config.optimizer_weight_decay,
                                 eps=config.optimizer_factory_adam_eps,
                                 betas=(config.optimizer_factory_adam_beta1, config.optimizer_factory_adam_beta2))

    # Try to load checkpoint if it exists
    start_epoch = 0
    start_step = 0
    checkpoint_path = config.checkpoints_path + '/checkpoint_final.pt'  # or specify a specific checkpoint like 'checkpoint_step_500.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, start_step, loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Last Saved epoch {start_epoch} and step {start_step} with loss {loss}")
        start_step = start_step + 1
        print(f"Resuming from epoch {start_epoch} and next step {start_step} with loss {loss}")

    # print("Testing the model before training...")
    # test(model, tokenizer, device, config)

    # Initialize dataset and dataloader
    dataset = StreamingDataset(tokenizer, block_size=config.nn_train_tok_seq)
    dataloader = DataLoader(dataset, batch_size=config.micro_batch_size + 1) # + 1 to get an extra token for token we use [0..n] for input and [1..n+1] for target

    # Training loop
    model.train()
    # for epoch in range(start_epoch, 10):  # For this assignment we are not going to go over 1 epoch
    epoch = start_epoch

    checkpoint_interval = 500
    max_steps = 5000
    if start_step > 0:
        max_steps = 5050 # id already trained for maxSteps, then add 50 more as per the assignment

    if start_step >= max_steps:
        print(f"Already Trained for max steps {max_steps}. Only testing and existing")
        test(model, tokenizer, device, config)
        return
        
    gradient_accumalate_steps = config.intended_batch_size // config.micro_batch_size
    print(f"gradient_accumalate_steps: {gradient_accumalate_steps}")
    optimizer.zero_grad() # TODO I am not sure if I need to call this one time in beginning when using the accumalating gradient
    for step, batch in enumerate(dataloader, start=start_step):
        batch = batch.to(device)
        start_time = time.time()
        
        # Create targets (shifted by 1 position)
        targets = batch[:, 1:].contiguous()
        inputs = batch[:, :-1].contiguous()
        # print(f"Inputs: {inputs.shape}, Targets: {targets.shape}")
        # print(f"inputs: {[inputs[0]]}")
        # print(f"targets: {[targets[0]]}")

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
        # What is the following line and who added it?
        #torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer_clip_grad)
        
        # Accumulate Gradient until meet the intended batch size or if its the last step
        if ((step + 1) % gradient_accumalate_steps == 0 or step >= max_steps):
            optimizer.step()
            optimizer.zero_grad()   

        end_time = time.time()
        token_per_second = (inputs.shape[1] * inputs.shape[0]) / (end_time - start_time)
        print(f"Epoch: {epoch}, Step: {step}, Batch(micro): {step}, Batch (considering grad accum): {step // gradient_accumalate_steps},  Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f}s, Token/s: {token_per_second:.2f}")
        # print(f"Epoch: {epoch}, Step: {step}, Batch(micro): {step}, Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f}s, Token/s: {token_per_second:.2f}")
        
        # Save checkpoint and Test the model every 500 steps
        if step % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, step, loss, f'{config.checkpoints_path}/checkpoint_step_{step}.pt')
            print(f"Saved checkpoint at step {step}")
            test(model, tokenizer, device, config)
        
        if (step  >= max_steps):
            #   Save final checkpoint 
            save_checkpoint(model, optimizer, epoch, step, loss, f'{config.checkpoints_path}/checkpoint_final.pt')
            print("Saved final checkpoint")
            test(model, tokenizer, device, config)
            # Save the model TODO Already Saved
            # torch.save(model.state_dict(), f'{config.checkpoints_path}/model_final.pt')
            # print("Saved the trained model")
            break

    print("Training complete")

if __name__ == "__main__":
    train_model()