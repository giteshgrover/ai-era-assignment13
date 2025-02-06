import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
from torchsummary import summary
from model import SmolLM2
from utils import get_device
from config import Config

class StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, block_size=512):
        self.dataset = load_dataset("smollm-ai/smollm-corpus", streaming=True)["train"]
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
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return model, tokenizer

def get_custom_tokenizer_n_model():
    model = SmolLM2(config=Config())
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # You can use a different tokenizer
    return model, tokenizer


    # return mask.to(device)

def compareModels(device):
    model1, tokenizer1 = get_pretrained_tokenizer_n_model()
    model1.to(device)
    model2, tokenizer2 = get_custom_tokenizer_n_model()
    model2.to(device)

    print("Model 1 - HuggingFaceTB/SmolLM2-135M:")
    print(model1)
    print("Model 2 - Custom SmolLM2-135M Model :")
    print(model2)

def train_model():
    device = get_device()

    compareModels(device)

    # Initialize model
    # model, tokenizer = get_pretrained_tokenizer_n_model()
    model, tokenizer = get_custom_tokenizer_n_model()
    model.to(device)
    vocab_size = tokenizer.vocab_size

    
    inputs = tokenizer.encode("What is Gravity?", return_tensors="pt").to(device)
    B, T = inputs.size()
    # # Create causal mask for inference
    # attention_mask = create_causal_mask(T).to(device)
    # # Expand mask for batch size and number of heads
    # attention_mask = attention_mask.view(1, 1, T, T).expand(B, -1, -1, -1)

    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))
    

    # # Initialize dataset and dataloader
    # dataset = StreamingDataset(tokenizer)
    # dataloader = DataLoader(dataset, batch_size=8)

    # # Training parameters
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = nn.CrossEntropyLoss()

    # # Training loop
    # model.train()
    # for epoch in range(10):
    #     for batch_idx, batch in enumerate(dataloader):
    #         batch = batch.to(device)
            
    #         # Create targets (shifted by 1 position)
    #         targets = batch[:, 1:].contiguous()
    #         inputs = batch[:, :-1].contiguous()

    #         # Forward pass
    #         outputs = model(inputs)
    #         loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if batch_idx % 100 == 0:
    #             print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_model()