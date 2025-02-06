class SmolLM2Trainer:
    def __init__(self, model: SmolLM2, learning_rate: float = 3e-4, max_grad_norm: float = 1.0,):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        self.device = next(model.parameters()).device

    def train_step(self, batch) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Compute loss (using cross entropy)
        # Reshape logits and labels for cross entropy
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()

    def train(self, dataset, batch_size: int = 4, num_epochs: int = 1):
        """Training loop"""
        
        # Load dataset
        dataset = load_dataset("smollm-ai/smollm-corpus", streaming=True)
        train_dataset = dataset['train']
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True
        )
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
                
                if num_batches % 100 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch {epoch+1}, Batch {num_batches}, Average Loss: {avg_loss:.4f}")
            
            avg_epoch_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")