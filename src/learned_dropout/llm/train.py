import math
import torch
from torch.optim import AdamW

def train_model(model, train_seq_gen, val_sequences, tokenizer, 
                num_train_steps=1000, batch_size=8, grad_accum_steps=1, 
                learning_rate=1e-4, device="cuda", eval_interval=100):
    """
    Train the model on the given training sequence generator and evaluate on validation sequences.
    """
    model.to(device)
    model.train()
    # Use AdamW optimizer (Adam with weight decay)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # Set up mixed precision scaler for AMP
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    total_steps = num_train_steps
    print(f"Starting training for {total_steps} steps...")
    running_loss = 0.0
    for step in range(1, total_steps + 1):
        # Gradient accumulation inner loop
        optimizer.zero_grad()  # we'll accumulate manually, so reset here
        for micro_step in range(grad_accum_steps):
            try:
                # Get the next sequence batch from generator
                batch_sequences = [next(train_seq_gen) for _ in range(batch_size)]
            except StopIteration:
                print("Training data exhausted earlier than expected.")
                break  # exit if we run out of data
            
            # Convert batch to tensor (shape: batch_size x seq_len)
            batch_tensor = torch.tensor(batch_sequences, dtype=torch.long, device=device)
            # Use automatic mixed precision for forward pass
            with torch.cuda.amp.autocast():
                outputs = model(batch_tensor, labels=batch_tensor)
                # `outputs.loss` is the mean loss over tokens in the sequence
                loss = outputs.loss / grad_accum_steps  # scale loss for accumulation
            # Backpropagate with scaled loss
            scaler.scale(loss).backward()
            running_loss += loss.item()
        # Perform optimizer step with scaled gradients
        scaler.step(optimizer)
        scaler.update()
        # Optionally clip gradients (not strictly needed for small LR, but can be used)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Periodically evaluate on validation set
        if step % eval_interval == 0 or step == total_steps:
            model.eval()
            # Compute average validation loss
            val_loss = 0.0
            num_val = len(val_sequences)
            # Evaluate in batches to manage memory
            batch_count = 0
            with torch.no_grad():
                for i in range(0, num_val, batch_size):
                    batch = val_sequences[i:i+batch_size]
                    batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)
                    outputs = model(batch_tensor, labels=batch_tensor)
                    # Sum up the loss (already averaged per batch by model) times batch size
                    val_loss += outputs.loss.item() * batch_tensor.size(0)
                    batch_count += batch_tensor.size(0)
            val_loss = val_loss / batch_count
            val_ppl = math.exp(val_loss)
            print(f"Step {step}/{total_steps} - Train loss: {running_loss/step:.4f} - Val loss: {val_loss:.4f} - Val PPL: {val_ppl:.2f}")
            model.train()
    print("Training complete.")
