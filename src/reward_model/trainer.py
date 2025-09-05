from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Define forward pass for reward model: get hidden states, take last token's hidden state, apply reward_head
def compute_reward_scores(model, input_ids, attention_mask, device):
    # We use the model's forward to get last hidden state. 
    # AutoModelForCausalLM returns logits by default, but we can get hidden states by passing output_hidden_states=True.
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    last_hidden_state = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_size]
    # Take hidden state of last non-padded token for each sequence. We find index of last token via attention mask.
    seq_lengths = attention_mask.sum(dim=1) - 1  # index of last token for each sequence
    last_token_hidden = last_hidden_state[torch.arange(last_hidden_state.size(0), device=device), seq_lengths]
    rewards = model.reward_head(last_token_hidden)  # shape: [batch_size, 1]
    return rewards

def train(model, train_loader, val_loader, c, device):
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=c.learning_rate, weight_decay=c.weight_decay
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, c.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)
        for step, (input_ids, attention_mask) in enumerate(progress_bar, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            # Compute reward scores for all sequences in the batch (which includes chosen and rejected pairs)
            rewards = compute_reward_scores(model, input_ids, attention_mask, device)
            # Split rewards into chosen and rejected halves
            batch_size = input_ids.size(0) // 2
            chosen_scores = rewards[:batch_size].view(-1)      # first half corresponds to chosen
            rejected_scores = rewards[batch_size:].view(-1)    # second half corresponds to rejected
            # Compute Bradley-Terry loss: -log(sigmoid(chosen - rejected))
            score_diff = (chosen_scores - rejected_scores).float()  # convert to float32 for stability
            # Target is 1 for all pairs (chosen is better than rejected)
            target = torch.ones_like(score_diff, device=device)
            loss = F.binary_cross_entropy_with_logits(score_diff, target)
            loss.backward()
            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1
            # Update progress bar description
            progress_bar.set_postfix({"loss": loss.item()})
            # Log training info every log_interval steps
            if step % c.log_interval == 0:
                avg_loss = total_loss / total_steps
                print(f"Epoch {epoch}, Step {step}: Avg Train Loss = {avg_loss:.4f}")
        # End of epoch, compute average training loss
        avg_train_loss = total_loss / total_steps if total_steps > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        # Validation after each epoch
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for input_ids, attention_mask in val_loader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    rewards = compute_reward_scores(model, input_ids, attention_mask, device)
                    batch_size = input_ids.size(0) // 2
                    chosen_scores = rewards[:batch_size].view(-1)
                    rejected_scores = rewards[batch_size:].view(-1)
                    score_diff = (chosen_scores - rejected_scores).float()
                    # Compute the same logistic loss on validation data
                    target = torch.ones_like(score_diff, device=device)
                    loss = F.binary_cross_entropy_with_logits(score_diff, target)
                    val_loss += loss.item()
                    val_steps += 1
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch} completed. Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            # Early stopping check
            if c.early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0  # reset counter if improvement
                    # Optionally save the best model
                    model_path = os.path.join(c.output_dir, "best_model.pt")
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= c.patience:
                        print(f"No improvement in validation loss for {c.patience} epochs. Early stopping at epoch {epoch}.")
                        break
        else:
            # If no validation set is provided
            print(f"Epoch {epoch} completed. Train Loss = {avg_train_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(c.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
