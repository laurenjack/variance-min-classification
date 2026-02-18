import json
import logging
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks timing of different training phases to identify bottlenecks."""

    def __init__(self, device, enabled=True):
        self.device = device
        self.enabled = enabled
        self.use_cuda = device.type == 'cuda'
        self.reset()

    def reset(self):
        """Reset all accumulated times."""
        self.data_load_time = 0.0
        self.device_transfer_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.optimizer_time = 0.0
        self.step_count = 0

    def _sync_cuda(self):
        """Synchronize CUDA for accurate timing."""
        if self.use_cuda:
            torch.cuda.synchronize()

    def time_data_load(self, start_time):
        """Record time spent waiting for data."""
        if self.enabled:
            self.data_load_time += time.time() - start_time

    def time_device_transfer(self, start_time):
        """Record time spent transferring data to device."""
        if self.enabled:
            self._sync_cuda()
            self.device_transfer_time += time.time() - start_time

    def time_forward(self, start_time):
        """Record time spent in forward pass."""
        if self.enabled:
            self._sync_cuda()
            self.forward_time += time.time() - start_time

    def time_backward(self, start_time):
        """Record time spent in backward pass."""
        if self.enabled:
            self._sync_cuda()
            self.backward_time += time.time() - start_time

    def time_optimizer(self, start_time):
        """Record time spent in optimizer step."""
        if self.enabled:
            self._sync_cuda()
            self.optimizer_time += time.time() - start_time

    def step(self):
        """Increment step counter."""
        self.step_count += 1

    def get_summary(self):
        """Return timing summary as a formatted string."""
        total = self.data_load_time + self.device_transfer_time + self.forward_time + self.backward_time + self.optimizer_time
        if total == 0:
            return "No timing data"

        def pct(t):
            return f"{t:.2f}s ({100*t/total:.1f}%)"

        return (
            f"data_load={pct(self.data_load_time)}, "
            f"device_transfer={pct(self.device_transfer_time)}, "
            f"forward={pct(self.forward_time)}, "
            f"backward={pct(self.backward_time)}, "
            f"optimizer={pct(self.optimizer_time)}"
        )

    def get_interval_summary(self):
        """Return summary for logging interval, then reset."""
        summary = self.get_summary()
        self.reset()
        return summary


# Define forward pass for reward model: get hidden states, take last token's hidden state, apply reward_head
def compute_reward_scores(model, input_ids, attention_mask, device):
    """Compute reward scores for input sequences."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]  # shape: [batch_size, seq_len, hidden_size]
    seq_lengths = attention_mask.sum(dim=1) - 1  # index of last token for each sequence
    last_token_hidden = last_hidden_state[torch.arange(last_hidden_state.size(0), device=device), seq_lengths]
    rewards = model.reward_head(last_token_hidden)  # shape: [batch_size, 1]
    return rewards


def train(model, train_loader, val_loader, c, device, output_path: str, learning_rate: float = None, warmup_steps: int = None):
    """Train the reward model.

    Args:
        model: Reward model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        c: RewardConfig
        device: torch device
        output_path: Path to save the final model
        learning_rate: Learning rate override (uses config value if None)
        warmup_steps: Warmup steps override (uses config value if None)
    """
    lr = learning_rate if learning_rate is not None else c.learning_rate
    warmup = warmup_steps if warmup_steps is not None else c.warmup_steps
    logger.info(f"Using learning rate: {lr}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr, weight_decay=c.weight_decay
    )

    # LR scheduler: quadratic warmup then cosine decay
    total_steps = len(train_loader) * c.num_epochs
    min_lr_ratio = c.min_lr_ratio

    def lr_lambda(current_step):
        if current_step < warmup:
            # Quadratic warmup: (step / warmup_steps)^2
            return (current_step / max(1, warmup)) ** 2
        else:
            progress = (current_step - warmup) / max(1, total_steps - warmup)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"LR schedule: {warmup} warmup steps (quadratic), cosine decay to {min_lr_ratio:.0%} over {total_steps} total steps")

    # Performance tracking
    tracker = PerformanceTracker(device, enabled=c.log_timing)
    total_steps_in_loader = len(train_loader)

    # Metrics files for structured logging
    metrics_path = os.path.join(output_path, "metrics.jsonl")
    val_metrics_path = os.path.join(output_path, "val_metrics.jsonl")
    metrics_file = open(metrics_path, "w")
    val_metrics_file = open(val_metrics_path, "w")
    logger.info(f"Writing metrics to {metrics_path}")
    logger.info(f"Writing validation metrics to {val_metrics_path}")

    # Training loop
    train_losses = []
    val_losses = []
    global_step = 0

    for epoch in range(1, c.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_pairs = 0
        total_steps = 0
        epoch_start = time.time()
        tracker.reset()

        logger.info(f"Epoch {epoch}/{c.num_epochs} starting ({total_steps_in_loader} steps)")

        data_start = time.time()
        for step, (input_ids, attention_mask) in enumerate(train_loader, start=1):
            tracker.time_data_load(data_start)

            # Device transfer
            transfer_start = time.time()
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            tracker.time_device_transfer(transfer_start)

            optimizer.zero_grad()

            # Forward pass
            forward_start = time.time()
            rewards = compute_reward_scores(model, input_ids, attention_mask, device)
            # Split rewards into chosen and rejected halves
            batch_size = input_ids.size(0) // 2
            chosen_scores = rewards[:batch_size].view(-1)
            rejected_scores = rewards[batch_size:].view(-1)
            # Compute Bradley-Terry loss: -log(sigmoid(chosen - rejected))
            score_diff = (chosen_scores - rejected_scores).float()
            target = torch.ones_like(score_diff, device=device)
            loss = F.binary_cross_entropy_with_logits(score_diff, target)
            tracker.time_forward(forward_start)

            # Backward pass
            backward_start = time.time()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            tracker.time_backward(backward_start)

            # Optimizer step
            optimizer_start = time.time()
            optimizer.step()
            scheduler.step()
            tracker.time_optimizer(optimizer_start)

            total_loss += loss.item()
            total_correct += (chosen_scores > rejected_scores).sum().item()
            total_pairs += batch_size
            total_steps += 1
            global_step += 1
            tracker.step()

            # Log training info every log_interval steps
            if step % c.log_interval == 0:
                avg_loss = total_loss / total_steps
                train_acc = total_correct / total_pairs if total_pairs > 0 else 0.0
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch}, Step {step}/{total_steps_in_loader}: loss={avg_loss:.4f}, acc={train_acc:.1%}")
                if c.log_timing:
                    logger.info(f"  Timing (last {c.log_interval} steps): {tracker.get_interval_summary()}")
                # Write structured metrics for graphing
                metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(avg_loss, 6),
                    "accuracy": round(train_acc, 4),
                    "lr": current_lr
                }
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()

            # Smoke test: exit early after configured number of steps
            if c.smoke_test and global_step >= c.smoke_test_steps:
                logger.info(f"Smoke test: exiting after {global_step} steps")
                break

            # Start timing next data load
            data_start = time.time()

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / total_steps if total_steps > 0 else 0.0
        train_losses.append(avg_train_loss)

        # Smoke test: skip remaining epochs
        if c.smoke_test and global_step >= c.smoke_test_steps:
            avg_train_loss = total_loss / total_steps if total_steps > 0 else 0.0
            train_acc = total_correct / total_pairs if total_pairs > 0 else 0.0
            logger.info(f"Smoke test complete: {global_step} steps, loss={avg_train_loss:.4f}, acc={train_acc:.1%}")
            break

        # Validation and epoch logging
        if val_loader:
            val_start = time.time()
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_steps = 0
            with torch.no_grad():
                for input_ids, attention_mask in val_loader:
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    rewards = compute_reward_scores(model, input_ids, attention_mask, device)
                    batch_size = input_ids.size(0) // 2
                    chosen_scores = rewards[:batch_size].view(-1)
                    rejected_scores = rewards[batch_size:].view(-1)
                    score_diff = (chosen_scores - rejected_scores).float()
                    target = torch.ones_like(score_diff, device=device)
                    loss = F.binary_cross_entropy_with_logits(score_diff, target)
                    val_loss += loss.item()
                    val_correct += (chosen_scores > rejected_scores).sum().item()
                    val_total += batch_size
                    val_steps += 1
            val_time = time.time() - val_start
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            val_losses.append(avg_val_loss)
            # Write validation metrics
            val_metrics = {
                "epoch": epoch,
                "val_loss": round(avg_val_loss, 6),
                "val_accuracy": round(val_accuracy, 4)
            }
            val_metrics_file.write(json.dumps(val_metrics) + "\n")
            val_metrics_file.flush()
            logger.info(f"Epoch {epoch} complete: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_accuracy:.1%}, epoch_time={epoch_time:.1f}s (train={epoch_time-val_time:.1f}s, val={val_time:.1f}s)")
        else:
            logger.info(f"Epoch {epoch} complete: train_loss={avg_train_loss:.4f}, epoch_time={epoch_time:.1f}s")

    # Close metrics files
    metrics_file.close()
    val_metrics_file.close()
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Validation metrics saved to {val_metrics_path}")

    # Save final model
    final_model_path = os.path.join(output_path, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
