import os
import torch
from dataclasses import dataclass
from src.reward_model.data_downloader import download_data
from src.reward_model.model import get_model
from src.reward_model.trainer import train


@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"          # HuggingFace model ID - Qwen3 1.7B parameter model
    dataset_name: str = "Anthropic/hh-rlhf"               # Official Anthropic HH-RLHF dataset
    cache_dir: str = "./cache"                             # Cache directory for model/dataset
    max_length: int = 1024                                  # Max sequence length (tokens) for prompt+response (to truncate long dialogues)
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    log_interval: int = 100                                # Print training loss every 100 steps
    early_stopping: bool = True
    patience: int = 2                                      # Stop if no improvement in val loss for 2 consecutive evaluations
    output_dir: str = "./reward_model_output"


def main():
    c = Config()

    # Create output directory
    os.makedirs(c.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Download and prepare data
    print("Loading and preparing dataset...")
    train_loader, val_loader = download_data(c)
    
    # Load and prepare model
    print("Loading model...")
    model = get_model(c, device)
    
    # Start training
    print("Starting training...")
    train(model, train_loader, val_loader, c, device)
    
    print("Training completed!")


if __name__ == "__main__":
    main()