import os
import logging
import time
import torch
from dataclasses import dataclass
from typing import Optional
from jl.reward_model.data_downloader import download_data
from jl.reward_model.model import get_model
from jl.reward_model.trainer import train

# Configure logging with timestamps for CloudWatch/SageMaker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    hf_dataset: str = "Anthropic/hh-rlhf"                 # Official Anthropic HH-RLHF dataset
    subset_name: Optional[str] = "helpful-base"           # Dataset subset/configuration to use
    cache_dir: str = "./cache"                             # Cache directory for model/dataset
    max_length: int = 1024                                  # Max sequence length (tokens) for prompt+response (to truncate long dialogues)
    train_batch_size: int = 64
    eval_batch_size: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    log_interval: int = 100                                # Print training loss every 100 steps
    early_stopping: bool = True
    patience: int = 2                                      # Stop if no improvement in val loss for 2 consecutive evaluations
    output_dir: str = "./reward_model_output"
    log_timing: bool = True                                # Enable performance timing instrumentation
    smoke_test: bool = True                                # Run only 50 steps for quick validation


def main():
    c = Config()
    total_start = time.time()

    # Create output directory
    os.makedirs(c.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Download and prepare data
    logger.info("Loading and preparing dataset...")
    data_start = time.time()
    train_loader, val_loader = download_data(c)
    data_time = time.time() - data_start
    logger.info(f"Data loading completed in {data_time:.2f}s")

    # Load and prepare model
    logger.info("Loading model...")
    model_start = time.time()
    model = get_model(c, device)
    model_time = time.time() - model_start
    logger.info(f"Model loading completed in {model_time:.2f}s")

    # Start training
    logger.info("Starting training...")
    train_start = time.time()
    train(model, train_loader, val_loader, c, device)
    train_time = time.time() - train_start

    total_time = time.time() - total_start
    logger.info(f"Training completed! Total time: {total_time:.2f}s (data: {data_time:.2f}s, model: {model_time:.2f}s, train: {train_time:.2f}s)")


if __name__ == "__main__":
    main()