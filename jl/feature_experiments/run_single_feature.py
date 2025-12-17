from sympy import N
from sympy.logic import false
import torch
import torch.nn as nn

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.config import Config
from jl.single_runner import train_once


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SingleFeatures
    clean_mode = False
    problem = SingleFeatures(
        d=4,
        f=8, 
        device=device,
        orthogonal_as_possible=False,
        # n_per_f=[32, 16, 8, 4],
        n_per_f=[2, 4, 8, 16, 2, 4, 8, 16]
    )
    # n = 64
    n = sum(problem.n_per_f)

    # Model configuration
    model_config = Config(
        model_type='multi-linear',
        d=problem.d,
        n_val=n,
        n=n,
        batch_size=n // 4,
        lr=0.01,
        epochs=1000,
        weight_decay=0.01,
        num_layers=0,
        num_class=problem.num_classes(),
        h= problem.f,
        is_weight_tracker=False,
        down_rank_dim=None,
        width_varyer=None,
        is_norm=True,
        is_adam_w=False,
        learnable_norm_parameters=False,
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        clean_mode=True
    )
    validation_set = (x_val, y_val, center_indices)

    # Train the model using single_runner
    model, tracker, x_train, y_train, train_center_indices = train_once(
        device, problem, validation_set, model_config, clean_mode=clean_mode
    )

    # Compute mean confidence for each center/feature
    with torch.no_grad():
        model.eval()
        
        # Get model predictions for training data
        logits = model(x_train)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get confidence (max probability) for each sample
        confidences, _ = torch.max(probs, dim=1)
        
        # Group by center and compute mean confidence
        print("\nMean confidence by feature/center:")
        print("-" * 40)
        for center_idx in range(problem.f):
            # Find all training samples that belong to this center
            mask = (train_center_indices == center_idx)
            center_confidences = confidences[mask]
            
            if center_confidences.numel() > 0:
                mean_confidence = center_confidences.mean().item()
                num_samples = center_confidences.numel()
                print(f"Center {center_idx}: {mean_confidence:.6f} (n={num_samples})")
            else:
                print(f"Center {center_idx}: No samples")
        
        print("-" * 40)
        print(f"Overall mean confidence: {confidences.mean().item():.6f}")
        print()

if __name__ == "__main__":
    main()

