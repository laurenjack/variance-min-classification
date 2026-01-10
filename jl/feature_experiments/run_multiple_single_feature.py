import torch

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.config import Config
from jl.multi_runner import train_parallel

# Number of validation examples to compare across models
M = 10

# Number of training runs
NUM_RUNS = 20


def main():
    # torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SingleFeatures
    clean_mode = False
    problem = SingleFeatures(
        true_d=4,
        f=8,
        device=device,
        is_orthogonal=False,
        percent_correct_per_f=[0.8] * 8,
        noisy_d=8,
    )
    n = 128

    # Model configuration (same for all runs)
    model_config = Config(
        model_type='resnet',
        d=problem.d,
        n_val=n,
        n=n,
        batch_size=n // 2,
        lr=0.03,
        epochs=100,
        weight_decay=0.1,
        num_layers=3,
        num_class=problem.num_classes(),
        h=20,
        weight_tracker=None,  # Explicitly set to None
        down_rank_dim=None,
        width_varyer=None,
        is_norm=True,
        optimizer="adam_w",
        learnable_norm_parameters=False,
        lr_scheduler=None,
        dropout_prob=0.2,
    )

    # Generate a single validation set for comparison across all models
    x_val, y_val, center_indices, _ = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
        clean_mode=True
    )
    x_val, y_val = x_val.to(device), y_val.to(device)

    # Train NUM_RUNS models in parallel using train_parallel
    print(f"Training {NUM_RUNS} models in parallel...")
    vectorized_model, _ = train_parallel(
        device,
        problem,
        model_config,
        NUM_RUNS,
        clean_mode,
        validation_set=(x_val, y_val),  # Optional: print basic metrics
        width_range=None,  # No width variation
    )

    # Run the first M validation examples through all models and compare outputs
    print(f"\n{'='*60}")
    print(f"Comparing model outputs for first {M} validation examples")
    print(f"{'='*60}\n")

    num_class = model_config.num_class
    num_models = vectorized_model.num_models

    # Get the first M validation examples and broadcast to all models
    x_subset = x_val[:M]
    y_subset = y_val[:M]
    
    # Broadcast input: [M, d] -> [num_models, M, d]
    x_broadcast = vectorized_model.broadcast_input(x_subset)
    
    # Get probabilities from all models: [num_models, M] for binary or [num_models, M, num_class] for multi-class
    all_probs = vectorized_model.get_probabilities(x_broadcast)

    for i in range(M):
        y_i = int(y_subset[i].item())

        if num_class == 2:
            # Binary classification: probs has shape [num_models, M]
            probs = all_probs[:, i]  # Shape: [num_models]
            mean_prob = probs.mean().item()
        else:
            # Multi-class: probs has shape [num_models, M, num_class]
            # Extract probability of correct class for each model
            probs = all_probs[:, i, y_i]  # Shape: [num_models]
            mean_prob = probs.mean().item()
        
        print(f"Example {i+1} | y={y_i}: mean={mean_prob:.3f}")
        print(probs)
        print("")


if __name__ == "__main__":
    main()
