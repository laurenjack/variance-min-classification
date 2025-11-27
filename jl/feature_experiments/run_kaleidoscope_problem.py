import torch

from jl.feature_experiments.feature_problem import Kaleidoscope
from jl.config import Config
from jl.single_runner import train_once


def main():
    torch.manual_seed(32912)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_mode = True

    problem = Kaleidoscope(
        d=10,
        centers=[10, 10],
        device=device,
    )

    n = 60
    model_config = Config(
        model_type="resnet",
        d=problem.d,
        n_val=n,
        n=n,
        batch_size=n // 4,
        lr=0.01,
        epochs=5000,
        weight_decay=0.1,
        num_layers=1,
        num_class=problem.num_classes(),
        h=20,
        is_weight_tracker=False,
        down_rank_dim=None,
        width_varyer=None,
        is_norm=True,
        is_adam_w=False,
        learnable_norm_parameters=False,
    )

    x_val, y_val, val_center_indices = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
        clean_mode=clean_mode,
    )
    validation_set = (
        x_val,
        y_val,
        val_center_indices,
    )

    model, tracker, x_train, y_train, train_center_indices = train_once(
        device,
        problem,
        validation_set,
        model_config,
        clean_mode=clean_mode,
    )

    # Compute confidence statistics for each center in each layer
    with torch.no_grad():
        model.eval()
        
        # Get model predictions for training data
        logits = model(x_train)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get confidence (max probability) for each sample
        confidences, _ = torch.max(probs, dim=1)
        
        # Iterate through each layer
        print("\nConfidence statistics by layer and center:")
        print("=" * 60)
        
        for layer_idx, num_centers in enumerate(problem.centers):
            print(f"\nLayer {layer_idx} (C_{layer_idx} = {num_centers} centers):")
            print("-" * 60)
            
            layer_center_indices = train_center_indices[layer_idx]
            
            for center_idx in range(num_centers):
                # Find all training samples that belong to this center at this layer
                mask = (layer_center_indices == center_idx)
                center_confidences = confidences[mask]
                
                if center_confidences.numel() > 0:
                    min_conf = center_confidences.min().item()
                    mean_conf = center_confidences.mean().item()
                    max_conf = center_confidences.max().item()
                    num_samples = center_confidences.numel()
                    print(f"  Center {center_idx}: min={min_conf:.6f}, mean={mean_conf:.6f}, max={max_conf:.6f} (n={num_samples})")
                else:
                    print(f"  Center {center_idx}: No samples")
        
        print("=" * 60)
        print(f"Overall mean confidence: {confidences.mean().item():.6f}")
        print()


if __name__ == "__main__":
    main()

