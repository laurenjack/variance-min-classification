import torch

from jl.feature_experiments.feature_problem import Kaleidoscope
from jl.config import Config
from jl.single_runner import train_once


def main():
    torch.manual_seed(32912)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = Kaleidoscope(
        d=10,
        centers=[10, 10, 10, 2],
        device=device,
    )

    n = 400
    model_config = Config(
        model_type="resnet",
        d=problem.d,
        n_val=n,
        n=n,
        batch_size=n,
        lr=0.03,
        epochs=1000,
        weight_decay=0.01,
        num_layers=3,
        num_class=problem.num_classes(),
        h=20,
        weight_tracker=None,
        width_varyer=None,
        is_norm=True,
        optimizer="sgd",
        learnable_norm_parameters=False,
    )

    x_val, y_val, val_center_indices, _ = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
    )
    validation_set = (
        x_val,
        y_val,
        val_center_indices,
    )

    model, tracker, x_train, y_train, train_center_indices, _ = train_once(
        device,
        problem,
        validation_set,
        model_config,
    )

    # Compute confidence statistics for each center in each layer
    with torch.no_grad():
        model.eval()
        
        # Get model predictions for training data
        logits = model(x_train)
        
        # Compute confidence based on classification type
        if model_config.num_class == 2:
            # Binary classification: logits have shape (batch_size,)
            # Use sigmoid to get probability, confidence is max(p, 1-p)
            probs = torch.sigmoid(logits)
            confidences = torch.max(probs, 1 - probs)
        else:
            # Multi-class: logits have shape (batch_size, num_classes)
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

