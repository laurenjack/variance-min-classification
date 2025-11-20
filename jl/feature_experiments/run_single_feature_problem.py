import torch
import torch.nn as nn

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.config import Config
from jl.single_runner import train_once


def main():
    # torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SingleFeatures
    clean_mode = False
    problem = SingleFeatures(
        d=5,
        f=5, 
        device=device
    )

    # Model configuration
    model_config = Config(
        model_type='multi-linear',
        d=problem.d,
        n_val=20,
        n=20,
        batch_size=20,
        lr=1 * 1e-3,
        epochs=1000,
        weight_decay=0.01,
        num_layers=1,
        num_class=problem.num_classes(),
        h=problem.f,
        is_weight_tracker=False,
        down_rank_dim=None,
        width_varyer=None,  # Must be None for multi-linear
        is_norm=False,
        is_adam_w=True
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        clean_mode=True
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)

    # Train the model using single_runner
    model, tracker, x_train, y_train, train_center_indices = train_once(
        device, problem, validation_set, model_config, clean_mode=clean_mode
    )

    # Compute end-to-end weight matrix W_full (shape [f, d])
    # Only valid for linear networks without activations (multi-linear setup)
    with torch.no_grad():
        weights: list[torch.Tensor] = []

        # Collect linear weights in forward order
        if hasattr(model, 'layers') and isinstance(model.layers, nn.Sequential):
            for m in model.layers:
                if isinstance(m, nn.Linear):
                    weights.append(m.weight.detach())
        # Optional down-rank layer
        if hasattr(model, 'down_rank_layer') and isinstance(getattr(model, 'down_rank_layer', None), nn.Linear):
            weights.append(model.down_rank_layer.weight.detach())
        # Final classification layer (required)
        if hasattr(model, 'final_layer') and isinstance(model.final_layer, nn.Linear):
            weights.append(model.final_layer.weight.detach())
        else:
            raise RuntimeError("Model does not have a compatible final linear layer for W_full computation.")

        if len(weights) == 0:
            raise RuntimeError("No linear weights found to compose W_full.")

        # Compose end-to-end matrix: W_full = W_last @ ... @ W_first
        W_full = weights[0]
        for k in range(1, len(weights)):
            W_full = weights[k] @ W_full

        # Normalize each row (each of the f rows)
        row_norms = torch.norm(W_full, dim=1, keepdim=True)
        row_norms = torch.clamp(row_norms, min=1e-12)
        V = W_full / row_norms

        # Frame operator S ∈ R^{d×d}: S = sum_i v_i v_i^T = V^T V
        S = V.T @ V

        # Calculate Frobenius-cosine similarity between W_full and problem.Q
        Q = problem.Q
        # Flatten both matrices
        w_flat = W_full.flatten()
        q_flat = Q.flatten()

        # Cosine similarity = (w . q) / (|w| * |q|)
        dot_prod = torch.dot(w_flat, q_flat)
        norm_w = torch.norm(w_flat)
        norm_q = torch.norm(q_flat)

        frob_cosine_sim = dot_prod / (norm_w * norm_q + 1e-12)

        print(f"Frobenius-cosine similarity (W_full, Q): {frob_cosine_sim.item():.6f}")
        print(f"Frobenius-cosine error (1 - sim): {1.0 - frob_cosine_sim.item():.6f}")

        # Print resulting matrix S
        print("Frame operator S (shape {}):".format(tuple(S.shape)))
        print(S)


if __name__ == "__main__":
    main()

