import torch

from src.learned_dropout.data_generator import TwoDirections
from src.learned_dropout.config import Config
from src.learned_dropout.sense_check import train_once


def main(analyze_weights: bool = True):
    # torch.manual_seed(4454)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    percent_correct = 1.0
    problem = TwoDirections(
        true_d=1,
        noisy_d=128,
        percent_correct=percent_correct,
        sigma=0.0,
        random_basis=True,
        device=device
    )

    # Model configuration
    model_config = Config(
        model_type='mlp',
        d=problem.d,
        n_val=1000,
        n=128,
        batch_size=16,
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        num_layers=0,
        h=None,
        is_weight_tracker=False,
        d_model=100,
        down_rank_dim=None,
        is_norm=True
    )

    # Generate validation set without label noise (use_percent_correct=False)
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val, 
        use_percent_correct=False,
        shuffle=True
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)

    # Train the model using sense_check (with label noise in training)
    model, tracker, x_train, y_train, train_center_indices = train_once(
        device, problem, validation_set, model_config, use_percent_correct=True
    )
    
    # Stop here if analyze_weights is False
    if not analyze_weights:
        return model, tracker, x_train, y_train, train_center_indices
    
    # Print magnitude of final weight matrix
    final_weight = model.final_layer.weight.data  # Shape: (1, d)
    weight_magnitude = torch.norm(final_weight).item()
    print(f"Final weight vector shape: {final_weight.shape}")
    print(f"\nFinal weight vector magnitude (L2 norm): {weight_magnitude:.3f}")
    
    # Analyze center direction vs learned weights
    print("\n" + "="*80)
    print("CENTER DIRECTION vs LEARNED WEIGHTS ANALYSIS")
    print("="*80)
    
    # TwoDirections has centers at μ and -μ where μ is a unit vector in true_d dimensions
    mu = problem.mu  # Shape: (true_d,)
    
    # If random_basis is enabled, we need to rotate μ to compare with learned weights
    if problem.random_basis and problem.basis is not None:
        # Pad μ with zeros for noisy dimensions, then rotate
        mu_padded = torch.zeros(problem.d, device=device)
        mu_padded[:problem.true_d] = mu
        mu_rotated_full = mu_padded @ problem.basis  # Shape: (d,)
        
        print(f"\nCenter 0 at +μ (class 0), rotated by basis")
        print(f"Center 1 at -μ (class 1), rotated by basis")
        print(f"\nOriginal μ direction (true_d={problem.true_d}):")
        print(f"  Original μ L2 norm: {torch.norm(mu).item():.6f}")
        
        # The optimal discriminant should align with the rotated μ
        cos_sim = torch.nn.functional.cosine_similarity(
            mu_rotated_full.unsqueeze(0), 
            final_weight, 
            dim=1
        ).item()
        
        print(f"\nRotated μ (after basis transformation):")
        print(f"  Rotated μ L2 norm:     {torch.norm(mu_rotated_full).item():.6f}")
        print(f"  Learned weights norm:  {torch.norm(final_weight).item():.6f}")
        print(f"  Cosine similarity (rotated μ vs weights): {cos_sim:.6f}")
        
        # Show components
        print(f"\nFirst 5 components of original μ:  {mu[:5].cpu().numpy()}")
        print(f"First 5 components of rotated μ:   {mu_rotated_full[:5].cpu().numpy()}")
        print(f"First 5 learned weights:            {final_weight[0, :5].cpu().numpy()}")
        
    else:
        # No rotation - compare directly
        learned_weights_true = final_weight[0, :problem.true_d]  # Shape: (true_d,)
        learned_weights_noisy = final_weight[0, problem.true_d:]  # Shape: (noisy_d,)
        
        # Calculate cosine similarity between learned weights and μ
        cos_sim = torch.nn.functional.cosine_similarity(
            mu.unsqueeze(0), 
            learned_weights_true.unsqueeze(0), 
            dim=1
        ).item()
        
        print(f"\nCenter 0 at +μ (class 0)")
        print(f"Center 1 at -μ (class 1)")
        print(f"\nμ direction (true_d={problem.true_d}):")
        print(f"  μ L2 norm:                     {torch.norm(mu).item():.6f}")
        print(f"  Learned weights (true_d) norm: {torch.norm(learned_weights_true).item():.6f}")
        print(f"  Learned weights (noisy_d) norm:{torch.norm(learned_weights_noisy).item():.6f}")
        print(f"  Cosine similarity (μ vs weights): {cos_sim:.6f}")
        
        # Show a few components for inspection
        print(f"\nFirst 5 components of μ:       {mu[:5].cpu().numpy()}")
        print(f"First 5 learned weights (true): {learned_weights_true[:5].cpu().numpy()}")
        print(f"First 5 learned weights (noisy):{learned_weights_noisy[:5].cpu().numpy()}")
    
    return model, tracker, x_train, y_train, train_center_indices
    
if __name__ == "__main__":
    main(analyze_weights=True)
