import torch
import matplotlib.pyplot as plt

from jl.variance_experiments.data_generator import SubDirections
from jl.config import Config
from jl.single_runner import train_once


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(4454)
    # Problem configuration
    true_d = 12
    sub_d = 4
    centers = 6
    num_class = 2
    sigma = 0.02
    noisy_d = 0
    random_basis = True

    # Sample per-center percent_correct values uniformly from {0.6, 0.8, 1.0}
    percent_choices = torch.tensor([0.6, 0.8, 1.0], dtype=torch.float32, device=device)
    percent_choice_idx_per_center = torch.randint(
        low=0,
        high=percent_choices.numel(),
        size=(centers,),
        device=device,
    )
    percent_correct_per_center = percent_choices[percent_choice_idx_per_center]

    # Instantiate problem with per-center percent_correct
    problem = SubDirections(
        true_d=true_d,
        sub_d=sub_d,
        centers=centers,
        num_class=num_class,
        sigma=sigma,
        noisy_d=noisy_d,
        random_basis=random_basis,
        percent_correct=percent_correct_per_center,
        device=device,
    )

    # Model configuration (same style as sub_direction.py)
    model_config = Config(
        model_type='resnet',
        d=problem.d,
        n_val=1000,
        n=512,
        batch_size=64,
        lr=1e-3,
        epochs=300,
        weight_decay=0.001,
        num_layers=2,
        num_class=problem.num_classes(),
        h=80,
        is_weight_tracker=False,
        d_model=40,
        down_rank_dim=None
    )

    # Generate validation set with requested label noise
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val,
        clean_mode=True,
        shuffle=True,
    )

    # Train the model using single_runner (kept consistent with sub_direction.py)
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)
    model, _, _, _, _ = train_once(device, problem, validation_set, model_config, clean_mode=False)

    # Evaluate model predictions against the true center class (exclude label noise)
    model.eval()
    with torch.no_grad():
        logits = model(x_val.to(device)).squeeze()
        probs = torch.sigmoid(logits).cpu()
        preds = (probs >= 0.5).long()

    true_center_class = problem.center_to_class[center_indices].cpu()
    sample_choice_idx = percent_choice_idx_per_center[center_indices].cpu()
    is_correct = (preds == true_center_class)

    # Collect confidences for each bucket, separated by correctness
    bucket_confidences_correct = []
    bucket_confidences_incorrect = []
    bucket_labels = []
    
    print("Model accuracy vs true center class by percent_correct bucket:")
    for idx in range(percent_choices.numel()):
        bucket_mask = (sample_choice_idx == idx)
        num_samples = int(bucket_mask.sum().item())
        bucket_prob = float(percent_choices[idx].item())
        if num_samples == 0:
            print(f"  p={bucket_prob:.1f}: n=0, acc=NA")
            continue
        bucket_acc = (preds[bucket_mask] == true_center_class[bucket_mask]).float().mean().item()
        probs_for_bucket = probs[bucket_mask]
        # The confidence, in the direction of the prediction made
        confidences = torch.max(probs_for_bucket, 1.0 - probs_for_bucket)
        confidence_mean = confidences.mean().item()
        print(f"  p={bucket_prob:.1f}: n={num_samples}, acc={bucket_acc:.3f}, confidence={confidence_mean:.3f}")
        
        # Store for plotting, separated by correctness
        correct_mask = bucket_mask & is_correct
        incorrect_mask = bucket_mask & (~is_correct)
        
        confidences_all = torch.max(probs, 1.0 - probs)
        bucket_confidences_correct.append(confidences_all[correct_mask].numpy())
        bucket_confidences_incorrect.append(confidences_all[incorrect_mask].numpy())
        bucket_labels.append(f"p={bucket_prob:.1f}")
    
    # Create visualization of confidence distributions
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    colors = ['blue', 'green', 'red']
    
    # Plot 1: Correct predictions
    ax1 = axes[0]
    for idx, (confidences, label) in enumerate(zip(bucket_confidences_correct, bucket_labels)):
        if len(confidences) > 0:
            ax1.hist(confidences, bins=30, alpha=0.5, label=f"{label} (n={len(confidences)})", 
                    color=colors[idx], density=True)
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Confidence Distributions - Correct Predictions', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Incorrect predictions
    ax2 = axes[1]
    for idx, (confidences, label) in enumerate(zip(bucket_confidences_incorrect, bucket_labels)):
        if len(confidences) > 0:
            ax2.hist(confidences, bins=30, alpha=0.5, label=f"{label} (n={len(confidences)})", 
                    color=colors[idx], density=True)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Confidence Distributions - Incorrect Predictions', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('confidence_distributions.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved confidence distribution plot to: confidence_distributions.png")
    plt.show()


if __name__ == "__main__":
    main()


