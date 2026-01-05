import torch

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.config import Config
from jl.single_runner import train_once


def main():
    torch.manual_seed(38173)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SingleFeatures
    clean_mode = False
    problem = SingleFeatures(
        true_d=2,
        f=2, 
        device=device,
        is_orthogonal=True,
        n_per_f=[4, 8],
        # n_per_f=[4, 8, 16, 32, 4, 8, 16, 32]
        # percent_correct_per_f=[0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9],
        # noisy_d=16,
    )
    # n = 128
    n = sum(problem.n_per_f)

    # Model configuration
    model_config = Config(
        model_type='multi-linear',
        d=problem.d,
        n_val=n,
        n=n,
        batch_size=n,
        lr=0.1,
        epochs=500,
        weight_decay=0.0,
        num_layers=0,
        num_class=problem.num_classes(),
        h= problem.f,
        weight_tracker=None,
        down_rank_dim=None,
        width_varyer=None,
        is_norm=True,
        optimizer="sgd",
        c=0.03,
        learnable_norm_parameters=False,
        lr_scheduler=None,
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices, _ = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        clean_mode=True
    )
    validation_set = (x_val, y_val, center_indices)

    # Train the model using single_runner
    model, tracker, x_train, y_train, train_center_indices, _ = train_once(
        device, problem, validation_set, model_config, clean_mode=clean_mode
    )

    print("ALL probs:" , torch.sigmoid(model(x_train)))

    # # Compute mean confidence for each center/feature
    # with torch.no_grad():
    #     model.eval()
        
    #     # Get model predictions for training data
    #     logits = model(x_train)
        
    #     # Apply softmax to get probabilities
    #     probs = torch.softmax(logits, dim=1)
        
    #     # Get confidence for the true label (center_indices) for each sample
    #     sample_indices = torch.arange(n)
    #     confidences = probs[sample_indices, train_center_indices]
        
    #     # Determine which samples are correctly labeled vs mislabeled
    #     is_correct = (y_train == train_center_indices)
        
    #     # Group by center and compute mean confidence for correctly labeled and mislabeled
    #     print("\nMean confidence by feature/center (confidence for true label):")
    #     print("-" * 60)
    #     for center_idx in range(problem.f):
    #         # Find all training samples that belong to this center
    #         center_mask = (train_center_indices == center_idx)
            
    #         if center_mask.sum() > 0:
    #             # Split into correctly labeled and mislabeled
    #             correct_mask = center_mask & is_correct
    #             mislabeled_mask = center_mask & ~is_correct
                
    #             correct_confidences = confidences[correct_mask]
    #             mislabeled_confidences = confidences[mislabeled_mask]
                
    #             # Build output string
    #             parts = []
    #             if correct_confidences.numel() > 0:
    #                 mean_correct = correct_confidences.mean().item()
    #                 n_correct = correct_confidences.numel()
    #                 parts.append(f"correct: {mean_correct:.6f} (n={n_correct})")
                
    #             if mislabeled_confidences.numel() > 0:
    #                 mean_mislabeled = mislabeled_confidences.mean().item()
    #                 n_mislabeled = mislabeled_confidences.numel()
    #                 parts.append(f"mislabeled: {mean_mislabeled:.6f} (n={n_mislabeled})")
                
    #             print(f"Center {center_idx}: {', '.join(parts)}")
    #         else:
    #             print(f"Center {center_idx}: No samples")
        
    #     print("-" * 60)
        
    #     # Overall statistics split by correct vs mislabeled
    #     overall_correct_confidences = confidences[is_correct]
    #     overall_mislabeled_confidences = confidences[~is_correct]
        
    #     overall_parts = []
    #     if overall_correct_confidences.numel() > 0:
    #         mean_correct = overall_correct_confidences.mean().item()
    #         n_correct = overall_correct_confidences.numel()
    #         overall_parts.append(f"correct: {mean_correct:.6f} (n={n_correct})")
        
    #     if overall_mislabeled_confidences.numel() > 0:
    #         mean_mislabeled = overall_mislabeled_confidences.mean().item()
    #         n_mislabeled = overall_mislabeled_confidences.numel()
    #         overall_parts.append(f"mislabeled: {mean_mislabeled:.6f} (n={n_mislabeled})")
        
    #     print(f"Overall mean confidence: {', '.join(overall_parts)}")
    #     print()

if __name__ == "__main__":
    main()

