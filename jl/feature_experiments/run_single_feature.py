import torch

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.feature_experiments.report import print_validation_probs, print_grouped_by_percent_correct
from jl.config import Config
from jl.single_runner import train_once


def main():
    VAL_TO_SHOW = 32
    GROUP_BY_PERCENT_CORRECT = False
    
    # torch.manual_seed(38175)
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
    n = 256
    # n = sum(problem.n_per_f)

    # Model configuration
    model_config = Config(
        model_type='resnet',
        d=problem.d,
        n_val=128,
        n=n,
        batch_size=n // 4,
        lr=0.03,
        epochs=100,
        weight_decay=0.03,
        num_layers=8,
        num_class=problem.num_classes(),
        h=80,
        weight_tracker="accuracy",
        width_varyer=None,
        is_norm=True,
        optimizer="adam_w",
        learnable_norm_parameters=True,
        lr_scheduler=None,
        # c=0.5,
        # dropout_prob=0.2,
        is_hashed_dropout=False,
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

    num_classes = model_config.num_class
    if num_classes == 2:
        prob_fn = torch.sigmoid
    else:
        prob_fn = lambda x: torch.softmax(x, dim=1)
    
    if GROUP_BY_PERCENT_CORRECT:
        # Validation predictions grouped by percent_correct
        print("\n" + "=" * 60)
        print("VALIDATION PREDICTIONS (GROUPED BY percent_correct)")
        print("=" * 60)
        with torch.no_grad():
            model.eval()
            val_probs = prob_fn(model(x_val))
        print_grouped_by_percent_correct(val_probs, y_val, center_indices, problem, "Model", num_classes, val_to_show=VAL_TO_SHOW)
    else:
        # Validation predictions
        print("\n" + "=" * 60)
        print("VALIDATION PREDICTIONS")
        print("=" * 60)
        val_probs = prob_fn(model(x_val[:VAL_TO_SHOW]))
        print_validation_probs(val_probs, y_val[:VAL_TO_SHOW], "Model", num_classes)



    # # Compute mean confidence for each center/feature
    # with torch.no_grad():
    #     model.eval()
        
    #     # Get model predictions for validation data
    #     logits = model(x_val)
        
    #     # Apply softmax to get probabilities
    #     probs = torch.softmax(logits, dim=1)
        
    #     # Get confidence for the true label (center_indices) for each sample
    #     sample_indices = torch.arange(model_config.n_val)
    #     confidences = probs[sample_indices, center_indices]
        
    #     # Determine which samples are correctly labeled vs mislabeled
    #     is_correct = (y_val == center_indices)
        
    #     # Group by center and compute mean confidence for correctly labeled and mislabeled
    #     print("\nMean confidence by feature/center (confidence for true label):")
    #     print("-" * 60)
    #     for center_idx in range(problem.f):
    #         # Find all validation samples that belong to this center
    #         center_mask = (center_indices == center_idx)
            
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

