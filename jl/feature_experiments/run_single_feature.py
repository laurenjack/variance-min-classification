import torch

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.config import Config
from jl.single_runner import train_once


def main():
    VAL_TO_SHOW = 32
    
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
        num_layers=4,
        num_class=problem.num_classes(),
        h=40,
        weight_tracker="accuracy",
        width_varyer=None,
        is_norm=True,
        optimizer="adam_w",
        learnable_norm_parameters=True,
        lr_scheduler=None,
        # c=0.5,
        # dropout_prob=0.5,
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
    
    def print_validation_probs(probs_tensor, labels, model_name, num_classes, samples_per_row=8):
        """
        Print validation probabilities in rows for easy viewing.
        
        Args:
            probs_tensor: Probability tensor of shape [num_samples, num_classes] or [num_samples] for binary
            labels: True labels of shape [num_samples]
            model_name: Name of the model (for header)
            num_classes: Number of classes (2 for binary, >2 for multi-class)
            samples_per_row: Number of samples to print per row
        """
        labels_list = labels.cpu().tolist()
        
        if num_classes == 2:
            # Binary classification: probs_tensor is [num_samples]
            # For binary, we show the probability of the positive class
            # If label is 1, show prob directly; if label is 0, show 1-prob
            probs_list = probs_tensor.cpu().tolist()
            correct_probs_list = [prob if label == 1 else 1 - prob 
                                 for prob, label in zip(probs_list, labels_list)]
        else:
            # Multi-class: select probability of correct class for each example
            sample_indices = torch.arange(len(labels))
            correct_probs = probs_tensor[sample_indices, labels]
            correct_probs_list = correct_probs.cpu().tolist()
        
        print(f"\n{model_name} - Validation Probabilities (correct class)")
        print("-" * 60)
        
        num_samples = len(correct_probs_list)
        for i in range(0, num_samples, samples_per_row):
            row_end = min(i + samples_per_row, num_samples)
            row_probs = correct_probs_list[i:row_end]
            row_labels = labels_list[i:row_end]
            
            # Format: "sample_idx:label=prob"
            row_str = "  ".join([f"{i+j}:{lab}={prob:.3f}" 
                                for j, (lab, prob) in enumerate(zip(row_labels, row_probs))])
            print(row_str)
    
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

