import torch

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.config import Config
from jl.double_runner import train_double


def main():
    VAL_TO_SHOW = 32
    
    # torch.manual_seed(38175)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SingleFeatures
    problem = SingleFeatures(
        true_d=4,
        f=8,
        device=device,
        is_orthogonal=False,
        percent_correct_per_f=[0.8] * 8,
        noisy_d=8,
    )
    n = 128

    # Model configuration (num_class must be > 2 for double_runner)
    model_config = Config(
        model_type='resnet',
        d=problem.d,
        n_val=128,
        n=n,
        batch_size=n // 2,
        lr=0.03,
        epochs=200,
        weight_decay=0.03,
        num_layers=4,
        num_class=problem.num_classes(),
        h=20,
        weight_tracker="accuracy",
        down_rank_dim=None,
        width_varyer=None,
        is_norm=True,
        optimizer="adam_w",
        learnable_norm_parameters=True,
        lr_scheduler=None,
        dropout_prob=0.2,
        # c=0.5,
        is_hashed_dropout=False,
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices, _ = problem.generate_dataset(
        model_config.n_val, 
        shuffle=True, 
        clean_mode=True
    )
    validation_set = (x_val, y_val, center_indices)

    # Train the models using double_runner
    model1, model2, x_train1, y_train1, x_train2, y_train2 = train_double(
        device, problem, validation_set, model_config
    )

    def print_validation_probs(probs_tensor, labels, model_name, samples_per_row=8):
        """
        Print validation probabilities in rows for easy viewing.
        
        Args:
            probs_tensor: Probability tensor of shape [num_samples, num_classes]
            labels: True labels of shape [num_samples]
            model_name: Name of the model (for header)
            samples_per_row: Number of samples to print per row
        """
        # Select probability of correct class for each example
        sample_indices = torch.arange(len(labels))
        correct_probs = probs_tensor[sample_indices, labels]
        correct_probs_list = correct_probs.cpu().tolist()
        
        print(f"\n{model_name} - Validation Probabilities (correct class)")
        print("-" * 60)
        
        num_samples = len(correct_probs_list)
        for i in range(0, num_samples, samples_per_row):
            row_end = min(i + samples_per_row, num_samples)
            row_probs = correct_probs_list[i:row_end]
            row_labels = labels[i:row_end].cpu().tolist()
            
            # Format: "sample_idx:label=prob"
            row_str = "  ".join([f"{i+j}:{lab}={prob:.3f}" 
                                for j, (lab, prob) in enumerate(zip(row_labels, row_probs))])
            print(row_str)
    
    # Model 1 validation predictions
    print("\n" + "=" * 60)
    print("MODEL 1 VALIDATION PREDICTIONS")
    print("=" * 60)
    val_probs1 = torch.softmax(model1(x_val[:VAL_TO_SHOW]), dim=1)
    print_validation_probs(val_probs1, y_val[:VAL_TO_SHOW], "Model 1")

    # Model 2 validation predictions
    print("\n" + "=" * 60)
    print("MODEL 2 VALIDATION PREDICTIONS")
    print("=" * 60)
    val_probs2 = torch.softmax(model2(x_val[:VAL_TO_SHOW]), dim=1)
    print_validation_probs(val_probs2, y_val[:VAL_TO_SHOW], "Model 2")

    # Ensemble validation predictions
    print("\n" + "=" * 60)
    print("ENSEMBLE VALIDATION PREDICTIONS")
    print("=" * 60)
    with torch.no_grad():
        val_logits1 = model1(x_val[:VAL_TO_SHOW])
        val_logits2 = model2(x_val[:VAL_TO_SHOW])
        ensemble_logits = (val_logits1 + val_logits2) / 2
        ensemble_probs = torch.softmax(ensemble_logits, dim=1)
    print_validation_probs(ensemble_probs, y_val[:VAL_TO_SHOW], "Ensemble")


if __name__ == "__main__":
    main()

