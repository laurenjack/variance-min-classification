import torch

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.feature_experiments.report import print_validation_probs, print_grouped_by_percent_correct
from jl.config import Config
from jl.ten_runner import train_multi


def main():
    VAL_TO_SHOW = 64
    GROUP_BY_PERCENT_CORRECT = False
    
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

    # Model configuration (num_class must be > 2 for ten_runner)
    model_config = Config(
        model_type='resnet',
        d=problem.d,
        n_val=128,
        n=n,
        batch_size=n // 4,
        lr=0.03,
        epochs=50,
        weight_decay=0.03,
        num_layers=4,
        num_class=problem.num_classes(),
        h=20,
        weight_tracker=None,  # No plots for 10 models
        width_varyer=None,
        is_norm=True,
        optimizer="adam_w",
        learnable_norm_parameters=True,
        lr_scheduler=None,
        is_hashed_dropout=False,
        prob_weight=3.0,
        num_models=10,
    )

    # Generate validation set with class-balanced sampling
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val,
        shuffle=True,
        clean_mode=True
    )
    validation_set = (x_val, y_val, center_indices)

    # Train the models using train_multi
    models = train_multi(device, problem, validation_set, model_config)

    num_classes = model_config.num_class

    if GROUP_BY_PERCENT_CORRECT:
        # Ensemble validation predictions grouped by percent_correct
        print("\n" + "=" * 60)
        print("ENSEMBLE VALIDATION PREDICTIONS (GROUPED BY percent_correct)")
        print("=" * 60)
        with torch.no_grad():
            all_logits = torch.stack([model(x_val) for model in models], dim=0)
            ensemble_logits = all_logits.mean(dim=0)
            ensemble_probs = torch.softmax(ensemble_logits, dim=1)
        print_grouped_by_percent_correct(ensemble_probs, y_val, center_indices, problem, "Ensemble", num_classes, val_to_show=VAL_TO_SHOW)
    else:
        # Ensemble validation predictions
        print("\n" + "=" * 60)
        print("ENSEMBLE VALIDATION PREDICTIONS")
        print("=" * 60)
        with torch.no_grad():
            all_logits = torch.stack([model(x_val[:VAL_TO_SHOW]) for model in models], dim=0)
            ensemble_logits = all_logits.mean(dim=0)
            ensemble_probs = torch.softmax(ensemble_logits, dim=1)
        print_validation_probs(ensemble_probs, y_val[:VAL_TO_SHOW], "Ensemble", num_classes)


if __name__ == "__main__":
    main()

