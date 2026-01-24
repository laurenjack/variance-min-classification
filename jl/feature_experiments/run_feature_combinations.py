import torch

from jl.feature_experiments.feature_combinations import FeatureCombinations
from jl.feature_experiments.feature_problem import Kaleidoscope
from jl.feature_experiments.report import print_validation_probs
from jl.config import Config
from jl.single_runner import train_once
from jl.ten_runner import train_multi


def main():
    IS_MULTI = True  # Set to True to train ensemble of 10 models
    VAL_TO_SHOW = 32
    
    torch.manual_seed(659)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # problem = FeatureCombinations(
    #     num_layers=3,
    #     random_basis=False,
    #     device=device,
    # )
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
        batch_size=n // 2,
        lr=0.01,
        epochs=100,
        weight_decay=0.01,
        num_layers=2,
        num_class=problem.num_classes(),
        h=40,
        weight_tracker="accuracy" if not IS_MULTI else None,
        width_varyer=None,
        is_norm=True,
        optimizer="adam_w",
        learnable_norm_parameters=True,
        lr_scheduler=None,
        num_models=10,
        prob_weight=1.0,    
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

    num_classes = model_config.num_class

    if IS_MULTI:
        models = train_multi(device, problem, validation_set, model_config)
        
        print("\n" + "=" * 60)
        print("ENSEMBLE VALIDATION PREDICTIONS")
        print("=" * 60)
        with torch.no_grad():
            all_logits = torch.stack([model(x_val[:VAL_TO_SHOW]) for model in models], dim=0)
            ensemble_logits = all_logits.mean(dim=0)
            val_probs = torch.softmax(ensemble_logits, dim=1)
        print_validation_probs(val_probs, y_val[:VAL_TO_SHOW], "Ensemble", num_classes)
    else:
        model, tracker, x_train, y_train, train_center_indices, _ = train_once(
            device,
            problem,
            validation_set,
            model_config,
        )
        
        print("\n" + "=" * 60)
        print("VALIDATION PREDICTIONS")
        print("=" * 60)
        val_probs = torch.softmax(model(x_val[:VAL_TO_SHOW]), dim=1)
        print_validation_probs(val_probs, y_val[:VAL_TO_SHOW], "Model", num_classes)


if __name__ == "__main__":
    main()
