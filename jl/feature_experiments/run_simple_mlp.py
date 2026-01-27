import torch

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.feature_experiments.report import print_validation_probs, print_grouped_by_percent_correct
from jl.config import Config
from jl.single_runner import train_once


def main():

    torch.manual_seed(38175)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SingleFeatures
    clean_mode = False
    problem = SingleFeatures(
        true_d=4,
        f=4,
        device=device,
        is_orthogonal=False,
        percent_correct_per_f=[0.8] * 4,
        noisy_d=8,
    )
    n = 40

    # Model configuration
    model_config = Config(
        model_type='simple-mlp',
        d=problem.d,
        n_val=128,
        n=n,
        batch_size=n // 2,
        lr=0.003,
        epochs=200,
        weight_decay=0.001,
        num_layers=1,
        num_class=problem.num_classes(),
        h=80,
        weight_tracker="accuracy",
        optimizer="adam_w",
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

    # Report total activation
    model.eval()
    model.use_total_activation = True
    with torch.no_grad():
        logits = model(x_train)

    # Correctly labeled = y matches true feature (center_indices)
    # Mislabeled = y differs from true feature (due to label noise)
    correctly_labeled_mask = y_train == train_center_indices

    # Get logit for the labeled class only
    labeled_logits = logits.gather(1, y_train.unsqueeze(1)).squeeze(1)

    correct_activations = model.total_activation[correctly_labeled_mask].tolist()
    correct_logits = labeled_logits[correctly_labeled_mask].tolist()
    mislabeled_activations = model.total_activation[~correctly_labeled_mask].tolist()
    mislabeled_logits = labeled_logits[~correctly_labeled_mask].tolist()

    print("\nTotal Activation Report:")
    print(f"Correctly labeled ({len(correct_activations)}):")
    for act, lg in zip(correct_activations, correct_logits):
        print(f"  activation={act:.2f}, logit={lg:.2f}")
    print(f"Mislabeled ({len(mislabeled_activations)}):")
    for act, lg in zip(mislabeled_activations, mislabeled_logits):
        print(f"  activation={act:.2f}, logit={lg:.2f}")


if __name__ == "__main__":
    main()
