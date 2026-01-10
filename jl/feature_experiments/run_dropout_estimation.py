import torch
import matplotlib.pyplot as plt

from jl.feature_experiments.feature_problem import SingleFeatures
from jl.config import Config
from jl.single_runner import train_once


NUM_RUNS = 50


def main():
    # torch.manual_seed(38175)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Problem: SingleFeatures with f=4 and varying percent_correct_per_f
    percent_correct_per_f = [0.6, 0.7, 0.8, 0.9]
    clean_mode = False
    problem = SingleFeatures(
        true_d=4,
        f=4,
        device=device,
        is_orthogonal=False,
        percent_correct_per_f=percent_correct_per_f,
        noisy_d=8,
    )
    n = 128

    # Model configuration
    model_config = Config(
        model_type='resnet',
        d=problem.d,
        n_val=n,
        n=n,
        batch_size=n // 4,
        lr=0.03,
        epochs=100,
        weight_decay=0.1,
        num_layers=3,
        num_class=problem.num_classes(),
        h=20,
        weight_tracker="accuracy",
        width_varyer=None,
        is_norm=True,
        optimizer="adam_w",
        learnable_norm_parameters=True,
        lr_scheduler=None,
        dropout_prob=0.5,
        is_hashed_dropout=False,
    )

    # Generate validation set with class-balanced sampling (clean_mode=True)
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

    # Put model in train mode to enable stochastic dropout
    model.train()
    
    n_val = x_val.shape[0]
    num_classes = model_config.num_class
    
    # Stack validation set NUM_RUNS times: (n_val * NUM_RUNS, d)
    x_val_stacked = x_val.repeat(NUM_RUNS, 1)
    
    with torch.no_grad():
        # Single forward pass with dropout active
        logits_stacked = model(x_val_stacked)  # (n_val * NUM_RUNS, num_classes)
        
        # Reshape to (NUM_RUNS, n_val, num_classes)
        logits_reshaped = logits_stacked.view(NUM_RUNS, n_val, num_classes)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits_reshaped, dim=2)  # (NUM_RUNS, n_val, num_classes)
        
        # Select probability of correct class for each sample
        # y_val is (n_val,), expand to (NUM_RUNS, n_val) for gathering
        y_val_expanded = y_val.unsqueeze(0).expand(NUM_RUNS, -1)  # (NUM_RUNS, n_val)
        
        # Gather correct class probabilities: (NUM_RUNS, n_val)
        correct_probs = probs.gather(2, y_val_expanded.unsqueeze(2)).squeeze(2)
        
        # Compute std across NUM_RUNS for each validation point: (n_val,)
        std_per_point = correct_probs.std(dim=0)
        
        # Compute average std for each center
        avg_std_per_center = []
        print("\nDropout Probability Standard Deviation by Center:")
        print("-" * 60)
        for center_idx in range(problem.f):
            center_mask = (center_indices == center_idx)
            if center_mask.sum() > 0:
                avg_std = std_per_point[center_mask].mean().item()
                avg_std_per_center.append(avg_std)
                print(f"Center {center_idx} (percent_correct={percent_correct_per_f[center_idx]:.1f}): "
                      f"avg_std={avg_std:.6f} (n={center_mask.sum().item()})")
            else:
                avg_std_per_center.append(0.0)
                print(f"Center {center_idx}: No samples")
        print("-" * 60)
    
    # Plot percent_correct_per_f vs average std
    plt.figure()
    plt.plot(percent_correct_per_f, avg_std_per_center, 'o-', color='steelblue', markersize=8)
    plt.xlabel("Percent Correct per Feature")
    plt.ylabel("Average Std of Correct Class Probability")
    plt.title(f"Dropout Variance vs Label Noise (NUM_RUNS={NUM_RUNS})")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()

