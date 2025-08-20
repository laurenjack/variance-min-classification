import torch
import pytest
from src.learned_dropout.config import EmpiricalConfig
from src.learned_dropout.empirical_variance import run_experiment_parallel, run_experiment_h_mask
from src.learned_dropout.models_standard import ResNetStandard
from src.learned_dropout.data_generator import Gaussian


def build_single_resnet(c: EmpiricalConfig, h: int, device: torch.device) -> torch.nn.Module:
    """Build a simple ResNet with a single residual block for testing."""
    h_list = [h]
    return ResNetStandard(c=c, h_list=h_list).to(device)


def test_parallel_vs_h_mask_equivalence():
    """Test that run_experiment_parallel and run_experiment_h_mask produce equivalent results."""
    
    # Set deterministic behavior
    torch.manual_seed(42)
    device = torch.device('cpu')  # Use CPU for deterministic results
    
    # Create minimal test configuration
    c = EmpiricalConfig(
        d=4,  # Small input dimension
        n=32,  # Small dataset size
        n_val=16,  # Small validation size
        epochs=3,  # Few epochs for speed
        num_runs=2,  # Few runs for speed
        h_range=[2, 4],  # Small hidden sizes
        batch_size=8,  # Small batch size
        lr=0.01,
        weight_decay=0.0,
        d_model=None,
        layer_norm="layer_norm",
        l1_final=None
    )
    
    # Create a simple problem
    problem = Gaussian(
        d=c.d,
        perfect_class_balance=True,
        device=device
    )
    
    # Generate validation set with fixed seed
    torch.manual_seed(123)
    x_val, y_val, _, _ = problem.generate_dataset(16, shuffle=True)
    validation_set = (x_val.to(device), y_val.to(device))
    
    # Run parallel experiment
    torch.manual_seed(42)  # Reset seed before each experiment
    parallel_vars, parallel_train_losses, parallel_val_accs, parallel_val_losses = run_experiment_parallel(
        build_single_resnet, device, validation_set, problem, c
    )
    
    # Run h_mask experiment with same seed
    torch.manual_seed(42)  # Reset seed before each experiment
    h_mask_vars, h_mask_train_losses, h_mask_val_accs, h_mask_val_losses = run_experiment_h_mask(
        build_single_resnet, device, validation_set, problem, c
    )
    
    # Compare results with reasonable tolerances
    tolerance = 1e-4  # Allow for small floating point differences
    
    # Convert to tensors for easier comparison
    parallel_vars_t = torch.tensor(parallel_vars)
    h_mask_vars_t = torch.tensor(h_mask_vars)
    parallel_train_losses_t = torch.tensor(parallel_train_losses)
    h_mask_train_losses_t = torch.tensor(h_mask_train_losses)
    parallel_val_accs_t = torch.tensor(parallel_val_accs)
    h_mask_val_accs_t = torch.tensor(h_mask_val_accs)
    parallel_val_losses_t = torch.tensor(parallel_val_losses)
    h_mask_val_losses_t = torch.tensor(h_mask_val_losses)
    
    # Check variance results
    print(f"Parallel variances: {parallel_vars}")
    print(f"H-mask variances: {h_mask_vars}")
    print(f"Variance difference: {torch.abs(parallel_vars_t - h_mask_vars_t).max().item()}")
    
    # Check training losses
    print(f"Parallel train losses: {parallel_train_losses}")
    print(f"H-mask train losses: {h_mask_train_losses}")
    print(f"Train loss difference: {torch.abs(parallel_train_losses_t - h_mask_train_losses_t).max().item()}")
    
    # Check validation accuracies
    print(f"Parallel val accs: {parallel_val_accs}")
    print(f"H-mask val accs: {h_mask_val_accs}")
    print(f"Val acc difference: {torch.abs(parallel_val_accs_t - h_mask_val_accs_t).max().item()}")
    
    # Check validation losses
    print(f"Parallel val losses: {parallel_val_losses}")
    print(f"H-mask val losses: {h_mask_val_losses}")
    print(f"Val loss difference: {torch.abs(parallel_val_losses_t - h_mask_val_losses_t).max().item()}")
    
    # Assert equivalence within tolerance
    assert torch.allclose(parallel_vars_t, h_mask_vars_t, atol=tolerance, rtol=tolerance), \
        f"Variances differ: parallel={parallel_vars}, h_mask={h_mask_vars}"
    
    assert torch.allclose(parallel_train_losses_t, h_mask_train_losses_t, atol=tolerance, rtol=tolerance), \
        f"Training losses differ: parallel={parallel_train_losses}, h_mask={h_mask_train_losses}"
    
    assert torch.allclose(parallel_val_accs_t, h_mask_val_accs_t, atol=tolerance, rtol=tolerance), \
        f"Validation accuracies differ: parallel={parallel_val_accs}, h_mask={h_mask_val_accs}"
    
    assert torch.allclose(parallel_val_losses_t, h_mask_val_losses_t, atol=tolerance, rtol=tolerance), \
        f"Validation losses differ: parallel={parallel_val_losses}, h_mask={h_mask_val_losses}"
    
    print("✓ All metrics match within tolerance!")


def test_parallel_vs_h_mask_single_step():
    """Test equivalence after just a single training step for faster debugging."""
    
    # Set deterministic behavior
    torch.manual_seed(42)
    device = torch.device('cpu')
    
    # Create even more minimal configuration
    c = EmpiricalConfig(
        d=3,
        n=8,  # Very small dataset
        n_val=8,  # Small validation size
        epochs=1,  # Single epoch
        num_runs=1,  # Single run
        h_range=[2],  # Single hidden size
        batch_size=4,
        lr=0.01,
        weight_decay=0.0,
        d_model=None,
        layer_norm="layer_norm",
        l1_final=None
    )
    
    # Create problem
    problem = Gaussian(
        d=c.d,
        perfect_class_balance=True,
        device=device
    )
    
    # Generate validation set
    torch.manual_seed(123)
    x_val, y_val, _, _ = problem.generate_dataset(8, shuffle=True)
    validation_set = (x_val.to(device), y_val.to(device))
    
    # Run both experiments
    torch.manual_seed(42)
    parallel_results = run_experiment_parallel(
        build_single_resnet, device, validation_set, problem, c
    )
    
    torch.manual_seed(42)
    h_mask_results = run_experiment_h_mask(
        build_single_resnet, device, validation_set, problem, c
    )
    
    # Compare all four metrics
    for i, (name, parallel_vals, h_mask_vals) in enumerate([
        ("variance", parallel_results[0], h_mask_results[0]),
        ("train_loss", parallel_results[1], h_mask_results[1]),
        ("val_acc", parallel_results[2], h_mask_results[2]),
        ("val_loss", parallel_results[3], h_mask_results[3])
    ]):
        parallel_t = torch.tensor(parallel_vals)
        h_mask_t = torch.tensor(h_mask_vals)
        diff = torch.abs(parallel_t - h_mask_t).max().item()
        print(f"{name}: parallel={parallel_vals}, h_mask={h_mask_vals}, diff={diff}")
        
        assert torch.allclose(parallel_t, h_mask_t, atol=1e-4, rtol=1e-4), \
            f"{name} differs: parallel={parallel_vals}, h_mask={h_mask_vals}"
    
    print("✓ Single-step test passed!")


if __name__ == "__main__":
    # Run the tests directly
    test_parallel_vs_h_mask_single_step()
    test_parallel_vs_h_mask_equivalence()
    print("All tests passed!")
