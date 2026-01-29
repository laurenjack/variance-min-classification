import torch
import torch.nn as nn
import torch.nn.functional as F

from jl.variance_experiments.data_generator import SubDirections
from jl.config import Config
from jl.single_runner import train_once


class TotalWeightBlock(nn.Module):
    """
    A block that tracks the total absolute weight sum through the network.
    """
    def __init__(self, linear_layer):
        """
        Args:
            linear_layer: The trained linear layer from the MLP
        """
        super(TotalWeightBlock, self).__init__()
        # Store the weight matrix
        self.register_buffer('weight', linear_layer.weight.data)
    
    def forward(self, x, total_weight_so_far):
        """
        Forward pass that computes both activations and accumulated weight sum.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            total_weight_so_far: Accumulated weight sum of shape (batch_size,)
            
        Returns:
            Tuple of (total_weight_so_far, a) where:
                - total_weight_so_far: Updated total weight sum of shape (batch_size,)
                - a: Activations after ReLU of shape (batch_size, output_dim)
        """
        # Input gate: which inputs are non-zero
        input_gate = (x != 0).float()  # Shape: (batch_size, input_dim)
        
        # Compute linear transformation
        a = F.linear(x, self.weight)
        
        # Apply ReLU to get activations
        a_relu = F.relu(a)
        
        # Output gate: which outputs are active after ReLU
        output_gate = (a_relu > 0).float()  # Shape: (batch_size, output_dim)
        
        # For each weight w_ij (connecting input i to output j):
        # Include |w_ij| only if both input_gate[i] AND output_gate[j] are non-zero
        # Compute: (input_gate @ |W|.T) * output_gate
        # Shape: (batch_size, output_dim)
        abs_weight = torch.abs(self.weight)  # Shape: (output_dim, input_dim)
        gated_weights = (input_gate @ abs_weight.t()) * output_gate  # Shape: (batch_size, output_dim)
        
        # Sum over output dimension to get total weight per sample
        weight_contribution = gated_weights.sum(dim=1)  # Shape: (batch_size,)
        
        # Accumulate to running total
        total_weight_so_far = total_weight_so_far + weight_contribution
        
        return total_weight_so_far, a_relu


class TotalWeight(nn.Module):
    """
    An MLP-like module that tracks total absolute weight sum through the network.
    Sums the absolute weights of active neurons at each layer.
    Only supports MLPs where is_norm=False.
    """
    def __init__(self, trained_mlp, c: Config):
        """
        Args:
            trained_mlp: The trained MLP model
            c: Configuration object
        """
        super(TotalWeight, self).__init__()
        
        if c.is_norm:
            raise ValueError("TotalWeight does not support MLPs with is_norm=True")
        
        if c.model_type != 'mlp':
            raise ValueError("TotalWeight only supports MLP models")
        
        
        self.input_dim = c.d
        self.d_model = c.d_model if c.d_model is not None else c.d
        self.num_layers = c.num_layers
        
        # Extract weight matrices from trained MLP
        # The MLP structure is: layers (Sequential) + final_layer
        # layers contains: Linear, ReLU, Linear, ReLU, ...
        # We need to extract the Linear layers from the Sequential
        
        self.blocks = nn.ModuleList()
        
        # Extract linear layers from the Sequential module
        layer_idx = 0
        for module in trained_mlp.layers:
            if isinstance(module, nn.Linear):
                self.blocks.append(TotalWeightBlock(module))
                layer_idx += 1
        
        # Store the final layer's weight
        self.register_buffer('final_weight', trained_mlp.final_layer.weight.data)
    
    def forward(self, x, total_weight_so_far):
        """
        Forward pass that computes total absolute weight sum and logits.
        
        Args:
            x: Input tensor of shape (batch_size, d)
            total_weight_so_far: Initial weight sum tensor of shape (batch_size,)
            
        Returns:
            Tuple of (total_weight_sum, logits) where:
                - total_weight_sum: Tensor of shape (batch_size,) containing the total accumulated weight sum
                - logits: Tensor of shape (batch_size,) containing the logits from the model
        """
        current = x
        
        # Propagate through all blocks
        for block in self.blocks:
            total_weight_so_far, current = block(current, total_weight_so_far)
        
        # Final layer: gate weights based on non-zero inputs
        # Input gate: which inputs to final layer are non-zero
        input_gate = (current != 0).float()  # Shape: (batch_size, d_model)
        
        # For final layer, include weights only if input is non-zero
        # No output gate since there's no ReLU after final layer
        abs_final_weight = torch.abs(self.final_weight)  # Shape: (1, d_model)
        final_weight_contribution = (input_gate @ abs_final_weight.t()).squeeze(1)  # Shape: (batch_size,)
        
        total_weight_sum = total_weight_so_far + final_weight_contribution
        
        # Compute logits using the actual weights (not absolute)
        logits = F.linear(current, self.final_weight).squeeze(1)
        
        return total_weight_sum, logits


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(4454)
    
    # Problem configuration
    true_d = 16
    sub_d = 4
    centers = 8
    num_class = 2
    sigma = 0.02
    noisy_d = 0
    random_basis = True
    
    # Fixed per-center percent_correct values
    # percent_correct_per_center = torch.tensor([0.9] * 8)
    percent_correct_per_center = torch.tensor([0.55, 0.95, 0.55, 0.95, 0.55, 0.95, 0.55, 0.95], dtype=torch.float32, device=device)
    
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
    
    # Print class and percent_correct for each center
    print("Center information:")
    for center_idx in range(centers):
        center_class = int(problem.center_to_class[center_idx].item())
        center_percent = float(percent_correct_per_center[center_idx].item())
        print(f"  Center {center_idx}: class={center_class}, percent_correct={center_percent:.2f}")
    print()
    
    # Model configuration
    model_config = Config(
        model_type='mlp',
        d=problem.d,
        n_val=1000,
        n=128,
        batch_size=16,
        lr=1e-3,
        epochs=1000,
        weight_decay=0.001,
        num_layers=1,
        num_class=problem.num_classes(),
        d_model=20,
        weight_tracker=None,
        is_norm=False
    )
    
    # Generate validation set
    x_val, y_val, val_center_indices = problem.generate_dataset(
        model_config.n_val,
        clean_mode=True,
        shuffle=True,
    )
    
    # Train the model
    validation_set = x_val.to(device), y_val.to(device), val_center_indices.to(device)
    model, _, x_train, y_train, train_center_indices, _ = train_once(
        device, problem, validation_set, model_config, clean_mode=False
    )
    
    # Create TotalWeight module
    print("\nCreating TotalWeight module...")
    total_weight_module = TotalWeight(model, model_config).to(device)
    
    # Calculate total_weight_sum for all training samples
    print("Calculating total absolute weight sum per sample...")
    model.eval()
    total_weight_module.eval()
    
    with torch.no_grad():
        # Initialize total_weight_so_far as zeros
        total_weight_so_far = torch.zeros(x_train.shape[0], device=device)
        
        # Feed through TotalWeight module
        total_weight_sum, logits = total_weight_module(x_train, total_weight_so_far)
    
    # Print individual and mean total_weight_sum per center
    print("\nTotal absolute weight sum per center:")
    for center_idx in range(centers):
        # Find all samples from this center
        center_mask = (train_center_indices == center_idx)
        center_samples_weight = total_weight_sum[center_mask]
        center_labels = y_train[center_mask]
        center_logits = logits[center_mask]
        
        if center_samples_weight.numel() > 0:
            mean_weight = float(center_samples_weight.mean().item())
            mean_abs_logit = float(torch.abs(center_logits).mean().item())
            center_class = int(problem.center_to_class[center_idx].item())
            center_percent = float(percent_correct_per_center[center_idx].item())
            num_samples = int(center_mask.sum().item())
            
            print(f"\n  Center {center_idx}: class={center_class}, percent_correct={center_percent:.2f}, n={num_samples}")
            print(f"    Individual samples:")
            for i, (weight, label, logit) in enumerate(zip(center_samples_weight.cpu().tolist(), 
                                                            center_labels.cpu().tolist(),
                                                            center_logits.cpu().tolist())):
                print(f"      Sample {i}: label={label}, logit={logit:.6f}, total_weight={weight:.6f}")
            print(f"    Mean total_weight={mean_weight:.6f}")
            print(f"    Mean absolute logit={mean_abs_logit:.6f}")
        else:
            print(f"\n  Center {center_idx}: No samples in training set")


if __name__ == "__main__":
    main()

