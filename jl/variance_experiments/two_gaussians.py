import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from jl.variance_experiments.data_generator import TwoGaussians
from jl.config import Config
from jl.single_runner import train_once

def main(with_validation=False):
    torch.manual_seed(7752)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    percent_correct = 0.8
    problem = TwoGaussians(
        true_d=20,
        noisy_d=0,
        percent_correct=percent_correct,
        device=device
    )

    # Model configuration
    model_config = Config(
        model_type='mlp',
        d=problem.d,
        n_val=1000,
        n=128,
        batch_size=16,
        lr= 1e-3,
        epochs=300,
        weight_decay=0.001,
        num_layers=1,
        h=None,
        is_weight_tracker=False,
        d_model=40,
        down_rank_dim=None,
        is_norm=False,
        c=0.05
    )

    # Generate validation set without label noise (clean_mode=True)
    x_val, y_val, center_indices = problem.generate_dataset(
        model_config.n_val, 
        clean_mode=False,
        shuffle=True
    )
    validation_set = x_val.to(device), y_val.to(device), center_indices.to(device)

    # Train the model using single_runner (with label noise in training)
    model, tracker, x_train, y_train, train_center_indices = train_once(
        device, problem, validation_set, model_config, clean_mode=False
    )

    # Create EffectiveWeight module
    print("\nCreating EffectiveWeight module...")
    effective_weight_module = EffectiveWeight(model, model_config).to(device)
    
    # Select dataset based on with_validation flag
    if with_validation:
        x_analysis = x_val
        y_analysis = y_val
        dataset_name = "validation"
    else:
        x_analysis = x_train
        y_analysis = y_train
        dataset_name = "training"
    
    # Calculate effective weight per data point
    print(f"Calculating effective weight per data point on {dataset_name} set...")
    model.eval()
    effective_weight_module.eval()
    
    with torch.no_grad():
        # Initialize weight_so_far as identity matrix per data point
        # Shape: [batch_size, d, d]
        batch_size = x_analysis.shape[0]
        d = model_config.d
        weight_so_far = torch.eye(d, device=device).unsqueeze(0).expand(batch_size, d, d).clone()
        
        # Feed through EffectiveWeight module
        logits, weight_per_data_point = effective_weight_module(x_analysis, weight_so_far)
    
    # weight_per_data_point shape: [batch_size, d, 1]
    # Squeeze to get [batch_size, d]
    weight_vectors = weight_per_data_point.squeeze(2)  # Shape: [batch_size, d]
    
    # Calculate weight norm per data point
    weight_norms = torch.norm(weight_vectors, dim=1)  # Shape: [batch_size]
    
    # Calculate cosine similarity with center direction μ
    mu = problem.mu  # Shape: (true_d,)
    # Expand mu to batch
    mu_expanded = mu.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, d]
    
    # Cosine similarity per data point
    cosine_sims = F.cosine_similarity(weight_vectors, mu_expanded, dim=1)  # Shape: [batch_size]
    
    # Print results
    print("\n" + "="*80)
    print(f"EFFECTIVE WEIGHT ANALYSIS ({dataset_name.upper()} SET)")
    print("="*80)
    print(f"  Mean weight norm: {weight_norms.mean().item():.6f}")
    print(f"  Std weight norm: {weight_norms.std().item():.6f}")
    print(f"  Mean cosine similarity (with μ): {cosine_sims.mean().item():.6f}")
    print(f"  Std cosine similarity: {cosine_sims.std().item():.6f}")
    
    # Calculate confidence metrics (borrowed from sub_direction_per_center)
    probs = torch.sigmoid(logits)
    confidences = torch.max(probs, 1.0 - probs)  # Confidence in prediction direction
    mean_confidence = confidences.mean().item()
    mean_abs_logits = torch.abs(logits).mean().item()
    
    print(f"\n  Mean confidence: {mean_confidence:.6f}")
    print(f"  Mean |logits|: {mean_abs_logits:.6f}")
    
    # Plot distribution of logits
    plt.figure(figsize=(10, 6))
    logits_cpu = logits.cpu().numpy()
    plt.hist(logits_cpu, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Logits', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Logits ({dataset_name.capitalize()} Set)', fontsize=14)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Decision boundary (logit=0)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('logits_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved logits distribution plot to: logits_distribution.png")
    plt.show()
    
    # # Print per-sample details for first few samples
    # print("\nFirst 10 samples:")
    # for i in range(min(10, batch_size)):
    #     print(f"  Sample {i}: label={y_analysis[i].item()}, "
    #           f"weight_norm={weight_norms[i].item():.6f}, "
    #           f"cos_sim={cosine_sims[i].item():.6f}, "
    #           f"logit={logits[i].item():.6f}")


class EffectiveWeightBlock(nn.Module):
    """
    A block that transforms the effective weight matrix through a layer.
    """
    def __init__(self, linear_layer):
        """
        Args:
            linear_layer: The trained linear layer from the MLP
        """
        super(EffectiveWeightBlock, self).__init__()
        # Store the weight matrix: [d_out, d_in]
        self.register_buffer('weight', linear_layer.weight.data)
    
    def forward(self, x, weight_so_far):
        """
        Forward pass that computes both activations and transforms the weight matrix.
        
        Args:
            x: Input tensor of shape (batch_size, d_in)
            weight_so_far: Effective weight matrix of shape (batch_size, d, d_in)
            
        Returns:
            Tuple of (a_relu, weight_so_far_updated) where:
                - a_relu: Activations after ReLU of shape (batch_size, d_out)
                - weight_so_far_updated: Updated weight matrix of shape (batch_size, d, d_out)
        """
        # Compute linear transformation and apply ReLU
        a = F.linear(x, self.weight)  # Shape: (batch_size, d_out)
        a_relu = F.relu(a)  # Shape: (batch_size, d_out)
        
        # Create gate: 1.0 where a_relu > 0, 0.0 otherwise
        gate = (a_relu > 0).float()  # Shape: (batch_size, d_out)
        
        # Transform weight_so_far through this layer
        # F.linear(weight_so_far, W) computes weight_so_far @ W.t()
        weight_transformed = F.linear(weight_so_far, self.weight)  # Shape: (batch_size, d, d_out)
        
        # Apply gate elementwise (need to unsqueeze for broadcasting)
        gate_expanded = gate.unsqueeze(1)  # Shape: (batch_size, 1, d_out)
        weight_so_far_updated = weight_transformed * gate_expanded
        
        return a_relu, weight_so_far_updated


class EffectiveWeight(nn.Module):
    """
    A module that tracks the effective weight transformation through an MLP.
    Only supports MLPs where is_norm=False and down_rank_dim=None.
    """
    def __init__(self, trained_mlp, c: Config):
        """
        Args:
            trained_mlp: The trained MLP model
            c: Configuration object
        """
        super(EffectiveWeight, self).__init__()
        
        if c.is_norm:
            raise ValueError("EffectiveWeight does not support MLPs with is_norm=True")
        
        if c.model_type != 'mlp':
            raise ValueError("EffectiveWeight only supports MLP models")
        
        if c.down_rank_dim is not None:
            raise ValueError("EffectiveWeight does not support down_rank_dim")
        
        self.input_dim = c.d
        self.d_model = c.d_model if c.d_model is not None else c.d
        self.num_layers = c.num_layers
        
        # Extract weight matrices from trained MLP
        self.blocks = nn.ModuleList()
        
        # Extract linear layers from the Sequential module
        if self.num_layers > 0:
            for module in trained_mlp.layers:
                if isinstance(module, nn.Linear):
                    self.blocks.append(EffectiveWeightBlock(module))
        
        # Store the final layer's weight
        self.register_buffer('final_weight', trained_mlp.final_layer.weight.data)
    
    def forward(self, x, weight_so_far):
        """
        Forward pass that computes effective weight transformation and logits.
        
        Args:
            x: Input tensor of shape (batch_size, d)
            weight_so_far: Initial weight matrix of shape (batch_size, d, d) (identity per sample)
            
        Returns:
            Tuple of (logits, weight_per_data_point) where:
                - logits: Tensor of shape (batch_size,) containing the logits from the model
                - weight_per_data_point: Tensor of shape (batch_size, d, 1) containing final weight per data point
        """
        current = x
        
        # Propagate through all blocks
        for block in self.blocks:
            current, weight_so_far = block(current, weight_so_far)
        
        # Final layer: multiply by final weight (no ReLU)
        # weight_so_far: [batch_size, d, d_model] (or [batch_size, d, d] if num_layers=0)
        # final_weight: [1, d_model] (or [1, d] if num_layers=0)
        # F.linear computes: weight_so_far @ final_weight.t()
        weight_per_data_point = F.linear(weight_so_far, self.final_weight)  # Shape: (batch_size, d, 1)
        
        # Compute logits using the actual activations
        logits = F.linear(current, self.final_weight).squeeze(1)  # Shape: (batch_size,)
        
        return logits, weight_per_data_point


if __name__ == "__main__":
    main()