"""Visualize the Kaleidoscope problem in 2D."""
import torch
import matplotlib.pyplot as plt

from jl.feature_experiments.feature_problem import Kaleidoscope


def visualize_kaleidoscope_2d():
    """Visualize Kaleidoscope problem with d=2, centers=[2, 2, 2]."""
    # Set up problem
    d = 2
    centers = [2, 2]
    n = 160
    
    # Create problem with a fixed seed for reproducibility
    device = torch.device("cpu")
    # generator = torch.Generator(device=device).manual_seed(42)
    
    problem = Kaleidoscope(
        d=d,
        centers=centers,
        device=device,
        is_standard_basis=True,
        # generator=generator
    )
    
    # Generate dataset
    x, y, _ = problem.generate_dataset(n, shuffle=False)
    
    # Convert to numpy for plotting
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot points colored by class
    colors = ['red', 'blue']
    for class_idx in range(problem.num_classes()):
        mask = y_np == class_idx
        ax.scatter(
            x_np[mask, 0],
            x_np[mask, 1],
            c=colors[class_idx],
            s=100,
            alpha=0.6,
            label=f'Class {class_idx}',
            edgecolors='black',
            linewidth=1.5
        )
    
    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(f'Kaleidoscope Problem: d={d}, centers={centers}, n={n}', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    
    # Make plot square
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('kaleidoscope_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to kaleidoscope_visualization.png")
    plt.show()


if __name__ == "__main__":
    visualize_kaleidoscope_2d()

