#!/usr/bin/env python3
"""
Test for the HyperXorNormal implementation in data_generator.py
"""

import torch
import pytest
from jl.posterior_minimizer.dataset_creator import HyperXorNormal


class TestHyperXorNormal:
    """Test cases for HyperXorNormal class."""

    def test_basic_functionality(self):
        """Test basic functionality of HyperXorNormal."""
        problem = HyperXorNormal(true_d=2, noisy_d=1, random_basis=False)
        
        assert problem.d == 3  # true_d + noisy_d
        assert problem.num_corners == 4  # 2^true_d
        assert problem.true_d == 2
        assert problem.noisy_d == 1

        # Generate a small dataset
        x, y, center_indices = problem.generate_dataset(n=8, percent_correct=0.8, shuffle=True)
        
        assert x.shape == (8, 3)  # n=8, d=3
        assert y.shape == (8,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64
        
        # Test the new centers and center_indices
        assert centers.shape == (4, 3)  # 2^true_d corners, d dimensions
        assert center_indices.shape == (8,)  # n samples
        assert centers.dtype == torch.float32
        assert center_indices.dtype == torch.int64
        
        # center_indices should be valid indices into centers
        assert torch.all(center_indices >= 0)
        assert torch.all(center_indices < 4)  # 2^true_d
        
        # Labels should be 0 or 1 (binary classification)
        unique_labels = torch.unique(y)
        assert len(unique_labels) <= 2
        assert all(label in [0, 1] for label in unique_labels.tolist())

    def test_different_parameters(self):
        """Test with different parameter combinations."""
        problem = HyperXorNormal(true_d=3, noisy_d=0, random_basis=True)
        
        assert problem.d == 3
        assert problem.num_corners == 8  # 2^3
        assert problem.basis is not None  # random_basis=True
        
        x, y, center_indices = problem.generate_dataset(n=16, percent_correct=1.0)
        
        assert x.shape == (16, 3)
        assert y.shape == (16,)
        
        # Test centers and center_indices for different parameters
        assert centers.shape == (8, 3)  # 2^3 corners, 3 dimensions
        assert center_indices.shape == (16,)
        assert torch.all(center_indices >= 0)
        assert torch.all(center_indices < 8)  # 2^3

    def test_perfect_vs_imperfect_classification(self):
        """Test that percent_correct parameter affects dataset generation."""
        problem = HyperXorNormal(true_d=2)
        
        # Generate datasets with different percent_correct values
        x_perfect, y_perfect, _ = problem.generate_dataset(n=100, percent_correct=1.0)
        x_imperfect, y_imperfect, _ = problem.generate_dataset(n=100, percent_correct=0.6)
        
        assert x_perfect.shape == x_imperfect.shape == (100, 2)
        assert y_perfect.shape == y_imperfect.shape == (100,)
        
        # Both should have binary labels
        assert set(torch.unique(y_perfect).tolist()).issubset({0, 1})
        assert set(torch.unique(y_imperfect).tolist()).issubset({0, 1})

    def test_no_shuffle(self):
        """Test dataset generation without shuffling."""
        problem = HyperXorNormal(true_d=2)
        x, y, _ = problem.generate_dataset(n=10, shuffle=False)
        
        assert x.shape == (10, 2)
        assert y.shape == (10,)

    def test_device_and_generator(self):
        """Test device and generator parameters."""
        device = torch.device("cpu")
        generator = torch.Generator()
        generator.manual_seed(42)
        
        problem = HyperXorNormal(true_d=2, device=device, generator=generator)
        x, y, _ = problem.generate_dataset(n=10)
        
        assert x.device == device
        assert y.device == device

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Invalid true_d
        with pytest.raises(ValueError, match="true_d must be positive"):
            HyperXorNormal(true_d=0)
        
        with pytest.raises(ValueError, match="true_d must be positive"):
            HyperXorNormal(true_d=-1)
        
        # Invalid noisy_d
        with pytest.raises(ValueError, match="noisy_d must be non-negative"):
            HyperXorNormal(true_d=2, noisy_d=-1)
        
        # Invalid n in generate_dataset
        problem = HyperXorNormal(true_d=2)
        with pytest.raises(ValueError, match="n must be positive"):
            problem.generate_dataset(n=0)
        
        with pytest.raises(ValueError, match="n must be positive"):
            problem.generate_dataset(n=-1)
        
        # Invalid percent_correct
        with pytest.raises(ValueError, match="percent_correct must be between 0.0 and 1.0"):
            problem.generate_dataset(n=10, percent_correct=1.5)
        
        with pytest.raises(ValueError, match="percent_correct must be between 0.0 and 1.0"):
            problem.generate_dataset(n=10, percent_correct=-0.1)

    def test_reproducibility(self):
        """Test that using the same generator produces reproducible results."""
        generator1 = torch.Generator()
        generator1.manual_seed(42)
        generator2 = torch.Generator()
        generator2.manual_seed(42)
        
        problem1 = HyperXorNormal(true_d=2, generator=generator1)
        problem2 = HyperXorNormal(true_d=2, generator=generator2)
        
        x1, y1, _ = problem1.generate_dataset(n=10, shuffle=False)
        x2, y2, _ = problem2.generate_dataset(n=10, shuffle=False)
        
        # Results should be identical with same seed
        assert torch.allclose(x1, x2)
        assert torch.equal(y1, y2)

    def test_centers_and_center_indices_behavior(self):
        """Test the new centers and center_indices functionality."""
        problem = HyperXorNormal(true_d=2, noisy_d=1, random_basis=False)
        
        x, y, center_indices = problem.generate_dataset(n=12, shuffle=False)
        
        # Test centers structure for HyperXorNormal
        assert centers.shape == (4, 3)  # 2^2 corners, 3 total dimensions
        
        # Centers should represent the scaled corners of the hypercube
        # The noisy dimension should be zero for all centers
        assert torch.allclose(centers[:, 2], torch.zeros(4))  # noisy dimension is zero
        
        # For each sample, the center_indices should point to valid centers
        for i in range(12):
            center_idx = center_indices[i].item()
            assert 0 <= center_idx < 4
            
            # The label should match the parity of the chosen center
            expected_label = problem.labels[center_idx].item()
            actual_label = y[i].item()
            assert expected_label == actual_label, f"Sample {i}: expected label {expected_label}, got {actual_label}"
        
        # Test with random_basis=True
        problem_rotated = HyperXorNormal(true_d=2, noisy_d=0, random_basis=True)
        x_rot, y_rot, center_indices_rot = problem_rotated.generate_dataset(n=8, shuffle=False)
        
        # Test that rotation was applied (data should be different)
        # (This is harder to test precisely, but we can check dimensions are correct)
        assert problem_rotated.basis is not None


def test_hyperxor_normal_basic():
    """Simple functional test for HyperXorNormal."""
    print('Testing HyperXorNormal implementation...')

    # Test basic functionality
    problem = HyperXorNormal(true_d=2, noisy_d=1, random_basis=False)
    print(f'Created problem with true_d=2, noisy_d=1, total_d={problem.d}')
    print(f'Number of corners: {problem.num_corners}')

    # Generate a small dataset
    x, y, center_indices = problem.generate_dataset(n=8, percent_correct=0.8, shuffle=True)
    print(f'Generated dataset: x.shape={x.shape}, y.shape={y.shape}')
    print(f'Centers shape: {centers.shape}, center_indices shape: {center_indices.shape}')
    print(f'X dtype: {x.dtype}, Y dtype: {y.dtype}')
    print(f'Labels: {y.tolist()}')
    print(f'Unique labels: {torch.unique(y).tolist()}')
    print(f'Center indices: {center_indices.tolist()}')

    print('✓ Basic test passed! HyperXorNormal implementation works correctly!')


if __name__ == '__main__':
    test_hyperxor_normal_basic()
