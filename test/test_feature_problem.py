import unittest
import torch
from jl.feature_experiments.feature_problem import SingleFeatures


class TestSingleFeaturesNPerF(unittest.TestCase):
    
    def test_default_behavior_no_n_per_f(self):
        """Test default behavior without n_per_f specified."""
        problem = SingleFeatures(d=10, f=3)
        x, y, center_indices = problem.generate_dataset(n=9, shuffle=False)
        
        # Check shapes
        self.assertEqual(x.shape, (9, 10))
        self.assertEqual(y.shape, (9,))
        self.assertEqual(center_indices.shape, (9,))
        
        # Check balanced distribution (round-robin)
        self.assertEqual(y.tolist(), [0, 1, 2, 0, 1, 2, 0, 1, 2])
        
        # Check that center_indices matches y
        self.assertTrue(torch.equal(y, center_indices))
    
    def test_n_per_f_specified(self):
        """Test with n_per_f specified."""
        problem = SingleFeatures(d=10, f=3, n_per_f=[1, 2, 3])
        x, y, center_indices = problem.generate_dataset(n=12, shuffle=False)
        
        # Check shapes
        self.assertEqual(x.shape, (12, 10))
        self.assertEqual(y.shape, (12,))
        
        # Count label frequencies
        label_counts = {0: 0, 1: 0, 2: 0}
        for label in y.tolist():
            label_counts[label] += 1
        
        # sum(n_per_f) = 6, n = 12 = 2 * 6
        # So we should have 2 samples of class 0, 4 of class 1, 6 of class 2
        self.assertEqual(label_counts, {0: 2, 1: 4, 2: 6})
    
    def test_n_per_f_with_larger_multiplier(self):
        """Test with larger multiplier."""
        problem = SingleFeatures(d=5, f=2, n_per_f=[3, 7])
        x, y, _ = problem.generate_dataset(n=30, shuffle=False)  # sum=10, n=3*10=30
        
        label_counts = {0: 0, 1: 0}
        for label in y.tolist():
            label_counts[label] += 1
        
        # Should have 9 samples of class 0, 21 of class 1
        self.assertEqual(label_counts, {0: 9, 1: 21})
    
    def test_n_not_multiple_of_sum_raises_error(self):
        """Test that ValueError is raised when n is not a multiple of sum(n_per_f)."""
        problem = SingleFeatures(d=10, f=3, n_per_f=[1, 2, 3])
        
        # sum(n_per_f) = 6, n = 10 is not divisible by 6
        with self.assertRaises(ValueError) as context:
            problem.generate_dataset(n=10)
        
        self.assertIn("must be a multiple of sum(n_per_f)=6", str(context.exception))
    
    def test_n_per_f_wrong_length_raises_error(self):
        """Test that ValueError is raised when n_per_f has wrong length."""
        with self.assertRaises(ValueError) as context:
            SingleFeatures(d=10, f=3, n_per_f=[1, 2])  # Wrong length
        
        self.assertIn("n_per_f must have length f=3", str(context.exception))
    
    def test_n_per_f_with_zero_raises_error(self):
        """Test that ValueError is raised when n_per_f contains non-positive values."""
        with self.assertRaises(ValueError) as context:
            SingleFeatures(d=10, f=3, n_per_f=[1, 0, 3])
        
        self.assertIn("All elements of n_per_f must be positive", str(context.exception))
    
    def test_n_per_f_with_negative_raises_error(self):
        """Test that ValueError is raised when n_per_f contains negative values."""
        with self.assertRaises(ValueError) as context:
            SingleFeatures(d=10, f=3, n_per_f=[1, -2, 3])
        
        self.assertIn("All elements of n_per_f must be positive", str(context.exception))
    
    def test_shuffle_preserves_counts(self):
        """Test that shuffling preserves label counts."""
        problem = SingleFeatures(d=10, f=3, n_per_f=[1, 2, 3])
        x, y, _ = problem.generate_dataset(n=12, shuffle=True)
        
        # Count label frequencies
        label_counts = {0: 0, 1: 0, 2: 0}
        for label in y.tolist():
            label_counts[label] += 1
        
        # Counts should still be correct after shuffling
        self.assertEqual(label_counts, {0: 2, 1: 4, 2: 6})
    
    def test_feature_vectors_are_correct(self):
        """Test that generated x vectors correspond to the correct features from Q."""
        problem = SingleFeatures(d=10, f=3, n_per_f=[2, 1, 1])
        x, y, _ = problem.generate_dataset(n=8, shuffle=False)  # sum=4, n=2*4=8
        
        # Check that each x[i] equals Q[y[i]]
        for i in range(x.shape[0]):
            expected = problem.Q[y[i]]
            self.assertTrue(
                torch.allclose(x[i], expected, atol=1e-6),
                f"Sample {i} with label {y[i].item()} does not match Q[{y[i].item()}]"
            )
    
    def test_covariance_uniform_distribution(self):
        """Test covariance matrix with uniform feature distribution."""
        torch.manual_seed(42)
        problem = SingleFeatures(d=5, f=3, generator=torch.Generator().manual_seed(42))
        
        # Check that covariance matrix is symmetric
        cov = problem.covariance
        self.assertEqual(cov.shape, (5, 5))
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-6), "Covariance matrix should be symmetric")
        
        # Manually compute expected covariance
        probs = torch.ones(3) / 3.0
        mean = torch.sum(probs.unsqueeze(1) * problem.Q, dim=0)
        E_xx = torch.zeros(5, 5)
        for i in range(3):
            qi = problem.Q[i]
            E_xx += probs[i] * torch.outer(qi, qi)
        expected_cov = E_xx - torch.outer(mean, mean)
        
        self.assertTrue(torch.allclose(cov, expected_cov, atol=1e-6))
    
    def test_covariance_with_n_per_f(self):
        """Test covariance matrix with non-uniform feature distribution."""
        torch.manual_seed(42)
        problem = SingleFeatures(d=5, f=3, n_per_f=[1, 2, 3], generator=torch.Generator().manual_seed(42))
        
        # Check that covariance matrix is symmetric
        cov = problem.covariance
        self.assertEqual(cov.shape, (5, 5))
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-6), "Covariance matrix should be symmetric")
        
        # Manually compute expected covariance
        probs = torch.tensor([1/6, 2/6, 3/6])
        mean = torch.sum(probs.unsqueeze(1) * problem.Q, dim=0)
        E_xx = torch.zeros(5, 5)
        for i in range(3):
            qi = problem.Q[i]
            E_xx += probs[i] * torch.outer(qi, qi)
        expected_cov = E_xx - torch.outer(mean, mean)
        
        self.assertTrue(torch.allclose(cov, expected_cov, atol=1e-6))
    
    def test_covariance_empirical_matches_theoretical(self):
        """Test that empirical covariance from samples matches theoretical covariance."""
        torch.manual_seed(42)
        problem = SingleFeatures(d=5, f=3, n_per_f=[1, 2, 3], generator=torch.Generator().manual_seed(42))
        
        # Generate large dataset
        x, _, _ = problem.generate_dataset(n=6000, shuffle=True)  # sum=6, n=1000*6
        
        # Compute empirical covariance
        empirical_mean = x.mean(dim=0)
        centered = x - empirical_mean
        empirical_cov = (centered.T @ centered) / x.shape[0]
        
        # Compare with theoretical covariance
        theoretical_cov = problem.covariance
        
        # Should be close for large sample size
        self.assertTrue(
            torch.allclose(empirical_cov, theoretical_cov, atol=1e-2),
            f"Empirical covariance should match theoretical.\nDiff max: {(empirical_cov - theoretical_cov).abs().max()}"
        )


if __name__ == '__main__':
    unittest.main()

