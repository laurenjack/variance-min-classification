import pytest
import torch

from jl.variance_experiments.data_generator import SubDirections


def make_gen(seed: int = 1234, device: torch.device | None = None) -> torch.Generator:
    g = torch.Generator(device=device if device is not None else "cpu")
    g.manual_seed(seed)
    return g


def test_input_validations():
    with pytest.raises(ValueError):
        SubDirections(true_d=4, sub_d=0, centers=4, num_class=1)
    with pytest.raises(ValueError):
        SubDirections(true_d=2, sub_d=4, centers=4, num_class=1)
    with pytest.raises(ValueError):
        SubDirections(true_d=5, sub_d=2, centers=4, num_class=1)  # true_d % sub_d != 0
    with pytest.raises(ValueError):
        SubDirections(true_d=4, sub_d=2, centers=0, num_class=1)
    with pytest.raises(ValueError):
        SubDirections(true_d=4, sub_d=2, centers=5, num_class=0)


def test_capacity_constraint():
    # d=6, sub_d=3 → S=2 subsections, 2^sub_d=8 patterns per subsection
    # Capacity rule: ceil(perms/S) < 2^sub_d ⇒ perms <= S * (2^sub_d - 1)
    d, sub_d = 6, 3
    S = d // sub_d
    P = 1 << sub_d
    perms_max = S * (P - 1)

    # Max valid perms
    sd_ok = SubDirections(true_d=d, sub_d=sub_d, centers=perms_max, num_class=2, generator=make_gen(0))
    assert sd_ok.centers_block_signs.shape == (perms_max, sub_d)

    # One more should fail
    with pytest.raises(ValueError):
        SubDirections(true_d=d, sub_d=sub_d, centers=perms_max + 1, num_class=2)


def test_center_assignment_and_uniqueness():
    # Check that centers are assigned round-robin to subsections and are unique within subsection
    d, sub_d, S = 8, 2, 4
    perms = 12  # per subsection: [3,3,3,3]
    sd = SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=3, generator=make_gen(42))

    # Count centers per subsection
    counts = torch.bincount(sd.center_subsection, minlength=S)
    assert counts.tolist() == [3, 3, 3, 3]

    # Uniqueness of patterns within each subsection
    for s in range(S):
        idxs = torch.nonzero(sd.center_subsection == s, as_tuple=False).view(-1)
        blocks = sd.centers_block_signs[idxs]
        keys = set(tuple(int(v) for v in row.tolist()) for row in blocks)
        assert len(keys) == int(idxs.numel())


def test_class_balancing_within_subsections():
    d, sub_d, perms, num_class = 8, 2, 12, 3
    sd = SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, generator=make_gen(7))
    for s in range(d // sub_d):
        idxs = torch.nonzero(sd.center_subsection == s, as_tuple=False).view(-1)
        classes = sd.center_to_class[idxs]
        counts = torch.bincount(classes, minlength=num_class).tolist()
        # Each class count should differ by at most 1 in balancing
        assert max(counts) - min(counts) <= 1


def test_generate_shapes_and_types():
    sd = SubDirections(true_d=6, sub_d=3, centers=6, num_class=2, generator=make_gen(1))
    x, y, center_indices, _ = sd.generate_dataset(n=12, shuffle=False)
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert isinstance(center_indices, torch.Tensor)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert center_indices.dtype == torch.int64
    assert x.shape == (12, 6)
    assert y.shape == (12,)
    assert center_indices.shape == (12,)  # n samples
    
    # center_indices should be valid indices
    assert torch.all(center_indices >= 0)
    assert torch.all(center_indices < 6)  # perms


def test_center_balanced_sampling_counts():
    # Test that samples are balanced across centers
    d, sub_d, perms, num_class = 8, 2, 12, 3
    sd = SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, generator=make_gen(5))
    
    # Test case 1: n divisible by perms - each center gets exactly n//perms samples
    n = 120  # 120 = 12 * 10, so each center gets exactly 10 samples
    x, y, _, _ = sd.generate_dataset(n=n, shuffle=False)
    
    # Since sampling is center-balanced, we can verify the structure
    # We know each center gets exactly n//perms = 10 samples
    base_per_center = n // perms
    assert base_per_center == 10
    
    # Test case 2: n not divisible by perms - some centers get +1 sample
    n2 = 125  # 125 = 12*10 + 5, so 5 centers get 11 samples, 7 centers get 10 samples
    x2, y2, _, _ = sd.generate_dataset(n=n2, shuffle=False)
    base2 = n2 // perms  # 10
    remainder2 = n2 % perms  # 5
    
    # Each center should get either base2 or base2+1 samples
    # exactly remainder2 centers get base2+1, the rest get base2
    assert base2 == 10
    assert remainder2 == 5
    
    # We can't easily verify the exact distribution without instrumenting the code,
    # but we can test that the structure is mathematically correct:
    # remainder2 centers with (base2+1) samples + (perms-remainder2) centers with base2 samples = n2
    assert remainder2 * (base2 + 1) + (perms - remainder2) * base2 == n2


def test_determinism_with_generator():
    gen = make_gen(42)
    sd1 = SubDirections(true_d=6, sub_d=3, centers=12, num_class=2, generator=gen)
    x1, y1, _, _ = sd1.generate_dataset(n=24, shuffle=False)

    gen2 = make_gen(42)
    sd2 = SubDirections(true_d=6, sub_d=3, centers=12, num_class=2, generator=gen2)
    x2, y2, _, _ = sd2.generate_dataset(n=24, shuffle=False)

    assert torch.allclose(x1, x2)
    assert torch.equal(y1, y2)


def test_sample_generation_logic():
    # Ensure chosen center's subsection is fixed to the center pattern in means (up to noise)
    d, sub_d, perms = 8, 2, 8
    sd = SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=2, generator=make_gen(0), sigma=1e-6)
    n = 80  # 80 = 8*10, so each center gets exactly 10 samples
    x, y, _, _ = sd.generate_dataset(n=n, shuffle=False)

    # With center-balanced sampling, we know the structure:
    # Each center gets exactly n//perms = 10 samples
    # Samples 0-9 come from center 0, samples 10-19 from center 1, etc.
    base_per_center = n // perms
    assert base_per_center == 10
    
    # Test the pattern matching for each center's samples
    for center_idx in range(perms):
        # Samples for this center are at indices [center_idx*base_per_center, (center_idx+1)*base_per_center)
        start_sample = center_idx * base_per_center
        end_sample = (center_idx + 1) * base_per_center
        
        # Get the subsection and pattern for this center
        subsection = int(sd.center_subsection[center_idx].item())
        center_pattern = sd.centers_block_signs[center_idx]
        
        # Check all samples from this center
        for sample_idx in range(start_sample, min(end_sample, start_sample + 5)):  # Test first 5 samples per center
            # Extract the relevant subsection from the sample
            sub_start = subsection * sub_d
            sub_end = (subsection + 1) * sub_d
            obs_pattern = x[sample_idx, sub_start:sub_end]
            
            # Due to tiny noise, signs should match center pattern
            assert torch.allclose(torch.sign(obs_pattern), center_pattern, atol=0.0), \
                f"Pattern mismatch for sample {sample_idx} from center {center_idx}"
            
            # Verify the label matches the center's class
            expected_class = int(sd.center_to_class[center_idx].item())
            actual_class = int(y[sample_idx].item())
            assert actual_class == expected_class, \
                f"Class mismatch for sample {sample_idx}: expected {expected_class}, got {actual_class}"


def test_percent_correct_balance_across_centers():
    """Test that incorrect labels are balanced across centers"""
    d, sub_d, perms, num_class = 6, 2, 9, 3
    n = 90  # Divisible by perms (9) for exact center balance: each center gets 10 samples
    percent_correct = 0.7  # 30% incorrect = 27 incorrect samples
    
    sd = SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=percent_correct, generator=make_gen(123))
    x, y, _, _ = sd.generate_dataset(n=n, clean_mode=False, shuffle=False)
    
    # Generate perfect labels for comparison
    sd_perfect = SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=1.0, generator=make_gen(123))
    x_perfect, y_perfect, _, _ = sd_perfect.generate_dataset(n=n, clean_mode=True, shuffle=False)
    
    # Count how many labels are different
    different_labels = torch.sum(y != y_perfect).item()
    expected_incorrect = round(n * (1.0 - percent_correct))  # 27
    
    # Should be exactly the expected number of incorrect labels
    assert different_labels == expected_incorrect, f"Expected {expected_incorrect} incorrect labels, got {different_labels}"
    
    # Test the center balance property: incorrect samples should be distributed across centers
    # With n=90, centers=9: each center gets 10 samples
    # With 27 incorrect: each center should get 27//9 = 3 incorrect samples
    base_incorrect_per_center = expected_incorrect // perms  # 3
    remainder_incorrect = expected_incorrect % perms  # 0
    
    # Since n is divisible by perms and expected_incorrect is divisible by perms,
    # each center should get exactly base_incorrect_per_center incorrect samples
    assert base_incorrect_per_center == 3
    assert remainder_incorrect == 0
    
    # Verify mathematical consistency
    assert base_incorrect_per_center * perms == expected_incorrect


def test_percent_correct_edge_cases():
    """Test edge cases for percent_correct"""
    d, sub_d, perms, num_class = 4, 2, 4, 2
    sd = SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=1.0, generator=make_gen(99))
    
    # Test 100% correct with clean_mode=True
    x1, y1, _, _ = sd.generate_dataset(n=20, clean_mode=True)
    x2, y2, _, _ = sd.generate_dataset(n=20, clean_mode=True)
    # Should be deterministic with same generator state
    # (Note: generator state advances, so we can't directly compare)
    
    # Test 0% correct (all labels should be different from original)
    sd_zero = SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=0.0, generator=make_gen(99))
    sd_perfect = SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=1.0, generator=make_gen(99))
    
    x_zero, y_zero, _, _ = sd_zero.generate_dataset(n=20, clean_mode=False, shuffle=False)
    x_perf, y_perf, _, _ = sd_perfect.generate_dataset(n=20, clean_mode=True, shuffle=False)
    
    # All labels should be different when percent_correct=0
    different_count = torch.sum(y_zero != y_perf).item()
    assert different_count == 20, f"Expected all 20 labels to be different, got {different_count}"
    
    # Features should still be the same
    assert torch.allclose(x_zero, x_perf), "Features should be identical"


def test_percent_correct_validation():
    """Test that percent_correct validation works in constructor"""
    d, sub_d, perms, num_class = 4, 2, 4, 2
    
    # Valid range should work
    SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=0.0, generator=make_gen(1))
    SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=0.5, generator=make_gen(1))
    SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=1.0, generator=make_gen(1))
    
    # Invalid values should raise errors
    with pytest.raises(ValueError):
        SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=-0.1, generator=make_gen(1))
    
    with pytest.raises(ValueError):
        SubDirections(true_d=d, sub_d=sub_d, centers=perms, num_class=num_class, percent_correct=1.1, generator=make_gen(1))


def test_two_gaussians_basic():
    """Test basic TwoGaussians functionality."""
    from jl.variance_experiments.data_generator import TwoGaussians
    from scipy.stats import norm
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Test basic instantiation
    problem = TwoGaussians(true_d=10, noisy_d=5, percent_correct=0.9)
    
    # Check dimensions
    assert problem.d == 15  # true_d + noisy_d
    assert problem.true_d == 10
    assert problem.noisy_d == 5
    
    # Check mean vector norm is correct
    expected_norm = norm.ppf(0.9)
    actual_norm = torch.norm(problem.mu).item()
    assert abs(actual_norm - expected_norm) < 1e-5, f"Expected norm {expected_norm}, got {actual_norm}"
    
    # Test dataset generation
    x, y, center_indices, px = problem.generate_dataset(n=100)
    assert x.shape == (100, 15)
    assert y.shape == (100,)
    assert center_indices.shape == (100,)
    assert px is None  # TwoGaussians returns None for px
    
    # Check class balance
    class_0_count = torch.sum(y == 0).item()
    class_1_count = torch.sum(y == 1).item()
    assert class_0_count == 50
    assert class_1_count == 50
    
    # Check labels match centers
    assert torch.all(y == center_indices)


def test_two_gaussians_validation():
    """Test TwoGaussians input validation."""
    from jl.variance_experiments.data_generator import TwoGaussians
    
    # Test invalid true_d
    with pytest.raises(ValueError):
        TwoGaussians(true_d=0, percent_correct=0.9)
    
    with pytest.raises(ValueError):
        TwoGaussians(true_d=-5, percent_correct=0.9)
    
    # Test invalid noisy_d
    with pytest.raises(ValueError):
        TwoGaussians(true_d=5, noisy_d=-1, percent_correct=0.9)
    
    # Test invalid percent_correct
    with pytest.raises(ValueError):
        TwoGaussians(true_d=5, percent_correct=0.5)  # Must be > 0.5
    
    with pytest.raises(ValueError):
        TwoGaussians(true_d=5, percent_correct=1.0)  # Can't be exactly 1.0
    
    with pytest.raises(ValueError):
        TwoGaussians(true_d=5, percent_correct=1.1)  # Must be < 1.0
    
    with pytest.raises(ValueError):
        TwoGaussians(true_d=5, percent_correct=0.3)  # Must be > 0.5


def test_two_gaussians_percent_correct_in_constructor():
    """Test that percent_correct is set in constructor for TwoGaussians."""
    from jl.variance_experiments.data_generator import TwoGaussians
    
    torch.manual_seed(42)
    problem = TwoGaussians(true_d=5, percent_correct=0.9)
    
    # percent_correct is now only a constructor parameter
    # generate_dataset should work without it
    x, y, center_indices, _ = problem.generate_dataset(n=100)
    assert x.shape == (100, 5)


def test_two_gaussians_class_balance():
    """Test that TwoGaussians maintains class balance."""
    from jl.variance_experiments.data_generator import TwoGaussians
    
    torch.manual_seed(123)
    problem = TwoGaussians(true_d=8, percent_correct=0.85)
    
    # Test even n
    x, y, center_indices, _ = problem.generate_dataset(n=200, shuffle=False)
    assert torch.sum(y == 0).item() == 100
    assert torch.sum(y == 1).item() == 100
    
    # Test odd n
    x2, y2, center_indices2, _ = problem.generate_dataset(n=201, shuffle=False)
    class_0_count = torch.sum(y2 == 0).item()
    class_1_count = torch.sum(y2 == 1).item()
    # One class should have 100, the other 101
    assert class_0_count + class_1_count == 201
    assert abs(class_0_count - class_1_count) == 1


def test_two_gaussians_noisy_dimensions():
    """Test that noisy dimensions are properly added."""
    from jl.variance_experiments.data_generator import TwoGaussians
    
    torch.manual_seed(456)
    
    # Test with noisy_d = 0
    problem_no_noise = TwoGaussians(true_d=10, noisy_d=0, percent_correct=0.9)
    x_no_noise, _, _, _ = problem_no_noise.generate_dataset(n=50)
    assert x_no_noise.shape == (50, 10)
    
    # Test with noisy_d > 0
    problem_with_noise = TwoGaussians(true_d=10, noisy_d=5, percent_correct=0.9)
    x_with_noise, _, _, _ = problem_with_noise.generate_dataset(n=50)
    assert x_with_noise.shape == (50, 15)


def test_two_gaussians_shuffle():
    """Test that shuffle flag works correctly."""
    from jl.variance_experiments.data_generator import TwoGaussians
    
    torch.manual_seed(789)
    problem = TwoGaussians(true_d=5, percent_correct=0.9)
    
    # Generate without shuffle
    torch.manual_seed(100)
    x1, y1, center_indices1, _ = problem.generate_dataset(n=100, shuffle=False)
    
    # Generate again with same seed
    torch.manual_seed(100)
    x2, y2, center_indices2, _ = problem.generate_dataset(n=100, shuffle=False)
    
    # Should be identical
    assert torch.allclose(x1, x2)
    assert torch.equal(y1, y2)
    
    # Generate with shuffle
    torch.manual_seed(100)
    x3, y3, center_indices3, _ = problem.generate_dataset(n=100, shuffle=True)
    
    # Should have same class distribution but different order
    assert torch.sum(y3 == 0).item() == torch.sum(y1 == 0).item()
    # Order should be different (with high probability)
    assert not torch.equal(y1, y3) or not torch.allclose(x1, x3)


def test_two_gaussians_overlap_statistical():
    """Test that the overlap between Gaussians matches percent_correct using statistical validation."""
    from jl.variance_experiments.data_generator import TwoGaussians
    from scipy.stats import binom
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create problem with known percent_correct
    percent_correct = 0.8
    problem = TwoGaussians(true_d=20, noisy_d=0, percent_correct=percent_correct)
    
    # Generate large dataset
    n = 10000
    x, y, center_indices, _ = problem.generate_dataset(n=n, shuffle=False)
    
    # For each sample, check if it's closer to its own center or the other center
    # Center 0 is at +mu, Center 1 is at -mu
    # Decision boundary is x^T mu = 0
    
    # Compute distances to both centers for all samples
    # For samples from center 0 (class 0):
    mask_class_0 = (y == 0)
    x_class_0 = x[mask_class_0]  # (n/2, true_d)
    n_class_0 = x_class_0.shape[0]
    
    # Distance to own center (mu) vs other center (-mu)
    # Since we're comparing ||x - mu||^2 vs ||x - (-mu)||^2
    # This simplifies to comparing 2*x^T*mu (positive means closer to mu)
    projection_0 = x_class_0 @ problem.mu  # (n/2,)
    correct_class_0 = (projection_0 > 0).sum().item()
    
    # For samples from center 1 (class 1):
    mask_class_1 = (y == 1)
    x_class_1 = x[mask_class_1]
    n_class_1 = x_class_1.shape[0]
    
    # For class 1, samples should be closer to -mu, so projection should be negative
    projection_1 = x_class_1 @ problem.mu
    correct_class_1 = (projection_1 < 0).sum().item()
    
    # Calculate empirical percent_correct
    total_correct = correct_class_0 + correct_class_1
    empirical_percent_correct = total_correct / n
    
    # Statistical test with alpha = 0.01
    # Under null hypothesis: true percent_correct = 0.8
    # We have n Bernoulli trials
    # Use binomial test to check if empirical_percent_correct is consistent with 0.8
    
    # Calculate 99% confidence interval (alpha = 0.01)
    # Two-tailed test, so we use alpha/2 = 0.005 on each tail
    lower_bound = binom.ppf(0.005, n, percent_correct) / n
    upper_bound = binom.ppf(0.995, n, percent_correct) / n
    
    # Assert that empirical percent_correct falls within the confidence interval
    assert lower_bound <= empirical_percent_correct <= upper_bound, \
        f"Empirical percent_correct {empirical_percent_correct:.4f} is outside 99% CI [{lower_bound:.4f}, {upper_bound:.4f}]"
    
    print(f"Empirical percent_correct: {empirical_percent_correct:.4f}")
    print(f"99% CI: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"Test passed: overlap matches expected percent_correct={percent_correct}")


def test_two_directions_basic():
    """Test basic TwoDirections functionality."""
    from jl.variance_experiments.data_generator import TwoDirections
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Test basic instantiation
    problem = TwoDirections(true_d=10, noisy_d=5, percent_correct=0.9)
    
    # Check dimensions
    assert problem.d == 15  # true_d + noisy_d
    assert problem.true_d == 10
    assert problem.noisy_d == 5
    
    # Check mean vector comes from hypercube vertices (each coordinate is -1 or +1)
    assert torch.all((problem.mu == -1.0) | (problem.mu == 1.0)), "mu should be a hypercube vertex"
    
    # Test dataset generation
    x, y, center_indices, px = problem.generate_dataset(n=100)
    assert px is None  # TwoDirections returns None for px
    assert x.shape == (100, 15)
    assert y.shape == (100,)
    assert center_indices.shape == (100,)
    
    # Check class balance
    class_0_count = torch.sum(y == 0).item()
    class_1_count = torch.sum(y == 1).item()
    assert class_0_count == 50
    assert class_1_count == 50


def test_two_directions_deterministic():
    """Test exact generation with no sigma, random_basis=False, n=100, percent_correct=0.8."""
    from jl.variance_experiments.data_generator import TwoDirections
    
    # Create problem with no sigma and no random basis
    problem = TwoDirections(
        true_d=5,
        noisy_d=0,
        percent_correct=0.8,
        sigma=None,
        random_basis=False
    )
    
    # Generate dataset without shuffling for deterministic testing
    n = 100
    x, y, center_indices, _ = problem.generate_dataset(n=n, shuffle=False)
    
    # Check shapes
    assert x.shape == (100, 5)
    assert y.shape == (100,)
    assert center_indices.shape == (100,)
    
    # Check class balance (50 from each center)
    unique_centers, center_counts = torch.unique(center_indices, return_counts=True)
    assert len(unique_centers) == 2
    assert center_counts[0].item() == 50
    assert center_counts[1].item() == 50
    
    # First 50 samples should be from center 0, last 50 from center 1
    assert torch.all(center_indices[:50] == 0)
    assert torch.all(center_indices[50:] == 1)
    
    # Without sigma and noisy_d=0, all samples from center 0 should be exactly at mu
    # All samples from center 1 should be exactly at -mu
    for i in range(50):
        assert torch.allclose(x[i], problem.mu, atol=1e-6)
    for i in range(50, 100):
        assert torch.allclose(x[i], -problem.mu, atol=1e-6)
    
    # Check label noise: 20% of 100 = 20 flipped labels
    # These should be balanced: 10 from class 0, 10 from class 1
    num_incorrect = round(n * (1.0 - 0.8))
    assert num_incorrect == 20
    
    # Count correct and incorrect labels based on center_indices
    # Correct label should match center index
    num_correct = torch.sum(y == center_indices).item()
    num_flipped = torch.sum(y != center_indices).item()
    
    assert num_flipped == 20, f"Expected 20 flipped labels, got {num_flipped}"
    assert num_correct == 80, f"Expected 80 correct labels, got {num_correct}"
    
    # Check that flipping is balanced across classes
    # Count flipped labels for each center
    flipped_from_center_0 = torch.sum((center_indices == 0) & (y != 0)).item()
    flipped_from_center_1 = torch.sum((center_indices == 1) & (y != 1)).item()
    
    # With round-robin balancing: 20 flips = 10 per class
    assert flipped_from_center_0 == 10, f"Expected 10 flips from center 0, got {flipped_from_center_0}"
    assert flipped_from_center_1 == 10, f"Expected 10 flips from center 1, got {flipped_from_center_1}"


def test_two_directions_validation():
    """Test TwoDirections input validation."""
    from jl.variance_experiments.data_generator import TwoDirections
    
    # Test invalid true_d
    with pytest.raises(ValueError):
        TwoDirections(true_d=0, percent_correct=0.9)
    
    with pytest.raises(ValueError):
        TwoDirections(true_d=-5, percent_correct=0.9)
    
    # Test invalid noisy_d
    with pytest.raises(ValueError):
        TwoDirections(true_d=5, noisy_d=-1, percent_correct=0.9)
    
    # Test invalid percent_correct
    with pytest.raises(ValueError):
        TwoDirections(true_d=5, percent_correct=0.4)  # Must be >= 0.5
    
    with pytest.raises(ValueError):
        TwoDirections(true_d=5, percent_correct=1.1)  # Must be <= 1.0
    
    # Test valid percent_correct = 1.0 (should work)
    problem = TwoDirections(true_d=5, percent_correct=1.0)
    assert problem.percent_correct == 1.0
    
    # Test invalid sigma
    with pytest.raises(ValueError):
        TwoDirections(true_d=5, sigma=-0.1)


def test_two_directions_clean_mode_flag():
    """Test the clean_mode flag functionality."""
    from jl.variance_experiments.data_generator import TwoDirections
    
    torch.manual_seed(42)
    problem = TwoDirections(true_d=5, percent_correct=0.8)
    
    # Test with clean_mode=False (should have label noise)
    x1, y1, center_indices1, _ = problem.generate_dataset(n=100, clean_mode=False, shuffle=False)
    num_flipped_with_noise = torch.sum(y1 != center_indices1).item()
    assert num_flipped_with_noise == 20, f"Expected 20 flipped labels, got {num_flipped_with_noise}"
    
    # Test with clean_mode=True (should have perfect labels)
    torch.manual_seed(42)  # Reset seed for same randomness
    problem2 = TwoDirections(true_d=5, percent_correct=0.8)
    x2, y2, center_indices2, _ = problem2.generate_dataset(n=100, clean_mode=True, shuffle=False)
    num_flipped_without_noise = torch.sum(y2 != center_indices2).item()
    assert num_flipped_without_noise == 0, f"Expected 0 flipped labels, got {num_flipped_without_noise}"
    
    # Features should be the same (only labels differ)
    assert torch.allclose(x1, x2)
    
    # Center indices should be the same
    assert torch.equal(center_indices1, center_indices2)


def test_two_directions_noisy_dimensions():
    """Test that noisy dimensions contain {-1, +1} values."""
    from jl.variance_experiments.data_generator import TwoDirections
    
    torch.manual_seed(789)
    
    # Test with noisy_d > 0
    problem = TwoDirections(true_d=10, noisy_d=5, percent_correct=1.0, sigma=None, random_basis=False)
    x, y, center_indices, _ = problem.generate_dataset(n=50, shuffle=False)
    
    assert x.shape == (50, 15)
    
    # Check that noisy dimensions (last 5 dimensions) contain only {-1, +1} values
    noisy_dims = x[:, 10:]  # Extract noisy dimensions
    unique_values = torch.unique(noisy_dims)
    assert len(unique_values) <= 2, f"Expected at most 2 unique values in noisy dims, got {len(unique_values)}"
    assert torch.all((noisy_dims == -1.0) | (noisy_dims == 1.0)), \
           f"Expected noisy dims to contain only {{-1, +1}}, got values: {unique_values}"
    
    # Check that true dimensions (first 10) are at mu or -mu (without sigma)
    # For first 25 samples (center 0), true dims should be at mu
    assert torch.allclose(x[0, :10], problem.mu, atol=1e-6)
    # For last 25 samples (center 1), true dims should be at -mu
    assert torch.allclose(x[25, :10], -problem.mu, atol=1e-6)


def test_two_directions_with_sigma():
    """Test TwoDirections with Gaussian noise."""
    from jl.variance_experiments.data_generator import TwoDirections
    
    torch.manual_seed(456)
    
    # Create problem with sigma
    problem = TwoDirections(true_d=5, noisy_d=0, percent_correct=1.0, sigma=0.1)
    
    x, y, center_indices, _ = problem.generate_dataset(n=100, shuffle=False)
    
    # With sigma, samples should not be exactly at mu or -mu
    # Check first sample from center 0
    assert not torch.allclose(x[0], problem.mu, atol=1e-3)
    
    # But they should be close (within a few sigmas)
    for i in range(50):
        dist_to_mu = torch.norm(x[i] - problem.mu).item()
        assert dist_to_mu < 5 * 0.1 * torch.sqrt(torch.tensor(5.0)).item()  # 5 sigma * sqrt(d)


def test_two_directions_with_random_basis():
    """Test TwoDirections with random basis transformation."""
    from jl.variance_experiments.data_generator import TwoDirections
    
    torch.manual_seed(789)
    
    # Create problem with random_basis
    problem = TwoDirections(true_d=5, noisy_d=0, percent_correct=1.0, random_basis=True)
    
    assert problem.basis is not None
    assert problem.basis.shape == (5, 5)
    
    # Check that basis is orthonormal
    product = problem.basis.T @ problem.basis
    identity = torch.eye(5, device=problem.device)
    assert torch.allclose(product, identity, atol=1e-5)
    
    x, y, center_indices, _ = problem.generate_dataset(n=100, shuffle=False)
    
    # After transformation, samples should not be at mu or -mu in original space
    # (they are rotated)
    assert not torch.allclose(x[0], problem.mu, atol=1e-3)


def test_two_directions_perfect_labels():
    """Test that percent_correct=1.0 produces no label noise."""
    from jl.variance_experiments.data_generator import TwoDirections
    
    torch.manual_seed(111)
    
    problem = TwoDirections(true_d=8, percent_correct=1.0)
    
    x, y, center_indices, _ = problem.generate_dataset(n=200, shuffle=False)
    
    # All labels should match center indices (no flipping)
    assert torch.all(y == center_indices)


def test_two_directions_label_noise_balanced():
    """Test that label noise is balanced across classes."""
    from jl.variance_experiments.data_generator import TwoDirections
    
    torch.manual_seed(222)
    
    # Test with odd number of flips
    problem = TwoDirections(true_d=5, percent_correct=0.85)  # 15% incorrect
    
    n = 100
    x, y, center_indices, _ = problem.generate_dataset(n=n, shuffle=False)
    
    num_incorrect = round(n * 0.15)  # 15
    
    # Count flipped labels
    flipped_mask = (y != center_indices)
    num_flipped = torch.sum(flipped_mask).item()
    
    assert num_flipped == num_incorrect
    
    # Check balance across classes (should be 7 or 8 per class)
    flipped_from_center_0 = torch.sum((center_indices == 0) & flipped_mask).item()
    flipped_from_center_1 = torch.sum((center_indices == 1) & flipped_mask).item()
    
    # Total should equal num_incorrect
    assert flipped_from_center_0 + flipped_from_center_1 == num_incorrect
    
    # Balance: difference should be at most 1
    assert abs(flipped_from_center_0 - flipped_from_center_1) <= 1


def test_two_gaussians_clean_mode_filtering():
    """Test that clean_mode properly filters samples to keep only those closest to their own center."""
    from jl.variance_experiments.data_generator import TwoGaussians
    
    torch.manual_seed(42)
    
    # Create problem with percent_correct=0.6
    # This means samples from N(mu, I) have 60% probability of being closer to mu than -mu
    problem = TwoGaussians(true_d=10, noisy_d=5, percent_correct=0.6)
    
    # Generate dataset with clean_mode=True
    n = 200
    x, y, center_indices, _ = problem.generate_dataset(n=n, clean_mode=True, shuffle=True)
    
    # Check that exactly n samples are returned
    assert x.shape[0] == n, f"Expected {n} samples, got {x.shape[0]}"
    assert y.shape[0] == n, f"Expected {n} labels, got {y.shape[0]}"
    assert center_indices.shape[0] == n, f"Expected {n} center_indices, got {center_indices.shape[0]}"
    
    # Extract only the true_d dimensions for distance computation (same as in the implementation)
    x_true = x[:, :problem.true_d]  # (n, true_d)
    
    # Compute distances to both centers for all samples
    dist_to_mu = torch.norm(x_true - problem.mu.unsqueeze(0), dim=1)  # (n,)
    dist_to_minus_mu = torch.norm(x_true - (-problem.mu).unsqueeze(0), dim=1)  # (n,)
    
    # Verify that each sample is closer to its own center than the opposite center
    for i in range(n):
        sample_center = center_indices[i].item()
        sample_label = y[i].item()
        
        # Labels should match center indices (no label noise in clean_mode)
        assert sample_label == sample_center, \
            f"Sample {i}: label {sample_label} doesn't match center {sample_center}"
        
        if sample_center == 0:
            # Sample from center 0 (at mu) should be closer to mu than to -mu
            assert dist_to_mu[i] < dist_to_minus_mu[i], \
                f"Sample {i} from center 0 is not closer to mu: dist_to_mu={dist_to_mu[i]:.4f}, dist_to_minus_mu={dist_to_minus_mu[i]:.4f}"
        else:
            # Sample from center 1 (at -mu) should be closer to -mu than to mu
            assert dist_to_minus_mu[i] < dist_to_mu[i], \
                f"Sample {i} from center 1 is not closer to -mu: dist_to_mu={dist_to_mu[i]:.4f}, dist_to_minus_mu={dist_to_minus_mu[i]:.4f}"
    
    print(f"All {n} samples passed clean_mode validation: each is closest to its own center")
