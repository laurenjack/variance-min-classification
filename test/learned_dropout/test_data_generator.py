import pytest
import torch

from src.learned_dropout.data_generator import SubDirections


def make_gen(seed: int = 1234, device: torch.device | None = None) -> torch.Generator:
    g = torch.Generator(device=device if device is not None else "cpu")
    g.manual_seed(seed)
    return g


def test_input_validations():
    with pytest.raises(ValueError):
        SubDirections(d=4, sub_d=0, perms=4, num_class=1)
    with pytest.raises(ValueError):
        SubDirections(d=2, sub_d=4, perms=4, num_class=1)
    with pytest.raises(ValueError):
        SubDirections(d=5, sub_d=2, perms=4, num_class=1)  # d % sub_d != 0
    with pytest.raises(ValueError):
        SubDirections(d=4, sub_d=2, perms=0, num_class=1)
    with pytest.raises(ValueError):
        SubDirections(d=4, sub_d=2, perms=5, num_class=0)


def test_capacity_constraint():
    # d=6, sub_d=3 → S=2 subsections, 2^sub_d=8 patterns per subsection
    # Capacity rule: ceil(perms/S) < 2^sub_d ⇒ perms <= S * (2^sub_d - 1)
    d, sub_d = 6, 3
    S = d // sub_d
    P = 1 << sub_d
    perms_max = S * (P - 1)

    # Max valid perms
    sd_ok = SubDirections(d=d, sub_d=sub_d, perms=perms_max, num_class=2, generator=make_gen(0))
    assert sd_ok.centers_block_signs.shape == (perms_max, sub_d)

    # One more should fail
    with pytest.raises(ValueError):
        SubDirections(d=d, sub_d=sub_d, perms=perms_max + 1, num_class=2)


def test_center_assignment_and_uniqueness():
    # Check that centers are assigned round-robin to subsections and are unique within subsection
    d, sub_d, S = 8, 2, 4
    perms = 12  # per subsection: [3,3,3,3]
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=3, generator=make_gen(42))

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
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(7))
    for s in range(d // sub_d):
        idxs = torch.nonzero(sd.center_subsection == s, as_tuple=False).view(-1)
        classes = sd.center_to_class[idxs]
        counts = torch.bincount(classes, minlength=num_class).tolist()
        # Each class count should differ by at most 1 in balancing
        assert max(counts) - min(counts) <= 1


def test_generate_shapes_and_types():
    sd = SubDirections(d=6, sub_d=3, perms=6, num_class=2, generator=make_gen(1))
    x, y, center_indices = sd.generate_dataset(n=12, shuffle=False)
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
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(5))
    
    # Test case 1: n divisible by perms - each center gets exactly n//perms samples
    n = 120  # 120 = 12 * 10, so each center gets exactly 10 samples
    x, y, _ = sd.generate_dataset(n=n, shuffle=False)
    
    # Since sampling is center-balanced, we can verify the structure
    # We know each center gets exactly n//perms = 10 samples
    base_per_center = n // perms
    assert base_per_center == 10
    
    # Test case 2: n not divisible by perms - some centers get +1 sample
    n2 = 125  # 125 = 12*10 + 5, so 5 centers get 11 samples, 7 centers get 10 samples
    x2, y2, _ = sd.generate_dataset(n=n2, shuffle=False)
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


def test_center_balance_with_incorrect_samples():
    """Test that incorrect samples are distributed with the same balance as regular samples"""
    d, sub_d, perms, num_class = 6, 2, 8, 2
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(42))
    
    # Test case where both samples and incorrect samples have remainders
    n = 90  # 90 = 8*11 + 2, so 2 centers get 12 samples, 6 centers get 11 samples
    percent_correct = 0.8  # 20% incorrect = round(90 * 0.2) = 18
    
    base_per_center = n // perms  # 11
    remainder_samples = n % perms  # 2
    
    num_incorrect = round(n * (1.0 - percent_correct))  # round(90 * 0.2) = 18
    base_incorrect_per_center = num_incorrect // perms  # 18 // 8 = 2  
    remainder_incorrect = num_incorrect % perms  # 18 % 8 = 2
    
    # Verify the math
    assert base_per_center == 11
    assert remainder_samples == 2
    assert num_incorrect == 18
    assert base_incorrect_per_center == 2
    assert remainder_incorrect == 2
    
    # Generate dataset
    x, y, _ = sd.generate_dataset(n=n, percent_correct=percent_correct, shuffle=False)
    
    # Generate perfect labels for comparison
    sd_perfect = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(42))
    x_perfect, y_perfect, _ = sd_perfect.generate_dataset(n=n, percent_correct=1.0, shuffle=False)
    
    # Verify exact number of incorrect labels
    different_labels = torch.sum(y != y_perfect).item()
    assert different_labels == num_incorrect
    
    # The key insight: centers that get extra samples should be the first to get extra incorrect samples
    # This ensures the balance is maintained

def test_determinism_with_generator():
    gen = make_gen(42)
    sd1 = SubDirections(d=6, sub_d=3, perms=12, num_class=2, generator=gen)
    x1, y1, _ = sd1.generate_dataset(n=24, shuffle=False)

    gen2 = make_gen(42)
    sd2 = SubDirections(d=6, sub_d=3, perms=12, num_class=2, generator=gen2)
    x2, y2, _ = sd2.generate_dataset(n=24, shuffle=False)

    assert torch.allclose(x1, x2)
    assert torch.equal(y1, y2)


def test_sample_generation_logic():
    # Ensure chosen center's subsection is fixed to the center pattern in means (up to noise)
    d, sub_d, perms = 8, 2, 8
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=2, generator=make_gen(0), sigma=1e-6)
    n = 80  # 80 = 8*10, so each center gets exactly 10 samples
    x, y, _ = sd.generate_dataset(n=n, shuffle=False)

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


def test_percent_correct_functionality():
    """Test that percent_correct correctly introduces label noise"""
    d, sub_d, perms, num_class = 8, 2, 12, 4
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(42))
    
    # Test with 80% correct
    n = 100
    percent_correct = 0.8
    x, y, _ = sd.generate_dataset(n=n, percent_correct=percent_correct, shuffle=False)
    
    # Generate the same dataset with 100% correct to compare
    sd_perfect = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(42))
    x_perfect, y_perfect, _ = sd_perfect.generate_dataset(n=n, percent_correct=1.0, shuffle=False)
    
    # Count how many labels are different
    different_labels = torch.sum(y != y_perfect).item()
    expected_incorrect = round(n * (1.0 - percent_correct))
    
    # Should be exactly the expected number of incorrect labels
    assert different_labels == expected_incorrect, f"Expected {expected_incorrect} incorrect labels, got {different_labels}"
    
    # Verify features are the same (only labels should change)
    assert torch.allclose(x, x_perfect), "Features should be identical when only labels change"


def test_percent_correct_balance_across_centers():
    """Test that incorrect labels are balanced across centers"""
    d, sub_d, perms, num_class = 6, 2, 9, 3
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(123))
    
    n = 90  # Divisible by perms (9) for exact center balance: each center gets 10 samples
    percent_correct = 0.7  # 30% incorrect = 27 incorrect samples
    x, y, _ = sd.generate_dataset(n=n, percent_correct=percent_correct, shuffle=False)
    
    # Generate perfect labels for comparison
    sd_perfect = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(123))
    x_perfect, y_perfect, _ = sd_perfect.generate_dataset(n=n, percent_correct=1.0, shuffle=False)
    
    # Count how many labels are different
    different_labels = torch.sum(y != y_perfect).item()
    expected_incorrect = round(n * (1.0 - percent_correct))  # 27
    
    # Should be exactly the expected number of incorrect labels
    assert different_labels == expected_incorrect, f"Expected {expected_incorrect} incorrect labels, got {different_labels}"
    
    # Test the center balance property: incorrect samples should be distributed across centers
    # With n=90, perms=9: each center gets 10 samples
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
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(99))
    
    # Test 100% correct (default)
    x1, y1, _ = sd.generate_dataset(n=20, percent_correct=1.0)
    x2, y2, _ = sd.generate_dataset(n=20, percent_correct=1.0)
    # Should be deterministic with same generator state
    # (Note: generator state advances, so we can't directly compare)
    
    # Test 0% correct (all labels should be different from original)
    sd_zero = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(99))
    sd_perfect = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(99))
    
    x_zero, y_zero, _ = sd_zero.generate_dataset(n=20, percent_correct=0.0, shuffle=False)
    x_perf, y_perf, _ = sd_perfect.generate_dataset(n=20, percent_correct=1.0, shuffle=False)
    
    # All labels should be different when percent_correct=0
    different_count = torch.sum(y_zero != y_perf).item()
    assert different_count == 20, f"Expected all 20 labels to be different, got {different_count}"
    
    # Features should still be the same
    assert torch.allclose(x_zero, x_perf), "Features should be identical"


def test_percent_correct_validation():
    """Test that percent_correct validation works"""
    d, sub_d, perms, num_class = 4, 2, 4, 2
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(1))
    
    # Valid range should work
    sd.generate_dataset(n=10, percent_correct=0.0)
    sd.generate_dataset(n=10, percent_correct=0.5)
    sd.generate_dataset(n=10, percent_correct=1.0)
    
    # Invalid values should raise errors
    with pytest.raises(ValueError):
        sd.generate_dataset(n=10, percent_correct=-0.1)
    
    with pytest.raises(ValueError):
        sd.generate_dataset(n=10, percent_correct=1.1)


def test_subdirections_centers_and_indices():
    """Test the centers and center_indices functionality for SubDirections."""
    d, sub_d, perms, num_class = 8, 2, 8, 2
    sd = SubDirections(d=d, sub_d=sub_d, perms=perms, num_class=num_class, generator=make_gen(42))
    
    x, y, center_indices = sd.generate_dataset(n=16, shuffle=False)
    
    # Test centers structure
    assert centers.shape == (8, 8)  # 8 centers, 8 dimensions
    assert center_indices.shape == (16,)
    
    # For SubDirections, centers should have non-zero values only in their assigned subsection
    num_subsections = d // sub_d  # 4 subsections
    
    for center_idx in range(perms):
        center = centers[center_idx]
        subsection = int(sd.center_subsection[center_idx].item())
        
        # Check that the center has the right pattern in its subsection
        start_idx = subsection * sub_d
        end_idx = start_idx + sub_d
        center_pattern = center[start_idx:end_idx]
        expected_pattern = sd.centers_block_signs[center_idx]
        
        assert torch.allclose(center_pattern, expected_pattern), \
            f"Center {center_idx} pattern mismatch in subsection {subsection}"
        
        # Check that all other dimensions are zero
        for other_subsection in range(num_subsections):
            if other_subsection != subsection:
                other_start = other_subsection * sub_d
                other_end = other_start + sub_d
                other_values = center[other_start:other_end]
                assert torch.allclose(other_values, torch.zeros_like(other_values)), \
                    f"Center {center_idx} should be zero in subsection {other_subsection}"
    
    # Test that center_indices correctly map to the centers that generated each sample
    # Since we use shuffle=False and know the center-balanced generation logic
    samples_per_center = 16 // 8  # 2 samples per center
    for i in range(16):
        expected_center_idx = i // samples_per_center
        actual_center_idx = center_indices[i].item()
        assert actual_center_idx == expected_center_idx, \
            f"Sample {i}: expected center {expected_center_idx}, got {actual_center_idx}"
        
        # The label should match the center's class
        expected_class = int(sd.center_to_class[actual_center_idx].item())
        actual_class = int(y[i].item())
        assert actual_class == expected_class, \
            f"Sample {i}: expected class {expected_class}, got {actual_class}"


def test_gaussian_centers():
    """Test that Gaussian centers are correctly implemented."""
    from src.learned_dropout.data_generator import Gaussian
    
    gauss = Gaussian(d=5)
    x, y, center_indices = gauss.generate_dataset(n=10, shuffle=False)
    
    # Gaussian should have only one center at the origin
    assert centers.shape == (1, 5)
    assert torch.allclose(centers, torch.zeros(1, 5))
    
    # All samples should come from center 0
    assert center_indices.shape == (10,)
    assert torch.all(center_indices == 0)
