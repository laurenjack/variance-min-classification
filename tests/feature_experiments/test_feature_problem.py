"""
Tests for SingleFeatures problem implementation.

Verifies:
- Output shapes are correct
- Class balance is maintained
- Features are orthonormal
- Each sample activates exactly one feature
"""
import pytest
import torch

from jl.feature_experiments.feature_problem import Kaleidoscope, SingleFeatures


@pytest.fixture
def device():
    """Provide device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def problem(device):
    """Create a SingleFeatures problem instance."""
    true_d = 20  # input dimensionality
    f = 5   # number of features/classes
    return SingleFeatures(true_d=true_d, f=f, device=device)


@pytest.fixture
def kaleidoscope_problem(device):
    """Create a Kaleidoscope problem with orthogonal layers."""
    d = 16
    centers = [4, 6, 5]
    return Kaleidoscope(d=d, centers=centers, device=device)


def test_initialization(device):
    """Test that SingleFeatures initializes correctly with valid parameters."""
    true_d = 20
    f = 5
    problem = SingleFeatures(true_d=true_d, f=f, device=device)
    
    assert problem.d == true_d
    assert problem.f == f
    assert problem.num_class == f
    assert problem.Q.shape == (f, true_d)


def test_initialization_invalid_params(device):
    """Test that SingleFeatures raises errors for invalid parameters."""
    # Negative true_d should fail
    with pytest.raises(ValueError, match="true_d must be positive"):
        SingleFeatures(true_d=-5, f=3, device=device)
    
    # Negative f should fail
    with pytest.raises(ValueError, match="f must be positive"):
        SingleFeatures(true_d=10, f=-3, device=device)
    
    # Zero f should fail
    with pytest.raises(ValueError, match="f must be positive"):
        SingleFeatures(true_d=10, f=0, device=device)


def test_orthonormal_rows(problem, device):
    """Test that Q has orthonormal rows when f <= d."""
    Q = problem.Q
    f = problem.f
    d = problem.d
    
    # This test only applies when f <= d
    if f > d:
        pytest.skip("Orthonormal rows test only applies when f <= d")
    
    # Compute Gram matrix Q @ Q.T (should be identity)
    gram = Q @ Q.T
    identity = torch.eye(f, device=device)
    max_diff = torch.max(torch.abs(gram - identity)).item()
    
    assert max_diff < 1e-5, f"Q rows are not orthonormal! Max diff: {max_diff}"


def test_generate_dataset_shapes(problem):
    """Test that generate_dataset returns tensors with correct shapes."""
    n = 100
    d = problem.d

    x, y, center_indices = problem.generate_dataset(n, shuffle=False)

    assert x.shape == (n, d), f"Expected x shape ({n}, {d}), got {x.shape}"
    assert y.shape == (n,), f"Expected y shape ({n},), got {y.shape}"
    assert center_indices.shape == (n,), f"Expected center_indices shape ({n},), got {center_indices.shape}"
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert center_indices.dtype == torch.int64


def test_center_indices_equals_labels(problem):
    """Test that center_indices equals y for SingleFeatures."""
    n = 100
    x, y, center_indices = problem.generate_dataset(n, shuffle=False)
    
    assert torch.all(center_indices == y), "center_indices should equal y!"


def test_class_balance_no_shuffle(problem):
    """Test that classes are perfectly balanced without shuffle."""
    n = 100
    f = problem.f
    
    x, y, center_indices = problem.generate_dataset(n, shuffle=False)
    
    # Check each class gets the expected number of samples
    for class_idx in range(f):
        count = torch.sum(y == class_idx).item()
        expected = (n // f) + (1 if class_idx < (n % f) else 0)
        assert count == expected, f"Class {class_idx}: expected {expected}, got {count}"


def test_class_balance_with_shuffle(problem):
    """Test that class balance is preserved with shuffle."""
    n = 100
    f = problem.f
    
    x, y, center_indices = problem.generate_dataset(n, shuffle=True)
    
    # Check all classes are represented
    unique_classes = torch.unique(y).sort()[0]
    expected_classes = torch.arange(f, device=y.device)
    assert torch.all(unique_classes == expected_classes), "Not all classes present after shuffle!"
    
    # Check counts are balanced
    for class_idx in range(f):
        count = torch.sum(y == class_idx).item()
        expected = (n // f) + (1 if class_idx < (n % f) else 0)
        assert count == expected, f"Class balance changed after shuffle for class {class_idx}"


def test_one_hot_feature_activation(problem):
    """Test that each sample activates exactly one feature."""
    n = 100
    f = problem.f
    
    x, y, center_indices = problem.generate_dataset(n, shuffle=False)
    
    # Project back to feature space
    x_projected = x @ problem.Q.T  # (n, f)
    
    # Each row should have one coordinate close to 1, others close to 0
    max_vals, activated_features = torch.max(torch.abs(x_projected), dim=1)
    
    # Check max activations are close to 1
    assert torch.allclose(max_vals, torch.ones_like(max_vals), atol=1e-6), \
        f"Max activations not close to 1: mean={max_vals.mean()}, min={max_vals.min()}"
    
    # Check that activated features match labels
    matches = (activated_features == y).sum().item()
    assert matches == n, f"Only {matches}/{n} activated features match labels!"
    
    # Check that non-activated features are small
    for i in range(min(10, n)):
        active_feature = int(y[i].item())
        activations = x_projected[i].abs()
        non_active = torch.cat([activations[:active_feature], activations[active_feature+1:]])
        if non_active.numel() > 0:
            max_non_active = non_active.max().item()
            assert max_non_active < 1e-6, \
                f"Sample {i} has large non-active feature: {max_non_active}"


def test_generator_reproducibility(device):
    """Test that using a generator produces reproducible results."""
    true_d = 20
    f = 5
    n = 50
    seed = 42
    
    # Generate with same seed twice
    gen1 = torch.Generator(device=device).manual_seed(seed)
    problem1 = SingleFeatures(true_d=true_d, f=f, device=device, generator=gen1)
    x1, y1, _ = problem1.generate_dataset(n, shuffle=True)
    
    gen2 = torch.Generator(device=device).manual_seed(seed)
    problem2 = SingleFeatures(true_d=true_d, f=f, device=device, generator=gen2)
    x2, y2, _ = problem2.generate_dataset(n, shuffle=True)
    
    # Q matrices should be identical
    assert torch.allclose(problem1.Q, problem2.Q), "Q matrices differ with same seed!"
    
    # Generated data should be identical
    assert torch.allclose(x1, x2), "Generated x differs with same seed!"
    assert torch.all(y1 == y2), "Generated y differs with same seed!"


def test_different_n_values(problem):
    """Test generation with various values of n."""
    f = problem.f
    
    # Test with n not divisible by f
    for n in [47, 53, 99, 101]:
        x, y, center_indices = problem.generate_dataset(n, shuffle=False)
        
        assert x.shape[0] == n
        assert y.shape[0] == n
        
        # Check that all classes are represented (as long as n >= f)
        if n >= f:
            assert len(torch.unique(y)) == f, f"Not all classes present for n={n}"


def test_edge_case_n_equals_f(problem):
    """Test generation when n equals f (one sample per class)."""
    f = problem.f
    n = f
    
    x, y, center_indices = problem.generate_dataset(n, shuffle=False)
    
    assert x.shape[0] == n
    assert len(torch.unique(y)) == f
    
    # Each class should have exactly one sample
    for class_idx in range(f):
        count = torch.sum(y == class_idx).item()
        assert count == 1, f"Class {class_idx} has {count} samples, expected 1"


def test_edge_case_small_n(problem):
    """Test generation when n < f."""
    f = problem.f
    n = max(1, f - 2)  # n = f-2, but at least 1
    
    x, y, center_indices = problem.generate_dataset(n, shuffle=False)
    
    assert x.shape[0] == n
    assert y.shape[0] == n
    # Not all classes will be represented, but it should still work
    assert len(torch.unique(y)) == n


def test_invalid_n(problem):
    """Test that invalid n values raise errors."""
    with pytest.raises(ValueError, match="n must be positive"):
        problem.generate_dataset(0)
    
    with pytest.raises(ValueError, match="n must be positive"):
        problem.generate_dataset(-5)


# Tests for f > d case (UNTF)
@pytest.fixture
def problem_f_gt_d(device):
    """Create a SingleFeatures problem instance with f > d."""
    true_d = 5   # input dimensionality
    f = 10  # number of features/classes (f > d)
    return SingleFeatures(true_d=true_d, f=f, is_orthogonal=False, device=device)


def test_initialization_f_gt_d(device):
    """Test that SingleFeatures initializes correctly with f > d."""
    true_d = 5
    f = 10
    problem = SingleFeatures(true_d=true_d, f=f, is_orthogonal=False, device=device)
    
    assert problem.d == true_d
    assert problem.f == f
    assert problem.num_class == f
    assert problem.Q.shape == (f, true_d)


def test_untf_unit_norm_rows(problem_f_gt_d, device):
    """Test that Q has unit norm rows when f > d (UNTF property)."""
    Q = problem_f_gt_d.Q
    f = problem_f_gt_d.f
    
    # Check each row has unit norm
    row_norms = torch.norm(Q, dim=1)
    expected_norm = torch.ones(f, device=device)
    max_diff = torch.max(torch.abs(row_norms - expected_norm)).item()
    
    assert max_diff < 1e-5, f"Q rows are not unit norm! Max diff: {max_diff}"


def test_untf_tight_frame_property(problem_f_gt_d, device):
    """Test that Q has unit norm rows when is_orthogonal=False (random features)."""
    Q = problem_f_gt_d.Q
    f = problem_f_gt_d.f
    
    # When is_orthogonal=False, we use random unit-norm features with rejection sampling
    # Check each row has unit norm
    row_norms = torch.norm(Q, dim=1)
    expected_norm = torch.ones(f, device=device)
    max_diff = torch.max(torch.abs(row_norms - expected_norm)).item()
    
    assert max_diff < 1e-5, f"Q rows are not unit norm! Max diff: {max_diff}"


def test_generate_dataset_f_gt_d_shapes(problem_f_gt_d):
    """Test that generate_dataset returns correct shapes for f > d."""
    n = 100
    d = problem_f_gt_d.d

    x, y, center_indices = problem_f_gt_d.generate_dataset(n, shuffle=False)

    assert x.shape == (n, d), f"Expected x shape ({n}, {d}), got {x.shape}"
    assert y.shape == (n,), f"Expected y shape ({n},), got {y.shape}"
    assert center_indices.shape == (n,), f"Expected center_indices shape ({n},), got {center_indices.shape}"
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert center_indices.dtype == torch.int64


def test_class_balance_f_gt_d(problem_f_gt_d):
    """Test that classes are balanced for f > d case."""
    n = 100
    f = problem_f_gt_d.f
    
    x, y, center_indices = problem_f_gt_d.generate_dataset(n, shuffle=False)
    
    # Check each class gets the expected number of samples
    for class_idx in range(f):
        count = torch.sum(y == class_idx).item()
        expected = (n // f) + (1 if class_idx < (n % f) else 0)
        assert count == expected, f"Class {class_idx}: expected {expected}, got {count}"


def test_one_hot_feature_activation_f_gt_d(problem_f_gt_d):
    """Test that each sample activates exactly one feature for f > d."""
    n = 100
    f = problem_f_gt_d.f
    
    x, y, center_indices = problem_f_gt_d.generate_dataset(n, shuffle=False)
    
    # Project back to feature space
    x_projected = x @ problem_f_gt_d.Q.T  # (n, f)
    
    # Each row should have one coordinate close to 1, others close to 0
    max_vals, activated_features = torch.max(torch.abs(x_projected), dim=1)
    
    # Check max activations are close to 1
    assert torch.allclose(max_vals, torch.ones_like(max_vals), atol=1e-5), \
        f"Max activations not close to 1: mean={max_vals.mean()}, min={max_vals.min()}"
    
    # Check that activated features match labels
    matches = (activated_features == y).sum().item()
    assert matches == n, f"Only {matches}/{n} activated features match labels!"


def test_generator_reproducibility_f_gt_d(device):
    """Test that using a generator produces reproducible results for f > d."""
    true_d = 5
    f = 10
    n = 50
    seed = 42
    
    # Generate with same seed twice
    gen1 = torch.Generator(device=device).manual_seed(seed)
    problem1 = SingleFeatures(true_d=true_d, f=f, is_orthogonal=False, device=device, generator=gen1)
    x1, y1, _ = problem1.generate_dataset(n, shuffle=True)
    
    gen2 = torch.Generator(device=device).manual_seed(seed)
    problem2 = SingleFeatures(true_d=true_d, f=f, is_orthogonal=False, device=device, generator=gen2)
    x2, y2, _ = problem2.generate_dataset(n, shuffle=True)
    
    # Q matrices should be identical
    assert torch.allclose(problem1.Q, problem2.Q, atol=1e-5), "Q matrices differ with same seed!"
    
    # Generated data should be identical
    assert torch.allclose(x1, x2, atol=1e-5), "Generated x differs with same seed!"
    assert torch.all(y1 == y2), "Generated y differs with same seed!"


def test_edge_case_f_equals_d(device):
    """Test the boundary case where f == d."""
    true_d = 10
    f = 10
    problem = SingleFeatures(true_d=true_d, f=f, device=device)
    
    assert problem.Q.shape == (f, true_d)
    # When f == d, should have orthonormal rows
    gram = problem.Q @ problem.Q.T
    identity = torch.eye(f, device=device)
    assert torch.allclose(gram, identity, atol=1e-5), "When f == d, Q should have orthonormal rows"


# ---------------------------------------------------------------------------
# Kaleidoscope problem tests
# ---------------------------------------------------------------------------


def test_kaleidoscope_initialization(kaleidoscope_problem, device):
    """Kaleidoscope should create one matrix per layer with correct shapes."""
    assert kaleidoscope_problem.d == 16
    assert kaleidoscope_problem.num_classes() == 5
    assert len(kaleidoscope_problem.Q_layers) == 3

    for count, Q_l in zip(kaleidoscope_problem.centers, kaleidoscope_problem.Q_layers):
        assert Q_l.shape == (count, kaleidoscope_problem.d)
        gram = Q_l @ Q_l.T
        identity = torch.eye(count, device=device, dtype=Q_l.dtype)
        assert torch.allclose(gram, identity, atol=1e-5)


def test_kaleidoscope_generate_dataset_shapes(kaleidoscope_problem):
    n = 64
    x, y, centers = kaleidoscope_problem.generate_dataset(n, shuffle=False)

    assert x.shape == (n, kaleidoscope_problem.d)
    assert x.dtype == torch.float32
    assert y.shape == (n,)
    assert y.dtype == torch.int64
    assert isinstance(centers, list)


def test_kaleidoscope_labels_within_range(kaleidoscope_problem):
    n = 100
    x, y, centers = kaleidoscope_problem.generate_dataset(n, shuffle=True)

    assert x.shape[0] == n
    assert torch.all((0 <= y) & (y < kaleidoscope_problem.num_classes()))
    assert isinstance(centers, list)


def test_kaleidoscope_generator_reproducibility(device):
    d = 10
    centers = [3, 5]
    n = 40
    seed = 123

    gen1 = torch.Generator(device=device).manual_seed(seed)
    problem1 = Kaleidoscope(d=d, centers=centers, device=device, generator=gen1)
    x1, y1, _ = problem1.generate_dataset(n, shuffle=True)

    gen2 = torch.Generator(device=device).manual_seed(seed)
    problem2 = Kaleidoscope(d=d, centers=centers, device=device, generator=gen2)
    x2, y2, _ = problem2.generate_dataset(n, shuffle=True)

    for Q1, Q2 in zip(problem1.Q_layers, problem2.Q_layers):
        assert torch.allclose(Q1, Q2, atol=1e-5)

    assert torch.allclose(x1, x2, atol=1e-5)
    assert torch.all(y1 == y2)


def test_kaleidoscope_invalid_centers(device):
    with pytest.raises(ValueError, match="centers must be a non-empty sequence"):
        Kaleidoscope(d=10, centers=[], device=device)

    with pytest.raises(ValueError, match="C_0 must be positive"):
        Kaleidoscope(d=10, centers=[0], device=device)

    with pytest.raises(ValueError, match="C_0 must be <= d"):
        Kaleidoscope(d=5, centers=[10], device=device)

