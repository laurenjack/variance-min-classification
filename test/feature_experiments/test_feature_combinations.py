import torch
import pytest
from jl.feature_experiments.feature_combinations import (
    FeatureCombinations,
    _build_consolidation_mapping,
    _build_favourites_consolidation_mapping,
)


class TestFeatureCombinations:
    """Test suite for FeatureCombinations problem."""

    def test_basic_properties(self):
        """Test basic properties of FeatureCombinations."""
        problem = FeatureCombinations(num_layers=2)
        assert problem.num_layers == 2
        assert problem.d == 8  # 2 * 2^2
        assert problem.num_class == 4
        assert not problem.has_favourites
        assert not problem.random_basis

    def test_has_favourites_parameter(self):
        """Test has_favourites parameter is stored correctly."""
        problem_no_fav = FeatureCombinations(num_layers=2, has_favourites=False)
        problem_with_fav = FeatureCombinations(num_layers=2, has_favourites=True)

        assert not problem_no_fav.has_favourites
        assert problem_with_fav.has_favourites

    def test_generate_dataset_basic(self):
        """Test basic dataset generation."""
        problem = FeatureCombinations(num_layers=2)
        x, y, center_indices_list = problem.generate_dataset(n=100)

        assert x.shape == (100, 8)
        assert y.shape == (100,)
        assert len(center_indices_list) == 2  # atomic + final layer
        assert center_indices_list[0].shape == (100, 2)  # 2 atomic subsections
        assert center_indices_list[1].shape == (100,)  # final layer

        # Check y values are in valid range
        assert torch.all((y >= 0) & (y < 4))

    def test_favourites_constraints(self):
        """Test that favourites mappings satisfy all constraints."""
        device = torch.device("cpu")
        generator = torch.Generator().manual_seed(42)

        # Test multiple mappings to ensure they all satisfy constraints
        for _ in range(10):
            mapping = _build_favourites_consolidation_mapping(generator, device)

            # Constraint 1: Each output feature has exactly one favourite on each side
            # and no feature occurs more than twice for any output feature
            for output_feature in range(4):
                # Get all combinations that map to this output feature
                output_combos = [i for i, m in enumerate(mapping) if m == output_feature]

                # Extract left and right features for these combinations
                left_features = [combo // 4 for combo in output_combos]
                right_features = [combo % 4 for combo in output_combos]

                # Count occurrences
                left_counts = torch.bincount(torch.tensor(left_features), minlength=4)
                right_counts = torch.bincount(torch.tensor(right_features), minlength=4)

                # Each output feature should have exactly one favourite (count=2) on each side
                assert torch.sum(left_counts == 2) == 1, f"Output {output_feature} left side"
                assert torch.sum(right_counts == 2) == 1, f"Output {output_feature} right side"

                # No feature should occur more than twice
                assert torch.all(left_counts <= 2), f"Output {output_feature} left side"
                assert torch.all(right_counts <= 2), f"Output {output_feature} right side"

            # Constraint 2: No two output features have the same favourites
            favourites = []
            for output_feature in range(4):
                output_combos = [i for i, m in enumerate(mapping) if m == output_feature]
                left_features = [combo // 4 for combo in output_combos]
                right_features = [combo % 4 for combo in output_combos]

                left_counts = torch.bincount(torch.tensor(left_features), minlength=4)
                right_counts = torch.bincount(torch.tensor(right_features), minlength=4)

                left_fav = torch.argmax(left_counts).item()
                right_fav = torch.argmax(right_counts).item()
                favourites.append((left_fav, right_fav))

            # Check all favourites are unique
            assert len(set(favourites)) == 4, f"Duplicate favourites found: {favourites}"

            # Constraint 3: Each input feature occurs exactly 4 times across all mappings
            all_left = []
            all_right = []
            for output_feature in range(4):
                output_combos = [i for i, m in enumerate(mapping) if m == output_feature]
                all_left.extend([combo // 4 for combo in output_combos])
                all_right.extend([combo % 4 for combo in output_combos])

            left_total_counts = torch.bincount(torch.tensor(all_left), minlength=4)
            right_total_counts = torch.bincount(torch.tensor(all_right), minlength=4)

            assert torch.all(left_total_counts == 4), f"Left side counts: {left_total_counts}"
            assert torch.all(right_total_counts == 4), f"Right side counts: {right_total_counts}"

    def test_random_vs_favourites_different(self):
        """Test that random and favourites mappings produce different results."""
        device = torch.device("cpu")
        generator = torch.Generator().manual_seed(42)

        random_mapping = _build_consolidation_mapping(generator, device)
        favourites_mapping = _build_favourites_consolidation_mapping(generator, device)

        # They should be different (with very high probability)
        assert not torch.equal(random_mapping, favourites_mapping)

    def test_favourites_deterministic_with_seed(self):
        """Test that favourites mappings are deterministic with the same seed."""
        device = torch.device("cpu")
        generator1 = torch.Generator().manual_seed(123)
        generator2 = torch.Generator().manual_seed(123)

        mapping1 = _build_favourites_consolidation_mapping(generator1, device)
        mapping2 = _build_favourites_consolidation_mapping(generator2, device)

        assert torch.equal(mapping1, mapping2)

    def test_favourites_integration(self):
        """Test that FeatureCombinations with has_favourites=True works end-to-end."""
        problem = FeatureCombinations(num_layers=3, has_favourites=True)
        x, y, center_indices_list = problem.generate_dataset(n=50)

        assert x.shape == (50, 16)  # 2 * 2^3
        assert y.shape == (50,)
        assert len(center_indices_list) == 3  # atomic + 2 consolidation layers

        # Check all layers have correct shapes
        assert center_indices_list[0].shape == (50, 4)  # 16/4 = 4 atomic subsections
        assert center_indices_list[1].shape == (50, 2)  # 4/2 = 2 subsections
        assert center_indices_list[2].shape == (50,)   # final layer

    def test_num_layers_validation(self):
        """Test that num_layers >= 2 is enforced."""
        with pytest.raises(ValueError, match="num_layers must be >= 2"):
            FeatureCombinations(num_layers=1)
