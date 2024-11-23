import torch
import numpy as np


from src.posterior_minimizer import variance
from src import dataset_creator, custom_modules as cm

def test_relu_mlp():
    n = 10000
    d = 20
    sizes = [d, 10, 1]

    analytical = variance.Analytical()
    empirical = variance.Empirical()
    empiricalAtFlat = variance.EmpiricalAtFlat()
    problem = dataset_creator.SingleDirectionGaussian(d=d)
    x, y = problem.generate_dataset(n)
    model = cm.Mlp(sizes, is_bias=False, all_linear=False)

    analytical_result = analytical.calculate(model, x, y)
    empirical_result = empirical.calculate(model, x, y)
    eaf_result = empiricalAtFlat.calculate(model, x, y)

    for a, e, eaf in zip(analytical_result, empirical_result, eaf_result):
        print(a)
        print(eaf)


def assert_tensors_almost_equal(tensor1, tensor2, rtol=1e-5, atol=1e-8, msg=None):
    """
    Compare two tensors for approximate equality within a tolerance level.
    Raises AssertionError if tensors are not almost equal.

    Args:
        tensor1 (torch.Tensor): First tensor to compare
        tensor2 (torch.Tensor): Second tensor to compare
        rtol (float): Relative tolerance
        atol (float): Absolute tolerance
        msg (str, optional): Custom error message

    Raises:
        AssertionError: If tensors are not almost equal or have different shapes
    """
    # Check if shapes match
    if tensor1.shape != tensor2.shape:
        raise AssertionError(f"Tensor shapes don't match: {tensor1.shape} != {tensor2.shape}")

    # Calculate absolute difference
    diff = torch.abs(tensor1 - tensor2)

    # Calculate tolerance for each element
    tol = atol + rtol * torch.abs(tensor2)

    # Check if any element exceeds tolerance
    if torch.any(diff > tol):
        # Find the maximum difference for informative error message
        max_diff = torch.max(diff).item()
        max_diff_idx = torch.argmax(diff)
        max_diff_coords = np.unravel_index(max_diff_idx.item(), tensor1.shape)

        error_msg = msg if msg else ""
        error_msg += f"\nTensors are not almost equal!"
        error_msg += f"\nMaximum difference: {max_diff}"
        error_msg += f"\nAt position: {max_diff_coords}"
        error_msg += f"\nValues at max difference: {tensor1[max_diff_coords]} != {tensor2[max_diff_coords]}"

        raise AssertionError(error_msg)


if __name__ == '__main__':
    test_relu_mlp()
