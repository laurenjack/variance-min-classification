import torch
import math


def relu_covariance(Sigma):
    """
    Vectorized version that, given a covariance matrix Sigma (for a Gaussian z),
    returns the covariance matrix for a = ReLU(z) using the formula:

      Cov(ReLU(z_i), ReLU(z_j)) = (sqrt(Sigma_ii * Sigma_jj) / (2*pi)) *
           [ sqrt(1 - rho_ij^2) + rho_ij*(pi/2 + asin(rho_ij)) - 1 ],

    where
      rho_ij = Sigma_ij / (sqrt(Sigma_ii * Sigma_jj)).

    We assume Sigma is a square tensor.
    """
    # Compute the standard deviations (sqrt of diagonal elements)
    diag = torch.sqrt(torch.diag(Sigma))  # shape: [d]
    # Outer product of standard deviations to get sigma_i * sigma_j for all (i,j)
    outer = diag.unsqueeze(1) * diag.unsqueeze(0)  # shape: [d, d]

    # To avoid division by zero, use a small epsilon.
    eps = 1e-6
    valid = outer > eps

    # Initialize correlation matrix R.
    R = torch.zeros_like(Sigma)
    R[valid] = Sigma[valid] / outer[valid]
    # Clamp values to [-1, 1] to avoid numerical issues with asin.
    R = torch.clamp(R, -1.0, 1.0)

    # Compute the term inside the parentheses: sqrt(1-R^2) + R*(pi/2 + asin(R)) - 1.
    term = torch.sqrt(1 - R ** 2) + R * ((math.pi / 2) + torch.asin(R)) - 1
    # Compute the covariance of the ReLU activations.
    cov = outer / (2 * math.pi) * term

    # For any (i,j) where outer was zero, set the covariance to 0.
    cov[~valid] = 0.0
    return cov


def calculate_variance(W_0, W_1, W_2):
    """
    Given weight matrices W_0, W_1, and W_2 (PyTorch tensors), this function
    computes the variance of the network output
         z_2 = W_2 * ReLU(W_1 * ReLU(W_0 * x))
    where x ~ N(0, I).

    The method is as follows:
      1. For x ~ N(0,I), the preactivation of the first layer is:
            z_0 = x @ W_0^T,  so that Cov(z_0) = W_0 W_0^T.
      2. The first-layer activations are a_0 = ReLU(z_0) with covariance:
            Cov(a_0) = relu_covariance(W_0 W_0^T).
      3. Then z_1 = a_0 @ W_1^T, so Cov(z_1) = W_1 Cov(a_0) W_1^T.
      4. The second-layer activations are a_1 = ReLU(z_1) with covariance:
            Cov(a_1) = relu_covariance(Cov(z_1)).
      5. Finally, z_2 = a_1 @ W_2^T and its variance is:
            Var(z_2) = W_2 Cov(a_1) W_2^T.
    """
    # Step 1: First-layer preactivation covariance.
    Sigma_z0 = W_0 @ W_0.t()  # Shape: [d0, d0]

    # Step 2: First-layer activation covariance.
    Sigma_a0 = relu_covariance(Sigma_z0)

    # Step 3: Second-layer preactivation covariance.
    Sigma_z1 = W_1 @ Sigma_a0 @ W_1.t()  # Shape: [d1, d1]

    # Step 4: Second-layer activation covariance.
    Sigma_a1 = relu_covariance(Sigma_z1)

    # Step 5: Output variance.
    # Assuming W_2 is of shape [1, d1] so that the output is scalar.
    variance = W_2 @ Sigma_a1 @ W_2.t()  # 1x1 tensor.
    return variance.squeeze()  # Return as a scalar.


def empirical_variance(n, batch_size, W_0, W_1, W_2):
    """
    Computes the empirical variance of the network output by generating n input
    samples from N(0, I), processing them through the network in batches, and
    then calculating the variance from the resulting n outputs.

    The network is:
         z_2 = W_2 * ReLU(W_1 * ReLU(W_0 * x))
    with x ~ N(0,I).

    Parameters:
      n          - total number of input samples
      batch_size - number of samples processed at once
      W_0, W_1, W_2 - weight matrices (PyTorch tensors)

    Returns:
      A scalar representing the empirical variance.
    """
    # Determine the input dimension from W_0.
    d_in = W_0.shape[1]
    outputs = []

    for i in range(0, n, batch_size):
        current_batch = min(batch_size, n - i)
        # Generate a batch of inputs: shape [current_batch, d_in]
        x = torch.randn(current_batch, d_in)
        # Forward pass through the network:
        # First layer: x @ W_0^T, then ReLU.
        z0 = x @ W_0.t()
        a0 = torch.relu(z0)
        # Second layer: a0 @ W_1^T, then ReLU.
        z1 = a0 @ W_1.t()
        a1 = torch.relu(z1)
        # Output layer: a1 @ W_2^T.
        z2 = a1 @ W_2.t()  # Shape: [current_batch, 1]
        outputs.append(z2)

    # Concatenate all batch outputs into a single tensor of shape [n, 1].
    outputs = torch.cat(outputs, dim=0).squeeze(-1)  # Shape: [n]
    # Compute and return the variance (using the population variance formula).
    return outputs.var(unbiased=False).item()


# Example usage:
if __name__ == "__main__":
    # Define dimensions. For instance, assume:
    #  - Input dimension: 3.
    #  - First hidden layer: 4 units.
    #  - Second hidden layer: 3 units.
    #  - Output layer: 1 unit.
    d_in = 1000
    d0 = 1000
    d1 = 1000
    # Create random weight matrices.
    W_0 = torch.randn(d0, d_in)
    W_1 = torch.randn(d1, d0)
    W_2 = torch.randn(1, d1)

    # Calculate theoretical variance.
    theor_var = calculate_variance(W_0, W_1, W_2)
    # Calculate empirical variance from, say, 100000 samples, processing in batches of 1000.
    emp_var = empirical_variance(n=100000, batch_size=1000, W_0=W_0, W_1=W_1, W_2=W_2)

    print("Theoretical variance:", theor_var.item() if isinstance(theor_var, torch.Tensor) else theor_var)
    print("Empirical variance1:  ", emp_var)
