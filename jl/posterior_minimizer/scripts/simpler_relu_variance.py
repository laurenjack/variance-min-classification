import torch
import math


def compute_z_t_variance(w_t: torch.Tensor, w_v: torch.Tensor, w_u: torch.Tensor) -> torch.Tensor:
    """
    Compute the theoretical variance of z_t = w_t[0]*a_v + w_t[1]*a_u,
    where:
      - a_v = ReLU(w_v^T x) with x ~ N(0, I),
      - a_u = ReLU(w_u^T x),
    and the moments are given by:
      E[a_v] = ||w_v||/sqrt(2pi),    Var(a_v) = ||w_v||^2/2 - ||w_v||^2/(2pi)
      E[a_u] = ||w_u||/sqrt(2pi),      Var(a_u) = ||w_u||^2/2 - ||w_u||^2/(2pi)
    and
      Cov(a_v,a_u) = (||w_v||*||w_u||/(2pi)) * (sqrt(1 - rho^2) + rho*(pi/2 + arcsin(rho)) - 1),
    with rho = (w_v^T w_u)/(||w_v||*||w_u||).
    """
    # Compute norms for w_v and w_u
    norm_wv = torch.norm(w_v)
    norm_wu = torch.norm(w_u)

    # Variances for a_v and a_u:
    var_av = (norm_wv ** 2) / 2 - (norm_wv ** 2) / (2 * math.pi)
    var_au = (norm_wu ** 2) / 2 - (norm_wu ** 2) / (2 * math.pi)

    # Correlation coefficient rho (with a small epsilon for numerical stability)
    eps = 1e-8
    rho = torch.dot(w_v, w_u) / (norm_wv * norm_wu + eps)
    rho = torch.clamp(rho, -1.0, 1.0)

    # Covariance between a_v and a_u:
    cov_av_au = (norm_wv * norm_wu) / (2 * math.pi) * (
            torch.sqrt(1 - rho ** 2) + rho * (math.pi / 2 + torch.asin(rho)) - 1
    )

    # Final variance of z_t = w_t[0]*a_v + w_t[1]*a_u:
    variance = (w_t[0] ** 2) * var_av + (w_t[1] ** 2) * var_au + 2 * w_t[0] * w_t[1] * cov_av_au
    return variance


def empirical_variance_z_t(w_t: torch.Tensor, w_v: torch.Tensor, w_u: torch.Tensor,
                           n: int, batch_size: int) -> torch.Tensor:
    """
    Compute the empirical variance of z_t by feeding n samples through the network.

    For each sample x ~ N(0, I) (of dimension d), we compute:
       z_v = w_v^T x,   a_v = ReLU(z_v)
       z_u = w_u^T x,   a_u = ReLU(z_u)
       z_t = w_t[0]*a_v + w_t[1]*a_u

    The samples are processed in batches of size `batch_size`.
    Returns the empirical variance of z_t across all n samples.
    """
    d = w_v.size(0)
    z_t_list = []

    # Process in batches:
    for i in range(0, n, batch_size):
        current_batch_size = min(batch_size, n - i)
        # Generate current batch of samples (each of dimension d)
        x = torch.randn(current_batch_size, d)

        # Compute pre-activations and activations for both units
        z_v = x @ w_v  # shape: [current_batch_size]
        z_u = x @ w_u  # shape: [current_batch_size]
        a_v = torch.relu(z_v)
        a_u = torch.relu(z_u)

        # Compute z_t for the batch
        z_t_batch = w_t[0] * a_v + w_t[1] * a_u
        z_t_list.append(z_t_batch)

    # Concatenate all z_t values and compute variance over all samples
    z_t_all = torch.cat(z_t_list, dim=0)
    empirical_var = torch.var(z_t_all, unbiased=False)  # population variance
    return empirical_var


if __name__ == '__main__':
    # For reproducibility, you can set a random seed:
    # torch.manual_seed(42)

    # Define the dimension of the input for w_v and w_u
    d = 3  # Feel free to change this value

    # Randomly generate weight vectors:
    # w_t is a 2-dimensional vector (for the final layer: weights for a_v and a_u)
    w_t = torch.randn(2)
    # w_v and w_u are d-dimensional vectors
    w_v = torch.randn(d)
    w_u = torch.randn(d)

    # Print the randomly generated weight vectors:
    print("w_t:", w_t)
    print("w_v:", w_v)
    print("w_u:", w_u)

    # Define the number of samples and the batch size for the empirical estimation:
    n = 10000  # total number of samples
    batch_size = 256  # process samples in batches

    # Compute the theoretical variance
    theoretical_var = compute_z_t_variance(w_t, w_v, w_u)

    # Compute the empirical variance
    empirical_var = empirical_variance_z_t(w_t, w_v, w_u, n, batch_size)

    # Print the results
    print("\nTheoretical Variance of z_t: {:.6f}".format(theoretical_var.item()))
    print("Empirical Variance of z_t:   {:.6f}".format(empirical_var.item()))
