import torch
import math


# ------------------------------
# Helper functions for standard normal PDF and CDF
# ------------------------------
def phi(x):
    """Standard normal PDF."""
    return torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def Phi(x):
    """Standard normal CDF."""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


# ------------------------------
# First-layer moments: a = [a_v, a_u] = ReLU([z_v, z_u])
# ------------------------------
def compute_first_layer_moments(w_v: torch.Tensor, w_u: torch.Tensor):
    """
    Compute the mean and covariance of the first-layer activations a_v and a_u,
    when z_v = w_v^T x and z_u = w_u^T x with x ~ N(0, I).
    For a zero-mean Gaussian input, one has:
       E[ReLU(z)] = ||w||/sqrt(2pi)
       E[ReLU(z)^2] = ||w||^2/2
       Var(ReLU(z)) = ||w||^2/2 - ||w||^2/(2pi)
    and for the cross moment one can show:
       E[a_v a_u] = (||w_v|| ||w_u||/(2pi)) * ( sqrt(1-rho^2) + rho*(pi/2 + arcsin(rho)) )
    with rho = (w_v^T w_u)/(||w_v|| ||w_u||).
    """
    norm_wv = torch.norm(w_v)
    norm_wu = torch.norm(w_u)
    # Means:
    m_av = norm_wv / math.sqrt(2 * math.pi)
    m_au = norm_wu / math.sqrt(2 * math.pi)
    m_a = torch.tensor([m_av, m_au])
    # Variances:
    var_av = norm_wv ** 2 / 2 - norm_wv ** 2 / (2 * math.pi)
    var_au = norm_wu ** 2 / 2 - norm_wu ** 2 / (2 * math.pi)
    # Correlation of z_v and z_u:
    eps = 1e-8
    rho = torch.dot(w_v, w_u) / (norm_wv * norm_wu + eps)
    rho = torch.clamp(rho, -1.0, 1.0)
    # Cross moment E[a_v a_u]:
    cross = (norm_wv * norm_wu) / (2 * math.pi) * (torch.sqrt(1 - rho ** 2) + rho * (math.pi / 2 + torch.asin(rho)))
    # Note: Since E[a_v]E[a_u] = (norm_wv*norm_wu)/(2*pi), the covariance is:
    cov_av_au = cross - (norm_wv * norm_wu) / (2 * math.pi)
    # Assemble covariance matrix:
    Sigma_a = torch.tensor([[var_av, cov_av_au],
                            [cov_av_au, var_au]])
    return m_a, Sigma_a


# ------------------------------
# Second-layer pre-activation moments:
# z_s = w_s^T a,   z_q = w_q^T a.
# ------------------------------
def compute_second_layer_params(m_a: torch.Tensor, Sigma_a: torch.Tensor,
                                w_s: torch.Tensor, w_q: torch.Tensor):
    """
    Compute the means, variances, and correlation for z_s and z_q.
    """
    mu_s = torch.dot(w_s, m_a)
    mu_q = torch.dot(w_q, m_a)
    sigma_s2 = (w_s.unsqueeze(0) @ Sigma_a @ w_s.unsqueeze(1)).squeeze()  # scalar
    sigma_q2 = (w_q.unsqueeze(0) @ Sigma_a @ w_q.unsqueeze(1)).squeeze()
    sigma_s = torch.sqrt(sigma_s2)
    sigma_q = torch.sqrt(sigma_q2)
    # Covariance between z_s and z_q:
    cov_zsq = (w_s.unsqueeze(0) @ Sigma_a @ w_q.unsqueeze(1)).squeeze()
    rho_z = cov_zsq / (sigma_s * sigma_q + 1e-8)
    rho_z = torch.clamp(rho_z, -1.0, 1.0)
    return mu_s, sigma_s, mu_q, sigma_q, cov_zsq, rho_z


# ------------------------------
# ReLU moments for a Gaussian with nonzero mean.
# ------------------------------
def relu_moments(mu, sigma):
    """
    For Y ~ N(mu, sigma^2), computes:
      E[ReLU(Y)] = mu * Phi(mu/sigma) + sigma * phi(mu/sigma)
      E[ReLU(Y)^2] = (mu^2+sigma^2) * Phi(mu/sigma) + mu*sigma*phi(mu/sigma)
    """
    # Avoid division by zero if sigma is very small:
    ratio = mu / (sigma + 1e-8)
    mean_relu = mu * Phi(ratio) + sigma * phi(ratio)
    second_moment = (mu ** 2 + sigma ** 2) * Phi(ratio) + mu * sigma * phi(ratio)
    return mean_relu, second_moment


# ------------------------------
# Approximate cross moment E[ReLU(z_s) ReLU(z_q)]
# using Monte Carlo integration.
# ------------------------------
def relu_cross_moment(mu_s, sigma_s, mu_q, sigma_q, rho_z, n_samples=100000):
    """
    Generate n_samples from the bivariate Gaussian:
       [z_s, z_q] ~ N( [mu_s, mu_q], [[sigma_s^2, rho_z*sigma_s*sigma_q],
                                       [rho_z*sigma_s*sigma_q, sigma_q^2]] )
    Then compute E[ReLU(z_s)*ReLU(z_q)].
    """
    mean = torch.tensor([mu_s, mu_q])
    cov = torch.tensor([[sigma_s ** 2, rho_z * sigma_s * sigma_q],
                        [rho_z * sigma_s * sigma_q, sigma_q ** 2]])
    # Create multivariate normal distribution:
    m = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    samples = m.sample((n_samples,))  # shape: [n_samples, 2]
    relu_samples = torch.relu(samples)
    return relu_samples[:, 0].mean() * 0.0 + (relu_samples[:, 0] * relu_samples[:, 1]).mean()
    # (Note: multiplying by 0.0 above is just to ensure a tensor output)


# ------------------------------
# Compute analytical (moment–propagation) variance of z_t for the 2–layer network.
# ------------------------------
def compute_z_t_variance_2layer(w_t: torch.Tensor,
                                w_v: torch.Tensor, w_u: torch.Tensor,
                                w_s: torch.Tensor, w_q: torch.Tensor,
                                n_cross: int = 100000) -> torch.Tensor:
    """
    Computes the variance of z_t = w_t[0]*c_s + w_t[1]*c_q,
    where c_s = ReLU(z_s) and c_q = ReLU(z_q) are the activations of the second hidden layer.
    The function propagates moments from the input through the two hidden layers.
    """
    # --- First layer ---
    m_a, Sigma_a = compute_first_layer_moments(w_v, w_u)

    # --- Second layer pre-activations: z_s and z_q ---
    mu_s, sigma_s, mu_q, sigma_q, _, rho_z = compute_second_layer_params(m_a, Sigma_a, w_s, w_q)

    # --- Second layer activations: c_s = ReLU(z_s), c_q = ReLU(z_q) ---
    m_cs, E_cs2 = relu_moments(mu_s, sigma_s)
    m_cq, E_cq2 = relu_moments(mu_q, sigma_q)
    var_cs = E_cs2 - m_cs ** 2
    var_cq = E_cq2 - m_cq ** 2

    # --- Cross moment for second layer ---
    E_cs_cq = relu_cross_moment(mu_s, sigma_s, mu_q, sigma_q, rho_z, n_samples=n_cross)
    cov_cs_cq = E_cs_cq - m_cs * m_cq

    # --- Final layer: z_t = w_t[0]*c_s + w_t[1]*c_q ---
    var_z_t = (w_t[0] ** 2) * var_cs + (w_t[1] ** 2) * var_cq + 2 * w_t[0] * w_t[1] * cov_cs_cq
    return var_z_t


# ------------------------------
# Empirical variance via forward pass through the network.
# ------------------------------
def empirical_variance_z_t_2layer(w_t: torch.Tensor,
                                  w_v: torch.Tensor, w_u: torch.Tensor,
                                  w_s: torch.Tensor, w_q: torch.Tensor,
                                  n: int, batch_size: int) -> torch.Tensor:
    """
    Feed n samples through the two-hidden-layer network and compute the empirical variance of z_t.
    """
    d = w_v.size(0)  # dimension of input x
    z_t_list = []
    for i in range(0, n, batch_size):
        current_bs = min(batch_size, n - i)
        # Input samples: x ~ N(0,I)
        x = torch.randn(current_bs, d)
        # First layer: z_v, z_u and activations a_v, a_u
        z_v = x @ w_v  # shape: [current_bs]
        z_u = x @ w_u  # shape: [current_bs]
        a_v = torch.relu(z_v)
        a_u = torch.relu(z_u)
        a = torch.stack([a_v, a_u], dim=1)  # shape: [current_bs, 2]

        # Second layer: z_s, z_q and activations c_s, c_q
        z_s = a @ w_s  # shape: [current_bs]
        z_q = a @ w_q  # shape: [current_bs]
        c_s = torch.relu(z_s)
        c_q = torch.relu(z_q)

        # Final output: z_t = w_t[0]*c_s + w_t[1]*c_q
        z_t_batch = w_t[0] * c_s + w_t[1] * c_q
        z_t_list.append(z_t_batch)

    z_t_all = torch.cat(z_t_list, dim=0)
    empirical_var = torch.var(z_t_all, unbiased=False)  # use population variance
    return empirical_var


# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    # torch.manual_seed(42)

    # Define dimensions:
    d = 3  # dimension of input (and thus of w_v and w_u)

    # Randomly generate weight vectors.
    # For the first layer:
    w_v = torch.randn(d)
    w_u = torch.randn(d)
    # For the second layer, the input dimension is 2 (from a = [a_v, a_u])
    w_s = torch.randn(2)
    w_q = torch.randn(2)
    # For the final layer, the input dimension is 2 (from c = [c_s, c_q])
    w_t = torch.randn(2)

    print("Random weights:")
    print("w_v:", w_v)
    print("w_u:", w_u)
    print("w_s:", w_s)
    print("w_q:", w_q)
    print("w_t:", w_t)

    # Set number of samples and batch size for empirical estimation.
    n_samples = 10000
    batch_size = 256

    # Compute analytical variance via moment propagation.
    analytical_var = compute_z_t_variance_2layer(w_t, w_v, w_u, w_s, w_q, n_cross=100000)

    # Compute empirical variance by forward passing samples.
    empirical_var = empirical_variance_z_t_2layer(w_t, w_v, w_u, w_s, w_q, n_samples, batch_size)

    print("\nAnalytical (moment-propagation) variance of z_t: {:.6f}".format(analytical_var.item()))
    print("Empirical variance of z_t:                      {:.6f}".format(empirical_var.item()))
