import torch


def uniform_one_true_dim(n_per_class, d, p):
    """
    Generate a binary classification problem of dimensionality d, that has a single dimension that is relevant (at
    dimension index 0). The remaining dimensions are just random noise. The problem will be perfectly class balanced, p
    is the probability that the instances of class 1 are greater than zero along the relevant dimension, less than zero
    for class 0
    """
    y = torch.cat([torch.ones(n_per_class), torch.zeros(n_per_class)], dim=0)
    n = y.shape[0]
    r = torch.bernoulli(torch.full((n_per_class,), p))
    b = torch.bernoulli(torch.full((n_per_class,), 1 - p))
    is_positive = torch.cat([r, b], dim=0)
    x0 = generate_uniform(n, 1, 0, 2)
    sign_flip = is_positive * 2 - 1
    x0 *= sign_flip.unsqueeze(1)
    shuffled = torch.randperm(n)
    x0 = x0[shuffled]
    y = y[shuffled]
    # Now generate the spurious dimensions
    if d > 1:
        x = generate_uniform(n, d - 1, -2, 2)
        x = torch.cat([x0, x], dim=1)
    else:
        x = x0

    return x, y


def generate_uniform(n, d, a, b):
    return a + (b - a) * torch.rand(n, d)

