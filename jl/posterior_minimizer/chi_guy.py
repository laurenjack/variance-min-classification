import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import gaussian_kde


def generate_generalized_chi_square_samples(c_array, num_samples=1000000):
    """
    Generate samples from a generalized chi-square distribution.

    :param c_array: NumPy array of scaling constants (c_i)
    :param num_samples: Number of samples to generate
    :return: Array of samples
    """
    d = c_array.shape[0]
    z = np.random.normal(0, 1, (d, num_samples))
    return np.sum(c_array[:, np.newaxis] * z ** 2, axis=0)


def plot_comparative_chi_square(c_array, num_samples=1000000):
    """
    Plot both the generalized chi-square distribution (using sampling)
    and the scaled standard chi-square distribution on the same set of axes.

    :param c_array: NumPy array of scaling constants (c_i)
    :param num_samples: Number of samples to generate for the generalized distribution
    """
    # d = c_array.shape[0]  # Number of chi-square distributions (degrees of freedom)
    # scale_factor = np.mean(c_array)  # Mean of scaling constants
    mean = np.sum(c_array)
    variance = 2 * np.sum(c_array ** 2)
    scale_factor = variance / (2 * mean)
    d = 2 * mean ** 2 / variance

    # Generate samples for the generalized chi-square distribution
    gen_samples = generate_generalized_chi_square_samples(c_array, num_samples)

    # Estimate the PDF using kernel density estimation
    kde = gaussian_kde(gen_samples)

    # Generate x values
    x_max = max(np.percentile(gen_samples, 99.9), scale_factor * chi2.ppf(0.999, d))
    x = np.linspace(0, x_max, 1000)

    # Calculate the probability density function for scaled standard chi-square
    y_std = chi2.pdf(x / scale_factor, d) / scale_factor

    # Calculate the estimated PDF for generalized chi-square
    y_gen = kde(x)

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Format array for display (limit to 6 elements if array is long)
    if d > 6:
        display_array = np.array2string(c_array[:6], precision=2, separator=', ') + '...'
    else:
        display_array = np.array2string(c_array, precision=2, separator=', ')

    plt.plot(x, y_gen, 'b-', lw=2, label=f'Generalized Chi-Square\nc = {display_array}')
    plt.plot(x, y_std, 'r--', lw=2,
             label=f'Scaled Standard Chi-Square\ndf = {d}, scale = {scale_factor:.2f}')

    plt.title(f'Comparison of Generalized and Scaled Standard Chi-Square Distributions\n'
              f'd = {d}, mean(c) = {scale_factor:.2f}')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Print some statistics for verification
    print(f"\nMean of generalized chi-square (empirical): {np.mean(gen_samples):.2f}")
    print(f"Mean of scaled standard chi-square (theoretical): {scale_factor * d:.2f}")
    print(f"Variance of generalized chi-square (empirical): {np.var(gen_samples):.2f}")
    print(f"Variance of scaled standard chi-square (theoretical): {2 * scale_factor ** 2 * d:.2f}")


# Example usage
# c_array = 2 * np.ones(10)
c_array = np.random.randn(10) ** 2  # Generate random normal variables as scaling constants
plot_comparative_chi_square(c_array)