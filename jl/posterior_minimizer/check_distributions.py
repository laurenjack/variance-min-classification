import numpy as np
import matplotlib.pyplot as plt

from jl.posterior_minimizer import dataset_creator as dc

n = 1000000
problem = dc.Gaussian(1, scale_of_mean=0.1)
x, _ = problem.generate_dataset(n, shuffle=False)
class_n = n // 2
x0 = x[0:class_n].numpy()
x1 = x[class_n:].numpy()

print(np.mean(x0))
print(np.mean(x1))

# # Create the histogram
# plt.figure(figsize=(10, 6))
# plt.hist(x0, bins=30, alpha=0.5, label='Distribution 1')
# plt.hist(x1, bins=30, alpha=0.5, label='Distribution 2')
#
# # Add labels and title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Comparison of Two Distributions')
# plt.legend()

# Show the plot
plt.show()

