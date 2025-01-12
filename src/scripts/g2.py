import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 100  # Number of samples
w_range = np.linspace(-1, 1, 500)  # Range for w

# Generate x from standard normal distribution
x = np.random.normal(0, 1, n)

# Create y with 50% 0 and 50% 1
# y = np.zeros(n)
# y[:n//2] = 1
# np.random.shuffle(y)
y = np.random.choice([0, 1], size=n)

# Function to compute f(w)
def f(w, x, y):
    a = 1 / (1 + np.exp(-w * x))
    return np.sum((y - a) * x) / n

# Compute f(w) for each value in w_range
f_values = [f(w, x, y) for w in w_range]

# Plot f(w)
plt.figure(figsize=(8, 6))
plt.plot(w_range, f_values, label='f(w)', color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=1, label='y = 0')
plt.title('Graph of f(w) vs. w')
plt.xlabel('w')
plt.ylabel('f(w)')
plt.legend()
plt.grid()
plt.show()
