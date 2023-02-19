import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

n = 100
d = 2

# Setup a little regression problem
x = np.random.randn(n, d).astype(np.float32)
noise = 0.5 * np.random.randn(n).astype(np.float32)
# y is proportional to x but there is some noise
y = np.sum(x, axis=1) + noise

w = np.random.randn(1, 2).astype(np.float32)
magnitude = np.random.randn(1, 2).astype(np.float32)


# Now bring over to the Pytorch world
x = torch.from_numpy(x)
y = torch.from_numpy(y.reshape(n, 1))
w = nn.Parameter(torch.from_numpy(w))
magnitude = nn.Parameter(torch.from_numpy(magnitude))
print(w)
print(magnitude)


W = w * magnitude



# Define the linear layer
model = torch.nn.Linear(2, 1)
# Override the weight tensor
as_param = nn.Parameter(W)
model.weight = as_param
print(model.weight)



optimizer = torch.optim.SGD([w, magnitude], lr=10.0, momentum = 0.9)
for i in range(10):
    y_hat = model(x)
    mse = F.mse_loss(y_hat, y)
    print(mse)
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

print(w)
print(magnitude)
print(model.weight)



