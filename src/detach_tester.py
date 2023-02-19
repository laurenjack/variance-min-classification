import torch

# Create a tensor with requires_grad set to True
x = torch.tensor([1.0], requires_grad=True)

# Compute a loss using x
y = x**2
loss = y.mean()

# # Compute the gradients
# loss.backward()
#
# # Print the gradient of x
# print(x.grad) # tensor([2.])

# Detach x
x = x.detach()

# Compute the gradients again
loss.backward()

# Print the gradient of x
print(x.grad) # None