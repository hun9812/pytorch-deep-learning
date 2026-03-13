import torch

x = torch.tensor([[1, 2], [3, 4]], dtype = torch.float32)
y = torch.ones(2, 2, requires_grad = True) # will track gradients for this tensor

z = x + y
print(z)

loss = (z - 10).sum()
loss.backward()

# loss = (z - 10).sum()
# z = x + y
# dz/dy = 1
# dloss/dz = 1

print(y.grad)
print(f"Shape of y: {y.shape}")
print(f"shape of y.grad: {y.grad.shape}")

# have to reset gradients to zero before backpropagating again
# because by default, gradients are accumulated in PyTorch
# cos of memory efficiency, PyTorch doesn't automatically zero out gradients after backpropagation

if y.grad is not None:
    y.grad.zero_()

loss = (z - 10).sum() * 2
loss.backward()
print(y.grad)


# why use pytorch: GPU acceleration, automatic gradients
