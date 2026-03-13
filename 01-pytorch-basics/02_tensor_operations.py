import torch

t = torch.arange(12) # [0,1,2,...,11]
print(t.shape)


view_t = t.view(3,4) # reshape t to 3 rows and 4 columns
print(view_t.shape)

# if you use -1, PyTorch will automatically calculate the size of that dimension based on the other dimensions and the total number of elements in the tensor
auto_view_t = t.view(-1, 2, 2)
print(auto_view_t.shape)

# reshape vs view
# reshape can return a view or a copy of the original tensor, depending on the memory layout of the original tensor and the requested shape. If the requested shape is compatible with the original tensor's memory layout, reshape will return a view. Otherwise, it will return a copy.

mat1 = torch.randn(3, 2)
mat2 = torch.randn(2, 4)

print("==================================================================")

res = mat1 @ mat2 # matrix multiplication
print(res.shape)

# broadcasting
a = torch.tensor([[1], [2], [3]]) # shape (3,1)
b = torch.tensor([[10, 20]]) # shape (1,2)
print(a+b)
# broadcasting rules:
# back to front (based on 1)

mat3 = torch.randn(3,4)
mat4 = torch.randn(3)
try:
    print(mat3 + mat4) # error because mat4 has shape (3,) and cannot be broadcasted to (3,4) !! 3,4
except:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

x = torch.tensor([1, 2, 3], device=device)
print(x, x.device)