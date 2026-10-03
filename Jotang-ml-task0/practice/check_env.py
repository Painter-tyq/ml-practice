import sys

import torch

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
c = a + b
d = a @ b

print("tensor a:\n", a)
print("tensor b:\n", b)
print("a + b:\n", c)
print("a @ b:\n", d)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU (expected if you have no NVIDIA GPU).")
