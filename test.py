import torch
a = torch.randn(5, 4)
print(a)
b = torch.argmax(a, dim=1)
print(b)
for num in b:
  print(num)