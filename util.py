import torch.nn as nn
from typing import Any

# The original PyTorch sequential cannot handle multiple input, so i wrote MySequential to support that
# Credit: https://stackoverflow.com/questions/61036671/either-too-little-or-too-many-arguments-for-a-nn-sequential
class UnpackedSequential(nn.Sequential):
  def forward(self, input: Any):
    for module in self._modules.values():
      if type(input) == tuple:
        input = module(*input)
      else:
        input = module(input)
    return input
