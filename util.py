import torch.nn as nn
import torch
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


def compare_models(model_1: nn.Module, model_2: nn.Module):
  models_differ = 0
  for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
    if torch.equal(key_item_1[1], key_item_2[1]):
      pass
    else:
      models_differ += 1
      if (key_item_1[0] == key_item_2[0]):
        print('Mismatch found at', key_item_1[0])
      else:
        raise Exception
  if models_differ == 0:
    print('Models match perfectly! :)')
  return models_differ
