import spacy
import preprocess
import torch
from torch import nn
from typing import Union
import gdown
import os
from preprocess import ColumnType, SpiderItem
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# Prepare Stanford's Stanza to tokenize the columns and tables names
# stanza.download('en')
# stanza_nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='en')

# Prepare spacy to tokenize and lemmatize the table names and columns names
spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "parser"])
# Fix the "Spacy breaks up 'id' into 'i' and 'd' issue" https://github.com/explosion/spaCy/discussions/10570
spacy_nlp.tokenizer.rules = { key: value for key, value in spacy_nlp.tokenizer.rules.items() if key != "id" }

device = "cpu"
if torch.backends.mps.is_available():
  device = "mps" # Apple Metal Performance Shader (M1 chip)
if torch.cuda.is_available():
  device = "cuda"
DEVICE = device


def main():

  BATCH_SIZE = 70

  #  [markdown]
  # # Self-attention

  #  [markdown]
  # ## The Masked Attention Math trick
  # 
  # The Triangular mask technique

  # 
  # The Attention Mask Math trick - The Triangular mask technique
  T = 300

  # A triangular mask is a tensor like this: dimension (block_size, block_size)
  # tensor([[1., 0., 0.,  ..., 0., 0., 0.],
  #        [1., 1., 0.,  ..., 0., 0., 0.],
  #        [1., 1., 1.,  ..., 0., 0., 0.],
  #        ...,
  #        [1., 1., 1.,  ..., 1., 0., 0.],
  #        [1., 1., 1.,  ..., 1., 1., 0.],
  #        [1., 1., 1.,  ..., 1., 1., 1.]])
  tril_mask = torch.tril(torch.ones(T, T))

  # Initial weight represents the initial *affinity* between each pair of tokens in the sequence
  # (right now we initialize them all with 0s, but they can be diff numbers coming from somewhere like the encoder for example)
  initial_weight = torch.zeros((T, T))

  # For the purpose of decoder calculation, we want prediction of the next step to only depend on values of previous steps
  # Therefore, we'll apply a *triangular mask* to the weight vector
  # By the structure of the triangular mask, we can see that, after masking,
  # we essentially "disable"/"mask away" future tokens since they will be multiplying with 0
  # since only positions corresponding to 1s will be retained. Positions with 0s is replaced with -inf
  weight = initial_weight.masked_fill(tril_mask == 0, float("-inf"))
  # A softmax layer convert these *affinity scores* to a weight number between 0 and 1
  weight = F.softmax(weight, dim=-1)
  # weight = tensor([[1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
  #                  [0.5000, 0.5000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
  #                  [0.3333, 0.3333, 0.3333,  ..., 0.0000, 0.0000, 0.0000],
  #                  ...,
  #                  [0.0034, 0.0034, 0.0034,  ..., 0.0034, 0.0000, 0.0000],
  #                  [0.0033, 0.0033, 0.0033,  ..., 0.0033, 0.0033, 0.0000],
  #                  [0.0033, 0.0033, 0.0033,  ..., 0.0033, 0.0033, 0.0033]])
  # --> See how the weight vector showcase how to calculate a prediction based on the weighted sum of previous values only
  # For example:
  #   2nd value = 1.00 * 1st val
  #   3rd prediction = 0.5 * 1st val + 0.5 * 2nd val
  #   4th prediction = 0.33 * 1st value + 0.33 * 2nd val + 0.33 * 3rd val

  #  [markdown]
  # ## Self-attention block

  # 

if __name__ == '__main__':
  pass




