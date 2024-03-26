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
from torch.nn.utils.rnn import pad_sequence
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


spider_url = 'https://drive.google.com/u/0/uc?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download' # https://yale-lily.github.io/spider
output = 'spider.zip'
if not os.path.exists(output):
  gdown.download(spider_url, output, quiet=False)





def main():

  # Create the dataset and dataloader
  def construct_input_block_toks(
      col_toks: list[list[str]],
      col_types: list[ColumnType],
      table_toks: list[list[str]],
      ques_toks: list[str],
  ):
      """Construct and return the 3 components of an input block as a tuple of the list of tokens
      An input block has 3 parts:
        The col name part: `CLS col_type ...col_name CLS col_type2 ...Col_name2`
        The table name part: `CLS ...tbl_name CLS ...tbl_name2`
        The question part: ` CLS ...question`

      Parameters:
        col_toks: nested list of all tokens across all column names, each element is a list of tokens that belongs in one column name
        tbl_toks: nested list of all tokens across all table names, each element is a list of tokens that belongs in one table name

      Returns:
        a 3-elemment tuple containing (col_name_part, tbl_name_part, ques_part) w/ each part being the list of tokens including the 'CLS' token
      """
      # List of column types should match length with list of columns
      assert len(col_toks) == len(col_types)
      # Construct the col name part, thus producing something like:
      # "CLS number builing size CLS text building name..."
      col_name_part: list[str] = []
      for col_name_toks, col_type in zip(col_toks, col_types):
          col_name_part.extend(["CLS", col_type] + col_name_toks)
      # Construct the col name part, thus producing something like:
      # "CLS building information CLS city information ...etc"
      tbl_name_part: list[str] = []
      for tbl_name_toks in table_toks:
          tbl_name_part.extend(["CLS"] + tbl_name_toks)
      # Construct the question part, thus producing something like:
      # "CLS what is the highest building in Chicago"
      question_part: list[str] = ques_toks
      return (col_name_part, tbl_name_part, question_part)

  DatasetItem = tuple[list[str], list[str], list[str], list[str]]
  class MyDataset(Dataset[DatasetItem]):
      """Each element in this dataset is a 4-element tuple
      (col_name_toks, table_name_toks, question_toks, target_toks)

      where `col_name_toks`, `table_name_toks`, `question_toks` has been preprocessed to add "CLS" separator token
      and `target_toks` is also processed to include <Start> and <End> tokens
      """
      def __init__(self, all_items: list[SpiderItem]):
          self.items = all_items

      def __len__(self):
          return len(self.items)

      def __getitem__(self, idx: int) -> DatasetItem:
          item = self.items[idx]
          columns_tokens = [c.name_toks for c in item.db.columns[1:]] # Skip the 0th-index "*" column
          columns_types: list[ColumnType] = [c.col_type for c in item.db.columns[1:]]
          tables_tokens = [t.name_toks for t in item.db.tables]
          question_tokens = item.qa_pair.question_toks
          target_tokens = item.qa_pair.query_toks + ["<END>"]

          return construct_input_block_toks(columns_tokens, columns_types, tables_tokens, question_tokens) + (target_tokens,)


  PAD_TOKEN_ID = src_tok_to_idx[PAD_TOKEN]

  train_dataset = MyDataset(train_items)
  val_dataset = MyDataset(val_items)
  # Find the max_length of each "category".
  # Category here correspondings to the 4 categories (col_name_toks, table_name_toks, question_toks, target_toks)
  max_lengths = [max(len(category) for category in item) for item in zip(*train_dataset)]
  max_len_col_name_part = max_lengths[0]
  max_len_tbl_name_part = max_lengths[1]
  max_len_ques_part = max_lengths[2]
  max_len_target = max_lengths[3]

  BATCH_SIZE = 70
  ModelInput = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

  def pad_collate(batch: list[tuple[list[str], list[str], list[str], list[str]]]) -> ModelInput:
      """The collate_fn to pass to Pytorch Dataloader.
      Pad each element in the dataset so they all have the same size"""
      batched_col_name_toks, batched_tbl_name_toks, batched_ques_toks, batched_target_toks = zip(*batch)

      batched_colname_tokIds = [torch.tensor(toks_encode(col_name_toks, mode="source"), device=DEVICE) for col_name_toks in batched_col_name_toks]
      # Now pad this batch so that every item of the batch is same size as a local longest item
      # From the doc: `pad_sequence` stacks a list of Tensors along a new dimension, and pads them to equal length.
      batched_colname_tokIds = pad_sequence(batched_colname_tokIds, batch_first=True, padding_value=PAD_TOKEN_ID)
      # Now pad this batch again so that every item of the batch is same size of the global longest item (max_len_col_name_part)
      batched_colname_tokIds= torch.nn.functional.pad(
          batched_colname_tokIds, (0, max_len_col_name_part - batched_colname_tokIds.size(1)), value=PAD_TOKEN_ID)
      batched_colname_tokIds = batched_colname_tokIds # (B, T)

      # After padding, each tensor should be dimension (B, T)
      # With B being the batch dimension (since we use batch_first=True) and T is max_len_col_name_part
      assert batched_colname_tokIds.shape == (BATCH_SIZE, max_len_col_name_part), f"Expected {BATCH_SIZE, max_len_col_name_part}, got {batched_colname_tokIds.shape}"

      # Do the same process for table name toks, question toks, and target toks
      batched_tblname_tokIds = [torch.tensor(toks_encode(tbl_name_toks, mode="source"), device=DEVICE) for tbl_name_toks in batched_tbl_name_toks]
      batched_tblname_tokIds = pad_sequence(batched_tblname_tokIds, batch_first=True, padding_value=PAD_TOKEN_ID)
      batched_tblname_tokIds= torch.nn.functional.pad(
          batched_tblname_tokIds, (0, max_len_tbl_name_part - batched_tblname_tokIds.size(1)), value=PAD_TOKEN_ID)
      batched_tblname_tokIds = batched_tblname_tokIds
      assert batched_tblname_tokIds.shape == (BATCH_SIZE, max_len_tbl_name_part)

      batched_ques_tokIds = [torch.tensor(toks_encode(ques_toks, mode="source"), device=DEVICE) for ques_toks in batched_ques_toks]
      batched_ques_tokIds = pad_sequence(batched_ques_tokIds, batch_first=True, padding_value=PAD_TOKEN_ID)
      batched_ques_tokIds = torch.nn.functional.pad(
          batched_ques_tokIds, (0, max_len_ques_part - batched_ques_tokIds.size(1)), value=PAD_TOKEN_ID)
      batched_ques_tokIds = batched_ques_tokIds
      assert batched_ques_tokIds.shape == (BATCH_SIZE, max_len_ques_part)

      batched_tgt_tokIds = [torch.tensor(toks_encode(tgt_toks, mode="target"), device=DEVICE) for tgt_toks in batched_target_toks]
      # Note that for the target, we need use `tgt_tok_to_idx`
      batched_tgt_tokIds = pad_sequence(batched_tgt_tokIds, batch_first=True, padding_value=tgt_tok_to_idx[PAD_TOKEN])
      batched_tgt_tokIds = torch.nn.functional.pad(
          batched_tgt_tokIds, (0, max_len_target - batched_tgt_tokIds.size(1)), value=PAD_TOKEN_ID)
      batched_tgt_tokIds = batched_tgt_tokIds
      assert batched_tgt_tokIds.shape == (BATCH_SIZE, max_len_target)

      return (batched_colname_tokIds, batched_tblname_tokIds, batched_ques_tokIds, batched_tgt_tokIds)

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

  train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate, drop_last=True, num_workers=1)
  val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate, drop_last=True, num_workers=1)

  with open("temptemp1.txt", "w") as f:
    for i in range(len(train_dataset)):
      f.write(f"{train_dataset[i]}\n\n")

  with open("temptemp2.txt", "w") as f:
    for i, batch in enumerate(train_dataloader):
      f.write(f"{i}\n")
      f.write(f"{batch[0]} {batch[1]} {batch[2]} {batch[3]}\n")




