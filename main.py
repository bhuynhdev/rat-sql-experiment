import spacy
import torch
from torch import nn
import json
from dataclasses import dataclass, field
from typing import Literal, Any, Union
import gdown
import os
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


# Start by loading in the databases, tables, and columns schemas

with open("spider/tables.json") as f:
  raw_schema_data = json.load(f)

ColumnType = Literal["time", "text", "number", "boolean", "others"]


@dataclass
class Column:
  db_id: str  # Which database this col belongs to
  table_index: int  # Which specific table in the db this col belongs to
  col_index: int  # Which specific column in this table
  name: str
  name_toks: list[str]  # Tokenized name
  orig_name: str
  col_type: ColumnType


@dataclass
class Table:
  db_id: str  # Which database this table belongs to
  table_index: int  # Which specific table in this database
  name: str
  name_toks: list[str]  # Tokenized name
  orig_name: str
  columns: list[Column] = field(default_factory=list)


@dataclass
class Database:
  db_id: str
  raw_schema: Any  # The original raw schema from json file
  tables: list[Table] = field(default_factory=list)
  columns: list[Column] = field(default_factory=list)


schema_lookup: dict[str, Database] = {}  # Map the db_id to that database's schema

raw_schema_data[0].keys()
# ['column_names', 'column_names_original', 'column_types', 'db_id',
# 'foreign_keys', 'primary_keys', 'table_names', 'table_names_original']


def lemmatized_tokenizer(s: str):
  doc = spacy_nlp(s)
  lemmatized_tokens = [token.lemma_.lower() for token in doc]
  # doc = stanza_nlp(s)
  # lemmatized_tokens = [word.lemma.lower() for sent in doc.sentences  for word in sent.words]
  return lemmatized_tokens


for i, db_info in enumerate(raw_schema_data):
  db_id = db_info["db_id"]
  db = Database(db_id, raw_schema=db_info)

  # Iterate the tables of this database
  for table_index, (tbl_name, tbl_orig_name) in enumerate(
    zip(db_info["table_names"], db_info["table_names_original"])
  ):
    new_table = Table(
      db_id,
      table_index=table_index,
      name=tbl_name,
      name_toks=lemmatized_tokenizer(tbl_name),
      orig_name=tbl_orig_name,
    )
    db.tables.append(new_table)

  # Iterate the columns of this database
  for col_index, ((table_idx, col_name), (_, col_orig_name), col_type) in enumerate(
    zip(
        db_info["column_names"],
        db_info["column_names_original"],
        db_info["column_types"],
    )
  ):
    new_col = Column(
        db_id=db_id,
        col_index=col_index,
        col_type=col_type,
        table_index=table_idx,
        name=col_name,
        name_toks=lemmatized_tokenizer(col_name),
        orig_name=col_orig_name,
    )
    db.columns.append(new_col)
    table_idx: int = table_idx
    if table_idx >= 0:  # Skip the -1 index table
        db.tables[table_idx].columns.append(new_col)

  schema_lookup[db_id] = db

all_columns = [col for s in schema_lookup.values() for col in s.columns]
all_tables = [tbl for s in schema_lookup.values() for tbl in s.tables]

# Load the questions
with open("spider/train_spider.json") as f:
  raw_train_data = json.load(f)

with open("spider/dev.json") as f:
  raw_val_data = json.load(f)

raw_train_data[0].keys()
# ['db_id', 'query', 'query_toks', 'query_toks_no_value', 'question', 'question_toks', 'sql']

@dataclass
class QAPair: # Class to encapsulate each NL query with corresponding SQL answer
  db_id: str
  question_toks: list[str]
  question: str
  query_toks: list[str]
  query: str
  sql_tree: Any # Raw sql tree as a nested dict from the raw JSON

@dataclass
class SpiderItem:
  """An item to feed into the neural network. Basically encapsulation of a training example"""
  db_id: str
  qa_pair: QAPair
  db: Database # Information about the database used in the qa


train_qas: list[QAPair] = []
val_qas: list[QAPair] = []
train_items: list[SpiderItem] = []
val_items: list[SpiderItem] = []

for i, qa_info in enumerate(raw_train_data):
  db_id = qa_info["db_id"]
  new_qa = QAPair(
    db_id=db_id,
    question_toks=qa_info["question_toks"],
    question=qa_info["question"],
    query_toks=qa_info["query_toks"],
    query=qa_info["query"],
    sql_tree=qa_info["sql"]
  )
  new_item = SpiderItem(
    db_id=db_id,
    qa_pair=new_qa,
    db=schema_lookup[db_id]
  )
  train_qas.append(new_qa)
  train_items.append(new_item)

for i, qa_info in enumerate(raw_val_data):
  db_id = qa_info["db_id"]
  new_qa = QAPair(
    db_id=db_id,
    question_toks=qa_info["question_toks"],
    question=qa_info["question"],
    query_toks=qa_info["query_toks"],
    query=qa_info["query"],
    sql_tree=qa_info["sql"]
  )
  new_item = SpiderItem(
    db_id=db_id,
    qa_pair=new_qa,
    db=schema_lookup[db_id]
  )
  val_qas.append(new_qa)
  val_items.append(new_item)


# 
col_name_vocab: set[str] = set()
tbl_name_vocab: set[str] = set()
col_types: set[str] = set()

# Build col names and tbl names vocab
for col in all_columns:
  col_name_vocab.update(col.name_toks)
  col_types.add(col.col_type)

for tbl in all_tables:
  tbl_name_vocab.update(tbl.name_toks)

target_vocab: set[str] = set() # Vocab of the SQL target
question_vocab: set[str] = set() # Vocab of the Natural Language questions

for qa_pair in train_qas:
  question_vocab.update(qa_pair.question_toks)
  target_vocab.update(qa_pair.query_toks)

for qa_pair in val_qas:
  question_vocab.update(qa_pair.question_toks)
  target_vocab.update(qa_pair.query_toks)

# Padding token used to pad shorter inputs so every sequence of inputs to the encoder
# is of same length (which would be the max length)
PAD_TOKEN = "<PAD>"

src_vocab: set[str] = set() # Source vocab used for inputs to the encoder, which includes questions & columns & tables names
src_vocab.add(PAD_TOKEN)
# Add the "CLS" token as a separator between each part in the input block (i.e. separator between each column name, table name, etc.)
src_vocab.add("CLS")
src_vocab.update(col_name_vocab)
src_vocab.update(col_types)
src_vocab.update(tbl_name_vocab)
src_vocab.update(question_vocab)

tgt_vocab: set[str] = set() # Target vocab used for decoder output; which is the set of SQL tokens plus the Start and End tokens
tgt_vocab.add("<START>")
tgt_vocab.add("<END>")
tgt_vocab.add(PAD_TOKEN)
tgt_vocab.update(target_vocab)

# Token to id: Assign a number to each token in the vocab
# Id to token: Retrieve the token give the index
src_tok_to_idx = { tok: i for i, tok in enumerate(sorted(list(src_vocab))) }
src_idx_to_tok = { i: tok for i, tok in enumerate(sorted(list(src_vocab))) }
tgt_tok_to_idx = { tok: i for i, tok in enumerate(sorted(list(tgt_vocab))) }
tgt_idx_to_tok = { i: tok for i, tok in enumerate(sorted(list(tgt_vocab))) }

def toks_encode(tokens_list: list[str], mode: Literal["source", "target"]):
  """Given a list of tokens, 'encode' them to return a list of token_ids
  Parameters:
    mode: Whether to use the source's or target's tok_to_idx lookup table
  """
  if mode == "source":
    return [src_tok_to_idx[token] for token in tokens_list]
  return [tgt_tok_to_idx[token] for token in tokens_list]

def toks_decode(ids_list: list[int], mode: Literal["source", "target"]):
  """Given a list of token_ids, 'decode' them to return the list of corresponding token
  Parameters:
    mode: Whether to use the source's or target's idx_to_tok lookup table
  """
  if mode == "source":
    return [src_idx_to_tok[token_id] for token_id in ids_list]
  return [tgt_idx_to_tok[token_id] for token_id in ids_list]


# 
# Create the dataset and dataloader
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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


train_dataset = MyDataset(train_items)
val_dataset = MyDataset(val_items)

PAD_TOKEN_ID = src_tok_to_idx[PAD_TOKEN]

# Find the max_length of each "category".
# Category here correspondings to the 4 categories (col_name_toks, table_name_toks, question_toks, target_toks)
max_lengths = [max(len(category) for category in item) for item in zip(*train_dataset)]
max_len_col_name_part = max_lengths[0]
max_len_tbl_name_part = max_lengths[1]
max_len_ques_part = max_lengths[2]
max_len_target = max_lengths[3]

BATCH_SIZE = 64
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

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate, drop_last=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate, drop_last=True, num_workers=2)

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
from math import sqrt

BLOCK_SIZE = max_len_col_name_part + max_len_tbl_name_part + max_len_ques_part
DROP_OUT = 0.2

class SingleHeadAttention(nn.Module):
  def __init__(self, emb_size: int, head_size: int):
    super().__init__()
    self.emb_size = emb_size
    self.head_size = head_size
    # Each head has its own W_Q, W_K, and W_V matrixes for transform the each tok emd to its corresponding q, k, v vectors
    self.query_matrix = nn.Linear(emb_size, head_size, bias=False)
    self.key_matrix = nn.Linear(emb_size, head_size, bias=False)
    self.value_matrix = nn.Linear(emb_size, head_size, bias=False)
    # tril_mask is a static non-learned parameter, so need to use `register_buffer`
    self.register_buffer("tril_mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    self.dropout = nn.Dropout(DROP_OUT)

  def forward(self, x: torch.Tensor, should_mask:bool):
    """
    Parameters:
      x: the input embedding (after summing token_emb and position_emb) --> dimension (B, T, E)
      should_mask: should this Attention block use masked attention (decoder should use mask, encoder shouldn't)
    """
    query = self.query_matrix(x) # (B, T, D) with D = head_size
    key = self.key_matrix(x) # (B, T, D)
    value = self.value_matrix(x) # (B, T, D)
    # `q @ k.T` will produce the itcanitial affinity matrix, basically how strong each query relates to each key
    # dimension: (B, T, D) @ (B, D, T) = (B, T, T)
    # Note The original "Attention is all you need paper" also scales down the affinity scores by multiplying `sqrt(head_size)`
    affinity = (query @ key.transpose(-2, -1)) * (sqrt(self.head_size))  # tranpose(-2, -1) avoid transposing the Batch dimension
    if should_mask:
      affinity = affinity.masked_fill(self.tril_mask == 0, float("-inf"))
    weight = F.softmax(affinity, dim=-1) # (B, T, T)
    weight = self.dropout(weight)
    # The output is the embeddings where each token's embedding have been tweaked
    # to also include information about other related tokens
    out = weight @ value # (B, T, D)
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, emb_size: int, num_head: int):
    super().__init__()
    self.emb_size = emb_size
    # Each head size is emb_size / num_head so that at the end, when we concat all vectors from each head, we still get a vector of emb_size
    self.heads = nn.ModuleList([SingleHeadAttention(emb_size, emb_size // num_head) for _ in range(num_head)])
    self.dropout = nn.Dropout(DROP_OUT)
  
  def forward(self, x: torch.Tensor, should_mask: bool=False):
    out = torch.cat([sa(x, should_mask=should_mask) for sa in self.heads], dim=-1)
    out = self.dropout(out)
    return out
  
class PositionWiseFeedForward(nn.Module):
  """After self-attention block is a Feed forward neural net (section 3.3)"""
  def __init__(self, emb_size: int):
    super().__init__()
    self.feed_forward = nn.Sequential(
      nn.Linear(emb_size, 4 * emb_size),
      nn.ReLU(),
      nn.Linear(4 * emb_size, emb_size),
      nn.Dropout(DROP_OUT)
    )

  def forward(self, x: torch.Tensor):
    return self.feed_forward(x)


#  [markdown]
# # Encoder block

# 
class EncoderBlock(nn.Module):
  """A Transformer Encoder block: A self-attention followed by feedforward net"""
  def __init__(self, emb_size: int, num_attention_heads: int):
    super().__init__()
    self.self_attention = MultiHeadAttention(emb_size, num_attention_heads)
    self.feed_forward = PositionWiseFeedForward(emb_size)
    self.layer_norm1 = nn.LayerNorm(emb_size) # Layer norm for the self-attention sublayer
    self.layer_norm2 = nn.LayerNorm(emb_size) # Layer norm for the feed-forward sublayer
  
  def forward(self, x: torch.Tensor):
    # Addition is needed to represent additive residual connection
    x = x + self.self_attention(self.layer_norm1(x))
    x = x + self.feed_forward(self.layer_norm2(x))
    return x

#  [markdown]
# # Positional Encoding
# Use the sine and cosine positional encoding scheme

# 
# Credit: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# Adjusted so that input `x` can be of shape `[batch_size, seq_len, embedding_dim]`
import math
import torch
import torch.nn as nn

def compute_pos_encoding(block_size: int, d_model: int):
  positions = torch.arange(block_size).unsqueeze(1)
  div_terms = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
  position_encodings = torch.zeros(block_size, d_model, device=DEVICE) # (B, T, E)
  position_encodings[:, 0::2] = torch.sin(positions * div_terms)
  position_encodings[:, 1::2] = torch.cos(positions * div_terms)
  return position_encodings

#  [markdown]
# # Transformer
from torch.nn import functional as F

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

class Transformer1(nn.Module):
  def __init__(self, emb_size: int = 256):
    """
    Parameters:
      emb_size: the size of each word embeddings. For example: GloVe embeddings is 300, BERT is 768
    """
    super().__init__()
    self.emb_size = emb_size
    # 4 encoder blocks
    self.encoder_layers = nn.Sequential(
      EncoderBlock(emb_size, num_attention_heads=4),
      EncoderBlock(emb_size, num_attention_heads=4),
      EncoderBlock(emb_size, num_attention_heads=4),
      EncoderBlock(emb_size, num_attention_heads=4),
      nn.LayerNorm(emb_size)
    )
    # ENCODER COMPONENTS
    self.encoder_token_emb = nn.Embedding(src_vocab_size, emb_size)
    # Position embedding table: Convert each token's position in its block to a position embedding
    # Since it is using the sine and cosine positional encoding scheme, it's actually static
    self.register_buffer("positional_embedding", compute_pos_encoding(BLOCK_SIZE, emb_size))

    # DECODER COMPONENTS
    self.decoder_hidden_state_size = 2 * emb_size
    # Embedding lookup table: Convert token_ids to that token's corresponding embeddings
    self.decoder_token_emb = nn.Embedding(tgt_vocab_size, emb_size)
    # Target language modeling head: Transform back from the embedding dimension to the tgt_vocab_size dimension
    # So that we can get the distribution and know which target token to choose
    self.tgt_lm_head = nn.Linear(self.decoder_hidden_state_size, tgt_vocab_size)
    # For the decoder, we try to replicate the RNN model to process sequences
    # decoder_hidden_state = sigmoid(W1 * context_matrix + W2 * prev_hidden_state + bias + W3 * decoder_input_tok_emb)
    # Thus we need 3 weights matrix for use in the decoder, to produce a new decoder_hidden_state
    self.decoder_context_linear = nn.Linear(emb_size, self.decoder_hidden_state_size)
    self.decoder_hiddenstate_linear = nn.Linear(self.decoder_hidden_state_size, self.decoder_hidden_state_size)
    self.decoder_token_emb_linear = nn.Linear(emb_size, self.decoder_hidden_state_size)

  def forward(self, input_idx: torch.Tensor, target_idx: Union[torch.Tensor, None] = None):
    """
    Parameters:
      target_idx: the list of target tokens across the batches. Dimension (B, T)
    """
    x = self.encode(input_idx) # (B, T, E)
    # Average the last hidden state of the encoder as the context
    context_emb = torch.mean(x, dim=1) # (B, E)
    # Feed the <START> token as the chosen token to the entire batch
    chosen_tokens = torch.tensor(toks_encode(["<START>"] * x.size(0), "target"), device=DEVICE) # (B)
    ys: list[torch.Tensor] = []
    # Initialize the first hidden state as 0s
    hidden_state = torch.zeros((BATCH_SIZE, self.decoder_hidden_state_size), device=DEVICE) # (B, H) where H = dec_hidden_state
    for _ in range(max_len_target):
      # hidden_state: (B, H)
      # tgt_probs: (B, C) where C = tgt_vocab_size
      hidden_state, tgt_probs = self.decode(context_emb, chosen_tokens, hidden_state)
      # Greedily select the token with highest prob from the distribution
      chosen_tokens = torch.argmax(tgt_probs, dim=1) # (B)
      ys.append(tgt_probs)

    # Note that ys is collected by looping over max_len_target, so when stacked, the first dimension is max_len_target
    y = torch.stack(ys) # (T, B, C) where C = tgt_vocab_size
    assert y.shape == (max_len_target, BATCH_SIZE, tgt_vocab_size)
    if target_idx is None:
      return ys, None
    # Cross_entropy requires the "Class" dimension to be the 2nd dimension
    T, B, C = y.shape
    y = y.view(B*T, C)
    target_idx = target_idx.view(B*T)
    print(target_idx)
    # Calculate loss
    loss = F.cross_entropy(y, target_idx, ignore_index=toks_encode([PAD_TOKEN], "target")[0])
    return ys, loss

  def encode(self, input_batch: torch.Tensor):
    # Input batch is of shape (B, T) (i.e. (batch size, block_size))
    token_emb = self.encoder_token_emb(input_batch) # (B, T, E) where E=emb_size
    # the position_embedding_table takes input the position of each token in the sequence (i.e. the T dimension)
    position_emb = self.positional_embedding # (T, E)
    x = token_emb + position_emb # (B, T, E)
    assert x.shape == (BATCH_SIZE, BLOCK_SIZE, self.emb_size)
    # Feed this x through layers of Transformer Self-Attention blocks
    x = self.encoder_layers(x)
    return x

  def decode(self, context_emb: torch.Tensor, input_tokenIds: torch.Tensor, prev_hidden_state: torch.Tensor):
    """Decode the logis from the encoder to produce a target token"""
    # Right now let's use an RNN-like decoder
    assert context_emb.shape == (BATCH_SIZE, self.emb_size)
    assert input_tokenIds.shape == (BATCH_SIZE,)
    assert prev_hidden_state.shape == (BATCH_SIZE, self.decoder_hidden_state_size), f"Got {prev_hidden_state.shape}"
    # For the decoder, we try to replicate the RNN model to process sequences
    # decoder_hidden_state = tanh(W1 * context_matrix + W2 * prev_hidden_state + W3 * decoder_input_tok_emb + bias)
    temp1 = self.decoder_context_linear(context_emb) # (B, dec_hidden_size)
    temp2 = self.decoder_hiddenstate_linear(prev_hidden_state) # (B, dec_hiden_size)
    tok_emb = self.decoder_token_emb(input_tokenIds) # (B, E)
    temp3 = self.decoder_token_emb_linear(tok_emb) # (B, dec_hidden_size)
    z = temp1 + temp2 + temp3 # (B, dec_hidden_size)
    hidden_state = torch.tanh(z) # (B, dec_hidden_size)
    assert hidden_state.shape == (BATCH_SIZE, self.decoder_hidden_state_size)

    tgt_distribution: torch.Tensor = self.tgt_lm_head(hidden_state) # (B, tgt_vocab_size)
    # Do NOT run softmax here as the Pytorch Cross Entropy Loss function expects unnormalized numbers
    # tgt_probs = F.softmax(tgt_distribution, dim=-1) # (B, tgt_vocab_size)
    return hidden_state, tgt_distribution

  def generate(self, input_idx: torch.Tensor, max_generated_tokens: int = 20):
    with torch.no_grad():
      encoder_last_hidden_state = self.encode(input_idx) # (B, T, E)
      # Average all input tokens embs across the encoder last hidden state as the context
      context_emb = torch.mean(encoder_last_hidden_state, dim=1) # (B, E)
      chosen_tokens = torch.tensor(toks_encode(["<START>" for _ in range(BATCH_SIZE)], "target"), device=DEVICE) # (B)
      first_batch_predicted_tokens: list[int] = []
      # Initialize the first hidden state as 0s
      hidden_state = torch.zeros((BATCH_SIZE, self.decoder_hidden_state_size), device=DEVICE) # (B, H) where H = dec_hidden_state
      for _ in range(max_generated_tokens):
        # hidden_state: dimension (B, H)
        # tgt_probs: dimension (B, tgt_vocab_size)
        hidden_state, tgt_probs = self.decode(context_emb, chosen_tokens, prev_hidden_state=hidden_state)
        # Greedily select the token with highest prob from the distribution
        chosen_tokens = torch.argmax(tgt_probs, dim=1) # (B)
        # print(chosen_tokens)
        chosen_token = chosen_tokens[0].item() # View result of only the first batch
        first_batch_predicted_tokens.append(int(chosen_token))
    return first_batch_predicted_tokens

if __name__ == '__main__':
  m1 = Transformer1()
  m1 = m1.to(DEVICE)

  m1.eval()
  train_dataloader_iter = iter(train_dataloader)
  x_batch = next(train_dataloader_iter)

  input = torch.cat((x_batch[0], x_batch[1], x_batch[2]), dim=1) # (B, block_size)
  print("SAMPLE GENERATING")
  print('Sample input', toks_decode(input.tolist()[0], "source"))
  first_batch_predicted_tokens = m1.generate(input)
  predicted = toks_decode(first_batch_predicted_tokens, "target")
  print(predicted)

  # Train the network
  # Create an optimizer
  m1.train()
  optimizer = torch.optim.AdamW(m1.parameters(), lr=5e-5)
  print("BOOTSTRAPPING THE DATALOADER")

  print("START TRAINING")

  # for epoch in range(30):
  #   try:
  #     batch = next(train_dataloader_iter)
  #   except StopIteration:
  #     # Reset the dataloader
  #     train_dataloader_iter = iter(train_dataloader)
  #     batch = next(train_dataloader_iter)
  #   input = torch.cat((batch[0], batch[1], batch[2]), dim=1) # (B, block_size)
  #   target = batch[3] # (B, max_len_target)
  #   ys, loss = m1(input, target)
  #   optimizer.zero_grad(set_to_none=True)
  #   loss.backward()
  #   optimizer.step()
  #   print(epoch, loss)

  # After training
  # print(losses)
  # Save the model
  # torch.save(m1.state_dict(), "transform1.pt")

  with open("temptemp.txt", "w") as f:
    for i, batch in enumerate(train_dataloader):
      f.write(f"{i}\n")
      f.write(f"{batch[0]} {batch[1]} {batch[2]} {batch[3]}\n")

  # Inference
  m1_trained = Transformer1()
  m1_trained.load_state_dict(torch.load("transform1.pt"))
  m1_trained = m1_trained.to(DEVICE)
  m1_trained.eval()

  x_batch = next(train_dataloader_iter)

  input = torch.cat((x_batch[0], x_batch[1], x_batch[2]), dim=1) # (B, block_size)
  target = x_batch[3].tolist()[0]
  print('Target tokens', target)
  print('Target words', toks_decode(target, "target"))

  y_batch = m1_trained.generate(input, max_generated_tokens=max_len_target)
  print('Inference input words', toks_decode(input.tolist()[0], "source"))
  print('Inferece output tokens', y_batch)
  print('Inference output words', toks_decode(y_batch, "target"))


