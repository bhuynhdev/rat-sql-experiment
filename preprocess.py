import spacy
import torch
import json
from dataclasses import dataclass, field
from typing import Literal, Any, NamedTuple
from torch.utils.data import Dataset, DataLoader
import os
import gdown

# Prepare Stanford's Stanza to tokenize the columns and tables names
# stanza.download('en')
# stanza_nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='en')


ColumnType = Literal["time", "text", "number", "boolean", "others"]


def lemmatized_tokenizer(s: str, spacy_nlp: spacy.language.Language):
  doc = spacy_nlp(s)
  lemmatized_tokens = [token.lemma_.lower() for token in doc]
  # doc = stanza_nlp(s)
  # lemmatized_tokens = [word.lemma.lower() for sent in doc.sentences  for word in sent.words]
  return lemmatized_tokens


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


def create_database_schemas(tables_file_path: str):

  # Start by loading in the databases, tables, and columns schemas
  with open(tables_file_path) as f:
    raw_schema_data = json.load(f)
  # raw_schema_data[0].keys() == ['column_names', 'column_names_original', 'column_types', 'db_id', 'foreign_keys', 'primary_keys', 'table_names', 'table_names_original']

  schema_lookup: dict[str, Database] = {}  # Map the db_id to that database's schema
  # Prepare spacy to tokenize and lemmatize the table names and columns names
  spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "parser"])
  # Fix the "Spacy breaks up 'id' into 'i' and 'd' issue" https://github.com/explosion/spaCy/discussions/10570
  spacy_nlp.tokenizer.rules = {key: value for key, value in spacy_nlp.tokenizer.rules.items() if key != "id"}

  for _, db_info in enumerate(raw_schema_data):
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
          name_toks=lemmatized_tokenizer(tbl_name, spacy_nlp),
          orig_name=tbl_orig_name,
      )
      db.tables.append(new_table)

    # Iterate the columns of this database
    for col_index, ((table_idx, col_name), (_, col_orig_name), col_type,) in enumerate(
      zip(db_info["column_names"], db_info["column_names_original"], db_info["column_types"])
    ):
      new_col = Column(
          db_id=db_id,
          col_index=col_index,
          col_type=col_type,
          table_index=table_idx,
          name=col_name,
          name_toks=lemmatized_tokenizer(col_name, spacy_nlp),
          orig_name=col_orig_name,
      )
      db.columns.append(new_col)
      table_idx: int = table_idx
      if table_idx >= 0:  # Skip the -1 index table
        db.tables[table_idx].columns.append(new_col)

    schema_lookup[db_id] = db

  return schema_lookup


@dataclass
class QAPair:  # Class to encapsulate each NL query with corresponding SQL answer
  db_id: str
  question_toks: list[str]
  question: str
  query_toks: list[str]
  query: str
  sql_tree: Any  # Raw sql tree as a nested dict from the raw JSON


@dataclass
class SpiderItem:
  """An item to feed into the neural network. Basically encapsulation of a training example"""
  db_id: str
  qa_pair: QAPair
  db: Database  # Information about the database used in the qa


def create_training_items(train_file_path: str, val_file_path: str, schema_lookup: dict[str, Database]):
  # Load the questions
  with open(train_file_path) as f:
    raw_train_data = json.load(f)
  # raw_train_data[0].keys()
  # ['db_id', 'query', 'query_toks', 'query_toks_no_value', 'question', 'question_toks', 'sql']

  with open(val_file_path) as f:
    raw_val_data = json.load(f)

  train_qas: list[QAPair] = []
  val_qas: list[QAPair] = []
  train_items: list[SpiderItem] = []
  val_items: list[SpiderItem] = []

  for _, qa_info in enumerate(raw_train_data):
    db_id = qa_info["db_id"]
    new_qa = QAPair(
        db_id=db_id,
        question_toks=qa_info["question_toks"],
        question=qa_info["question"],
        query_toks=qa_info["query_toks"],
        query=qa_info["query"],
        sql_tree=qa_info["sql"],
    )
    new_item = SpiderItem(db_id=db_id, qa_pair=new_qa, db=schema_lookup[db_id])
    train_qas.append(new_qa)
    train_items.append(new_item)

  for _, qa_info in enumerate(raw_val_data):
    db_id = qa_info["db_id"]
    new_qa = QAPair(
        db_id=db_id,
        question_toks=qa_info["question_toks"],
        question=qa_info["question"],
        query_toks=qa_info["query_toks"],
        query=qa_info["query"],
        sql_tree=qa_info["sql"],
    )
    new_item = SpiderItem(db_id=db_id, qa_pair=new_qa, db=schema_lookup[db_id])
    val_qas.append(new_qa)
    val_items.append(new_item)

  return (train_qas, val_qas, train_items, val_items)


# Padding token used to pad shorter inputs so every sequence of inputs to the encoder
# is of same length (which would be the max length)
PAD_TOKEN = "<PAD>"


def create_vocabs(schema_lookup: dict[str, Database], train_qas: list[QAPair], val_qas: list[QAPair]):
  col_name_vocab: set[str] = set()
  tbl_name_vocab: set[str] = set()
  col_types: set[str] = set()
  sql_vocab: set[str] = set()  # Vocab of the SQL target
  question_vocab: set[str] = set()  # Vocab of the Natural Language questions

  src_vocab: set[str] = set()  # Source vocab used for inputs to the encoder, which includes questions & columns & tables names
  tgt_vocab: set[str] = set()  # Target vocab used for decoder output; which is the set of SQL tokens plus the Start and End tokens

  all_columns = [col for s in schema_lookup.values() for col in s.columns]
  all_tables = [tbl for s in schema_lookup.values() for tbl in s.tables]

  # Build col names and tbl names vocab
  for col in all_columns:
    col_name_vocab.update(col.name_toks)
    col_types.add(col.col_type)

  for tbl in all_tables:
    tbl_name_vocab.update(tbl.name_toks)

  # Build questions and SQL vocab
  for qa_pair in train_qas:
    question_vocab.update(qa_pair.question_toks)
    sql_vocab.update(qa_pair.query_toks)

  for qa_pair in val_qas:
    question_vocab.update(qa_pair.question_toks)
    sql_vocab.update(qa_pair.query_toks)

  src_vocab.add(PAD_TOKEN)
  # Add the "CLS" token as a separator between each part in the input block (i.e. separator between each column name, table name, etc.)
  src_vocab.add("CLS")
  src_vocab.update(col_name_vocab)
  src_vocab.update(col_types)
  src_vocab.update(tbl_name_vocab)
  src_vocab.update(question_vocab)

  tgt_vocab.add("<START>")
  tgt_vocab.add("<END>")
  tgt_vocab.add(PAD_TOKEN)
  tgt_vocab.update(sql_vocab)

  return (src_vocab, tgt_vocab)


class TokenLookupTables(NamedTuple):
  src_tok_to_idx: dict[str, int]
  src_idx_to_tok: dict[int, str]
  tgt_tok_to_idx: dict[str, int]
  tgt_idx_to_tok: dict[int, str]


def create_token_lookup_tables(src_vocab: set[str], tgt_vocab: set[str]):
  # Token to id: Assign a number to each token in the vocab
  # Id to token: Retrieve the token give the index

  # Note that we intend to assign the PAD token to index 0, so we start enumerating from index 1
  # Also temporarily remove the PAD_TOKEN so all other indexes do not have gaps
  src_vocab.remove(PAD_TOKEN)
  src_tok_to_idx = {tok: i for i, tok in enumerate(sorted(list(src_vocab)), start=1)}
  src_tok_to_idx[PAD_TOKEN] = 0

  src_idx_to_tok = {i: tok for i, tok in enumerate(sorted(list(src_vocab)), start=1)}
  src_idx_to_tok[0] = PAD_TOKEN

  tgt_vocab.remove(PAD_TOKEN)
  tgt_vocab.remove("<START>")
  tgt_tok_to_idx = {tok: i for i, tok in enumerate(sorted(list(tgt_vocab)), start=2)}
  tgt_tok_to_idx[PAD_TOKEN] = 0
  tgt_tok_to_idx["<START>"] = 0

  tgt_idx_to_tok = {i: tok for i, tok in enumerate(sorted(list(tgt_vocab)), start=2)}
  tgt_idx_to_tok[0] = PAD_TOKEN
  tgt_idx_to_tok[1] = "<START>"

  return TokenLookupTables(src_tok_to_idx, src_idx_to_tok, tgt_tok_to_idx, tgt_idx_to_tok)


def toks_encode(tokens_list: list[str], token_lookup_tables: TokenLookupTables, mode: Literal["source", "target"]):
  """Given a list of tokens, 'encode' them to return a list of token_ids
  Parameters:
    mode: Whether to use the source's or target's tok_to_idx lookup table
  """
  if mode == "source":
    return [token_lookup_tables.src_tok_to_idx[token] for token in tokens_list]
  return [token_lookup_tables.tgt_tok_to_idx[token] for token in tokens_list]


def toks_decode(ids_list: list[int], token_lookup_tables: TokenLookupTables, mode: Literal["source", "target"]):
  """Given a list of token_ids, 'decode' them to return the list of corresponding token
  Parameters:
    mode: Whether to use the source's or target's idx_to_tok lookup table
  """
  if mode == "source":
    return [token_lookup_tables.src_idx_to_tok[token_id] for token_id in ids_list]
  return [token_lookup_tables.tgt_idx_to_tok[token_id] for token_id in ids_list]


# Each dataset item is 2-element tuple:
# First element is the input sequence (an int tensor of token ids)
# Second element is the target sequence (and int tensor of token ids)
class DatasetItem(NamedTuple):
  input_seq: torch.Tensor
  target_seq: torch.Tensor


def construct_input_parts(col_toks: list[list[str]], col_types: list[ColumnType], table_toks: list[list[str]], ques_toks: list[str]):
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
  col_name_part: list[str] = [tok for col_name_toks, col_type in zip(col_toks, col_types) for tok in (["CLS", col_type] + col_name_toks)]
  # Construct the col name part, thus producing something like:
  # "CLS building information CLS city information ...etc"
  tbl_name_part: list[str] = [tok for tbl_name_toks in table_toks for tok in ["CLS"] + tbl_name_toks]
  # Construct the question part, thus producing something like:
  # "CLS what is the highest building in Chicago"
  question_part: list[str] = ["CLS"] + ques_toks
  return (col_name_part, tbl_name_part, question_part)


class MyDataset(Dataset[DatasetItem]):
  """Each element in this dataset is a 4-element tuple
  (col_name_toks, table_name_toks, question_toks, target_toks)

  where `col_name_toks`, `table_name_toks`, `question_toks` has been preprocessed to add "CLS" separator token
  and `target_toks` is also processed to include <Start> and <End> tokens
  """

  def __init__(self, all_items: list[SpiderItem], token_lookup_tables: TokenLookupTables):
    self.token_lookup_tables = token_lookup_tables
    self.processed_items: list[DatasetItem] = []
    self.block_size = 0  # The maximum input sequence size
    self.target_size = 0  # The maximum target sequence size

    for item in all_items:
      columns_tokens = [c.name_toks for c in item.db.columns[1:]]  # Skip the 0th-index "*" column
      columns_types: list[ColumnType] = [c.col_type for c in item.db.columns[1:]]
      tables_tokens = [t.name_toks for t in item.db.tables]
      question_tokens = item.qa_pair.question_toks
      target_tokens = item.qa_pair.query_toks + ["<END>"]

      input_parts = construct_input_parts(columns_tokens, columns_types, tables_tokens, question_tokens)
      # Concat the input parts to create the input sequence
      input_parts_concat = input_parts[0] + input_parts[1] + input_parts[2]
      self.block_size = max(self.block_size, len(input_parts_concat))
      self.target_size = max(self.target_size, len(target_tokens))
      # Convert to tensors
      input_sequence = torch.tensor(toks_encode(input_parts_concat, self.token_lookup_tables, "source"), dtype=torch.int64)
      target_sequence = torch.tensor(toks_encode(target_tokens, self.token_lookup_tables, "target"), dtype=torch.int64)
      # Pad the sequences so they matches the max length
      input_sequence = torch.nn.functional.pad(input_sequence, (0, BLOCK_SIZE - input_sequence.size(0)), value=0)
      target_sequence = torch.nn.functional.pad(target_sequence, (0, TGT_SIZE - target_sequence.size(0)), value=0)
      assert input_sequence.shape == (BLOCK_SIZE,)
      assert target_sequence.shape == (TGT_SIZE,)

      self.processed_items.append(DatasetItem(input_sequence, target_sequence))

  def __len__(self):
    return len(self.processed_items)

  def __getitem__(self, idx: int):
    return self.processed_items[idx]


def get_global_block_size_and_tgt_size(train_dataset: MyDataset, val_dataset: MyDataset):
  # Find the global block size and target size across the datasets
  block_size = max(train_dataset.block_size, val_dataset.block_size)
  target_size = max(train_dataset.target_size, val_dataset.target_size)
  return block_size, target_size


# Block size, target sequence, src vocab size, and target vocab size as hard-coded constants for ease of implementation
# These numbers are found after running `get_global_block_size_and_tgt_size`` and `get_vocab`` isolatedly
BLOCK_SIZE = 1306
TGT_SIZE = 91
SRC_VOCAB_SIZE = 4195
TGT_VOCAB_SIZE = 5471


def create_dataloaders(train_dataset: Dataset[DatasetItem], val_dataset: Dataset[DatasetItem], batch_size: int = 64, num_workers: int = 4):
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
  return (train_dataloader, val_dataloader)


def everything():
  """The entire preprocessing pipeline"""
  schema_lookup = create_database_schemas("spider/tables.json")
  train_qas, val_qas, train_items, val_items = create_training_items(
      "spider/train_spider.json", "spider/dev.json", schema_lookup
  )
  src_vocab, tgt_vocab = create_vocabs(schema_lookup, train_qas, val_qas)
  print(len(src_vocab))
  print(len(tgt_vocab))
  token_lookup_tables = create_token_lookup_tables(src_vocab, tgt_vocab)
  train_dataset = MyDataset(train_items, token_lookup_tables)
  val_dataset = MyDataset(val_items, token_lookup_tables)
  train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, batch_size=64, num_workers=4)
  return (train_dataset, val_dataset, train_dataloader, val_dataloader, token_lookup_tables)


def download_spider_zip():
  spider_url = "https://drive.google.com/u/0/uc?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download"  # https://yale-lily.github.io/spider
  if not os.path.exists("spider.zip"):
    gdown.download(spider_url, "spider.zip", quiet=False)


if __name__ == "__main__":
  download_spider_zip()

  if not os.path.exists("spider"):
    raise Exception("'spider' folder not found")

  train_dataset, val_dataset, train_dataloader, val_dataloader, token_lookup_tables = everything()

  block_size, tgt_size = get_global_block_size_and_tgt_size(train_dataset, val_dataset)
  print(f"{block_size=} {tgt_size=}")

  # with open("temptemp1.txt", "w") as f:
  #   for i in range(len(train_dataset)):
  #     f.write(f"{train_dataset[i]}\n\n")

  # with open("temptemp2.txt", "w") as f:
  #   for i, batch in enumerate(train_dataloader):
  #     f.write(f"{i}\n")
  #     f.write(f"{batch[0]} {batch[1]}\n")
