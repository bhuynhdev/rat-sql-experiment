import spacy
import torch
from torch import nn
import json
from dataclasses import dataclass, field
from typing import Literal, Any, NamedTuple
# Prepare Stanford's Stanza to tokenize the columns and tables names
# stanza.download('en')
# stanza_nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='en')

device = "cpu"
if torch.backends.mps.is_available():
  device = "mps" # Apple Metal Performance Shader (M1 chip)
if torch.cuda.is_available():
  device = "cuda"
DEVICE = device


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
  spacy_nlp.tokenizer.rules = { key: value for key, value in spacy_nlp.tokenizer.rules.items() if key != "id" }

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
      sql_tree=qa_info["sql"]
    )
    new_item = SpiderItem(
      db_id=db_id,
      qa_pair=new_qa,
      db=schema_lookup[db_id]
    )
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
      sql_tree=qa_info["sql"]
    )
    new_item = SpiderItem(
      db_id=db_id,
      qa_pair=new_qa,
      db=schema_lookup[db_id]
    )
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
  sql_vocab: set[str] = set() # Vocab of the SQL target
  question_vocab: set[str] = set() # Vocab of the Natural Language questions
  
  src_vocab: set[str] = set() # Source vocab used for inputs to the encoder, which includes questions & columns & tables names
  tgt_vocab: set[str] = set() # Target vocab used for decoder output; which is the set of SQL tokens plus the Start and End tokens

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


def create_token_lookup_tables(src_vocab: set[str], tgt_vocab: set[str])
  # Token to id: Assign a number to each token in the vocab
  # Id to token: Retrieve the token give the index
  src_tok_to_idx = { tok: i for i, tok in enumerate(sorted(list(src_vocab))) }
  src_idx_to_tok = { i: tok for i, tok in enumerate(sorted(list(src_vocab))) }
  tgt_tok_to_idx = { tok: i for i, tok in enumerate(sorted(list(tgt_vocab))) }
  tgt_idx_to_tok = { i: tok for i, tok in enumerate(sorted(list(tgt_vocab))) }

  return TokenLookupTables(src_tok_to_idx, src_idx_to_tok, tgt_tok_to_idx, tgt_idx_to_tok)

def create_tokens_encode_decode_functions(token_lookup_tables: TokenLookupTables):
  def toks_encode(tokens_list: list[str], mode: Literal["source", "target"], ):
    """Given a list of tokens, 'encode' them to return a list of token_ids
    Parameters:
      mode: Whether to use the source's or target's tok_to_idx lookup table
    """
    if mode == "source":
      return [token_lookup_tables.src_tok_to_idx[token] for token in tokens_list]
    return [token_lookup_tables.tgt_tok_to_idx[token] for token in tokens_list]

  def toks_decode(ids_list: list[int], mode: Literal["source", "target"]):
    """Given a list of token_ids, 'decode' them to return the list of corresponding token
    Parameters:
      mode: Whether to use the source's or target's idx_to_tok lookup table
    """
    if mode == "source":
      return [token_lookup_tables.src_idx_to_tok[token_id] for token_id in ids_list]
    return [token_lookup_tables.tgt_idx_to_tok[token_id] for token_id in ids_list]

  return toks_encode, toks_decode

BLOCK_SIZE = max_len_col_name_part + max_len_tbl_name_part + max_len_ques_part




