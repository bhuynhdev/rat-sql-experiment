from grammars.asdl import Field, ASDLType
from grammars.transition_system import ApplyRuleAction, ReduceAction, Pos, Done, Partial, Action
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Union, TypedDict

import spacy
import torch
from torch.utils.data import DataLoader, Dataset
from download_spider import download_spider_zip, extract_zip
from grammars.spider_transition_system import (
  SpiderTransitionSystem,
  SpiderSingletonAction, SpiderGenTokenAction, SpiderColumnAction, SpiderTableAction, SpiderStringAction, SpiderObjectAction, SpiderIntAction
)

# Prepare Stanford's Stanza to tokenize the columns and tables names
# stanza.download('en')
# stanza_nlp = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='en')

BATCH_SIZE = 24

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
  spacy_nlp.tokenizer.rules = {key: value for key, value in spacy_nlp.tokenizer.rules.items() if key != "id"}  # type: ignore

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


class Vocab:
  """Vocab class contains set all words in a vocab, with helper method for encode/decoding token <-> id"""

  def __init__(self, name: str, items: list[str]):
    self.name = name
    self.items = set(items)
    self.tok_to_idx = {item: i for i, item in enumerate(items)}
    self.idx_to_tok = {i: item for i, item in enumerate(items)}

  def encode(self, words: list[str]) -> list[int]:
    """Encode an input word to a token id. Returns -1 if not exists ("UNK" token)"""
    return [self.tok_to_idx[word] for word in words]

  def decode(self, tokIds: list[int]) -> list[str]:
    """Decode a token id to a vocab word. Throws error if tokId not found"""
    return [self.idx_to_tok[tokId] for tokId in tokIds]


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


class AllVocabs(NamedTuple):
  src_vocab: Vocab
  action_vocab: Vocab


def create_vocabs(schema_lookup: dict[str, Database], train_qas: list[QAPair], val_qas: list[QAPair], transition_system: SpiderTransitionSystem) -> AllVocabs:
  # First create the vocab of the input, which comprises the column names, column types, table names, NL utterance
  col_name_vocab: set[str] = set()
  tbl_name_vocab: set[str] = set()
  col_types: set[str] = set()
  question_vocab: set[str] = set()  # Vocab of the Natural Language questions

  all_columns = [col for s in schema_lookup.values() for col in s.columns]
  all_tables = [tbl for s in schema_lookup.values() for tbl in s.tables]

  all_actions_from_items: list[Action] = []

  # Build col names and tbl names vocab
  for col in all_columns:
    col_name_vocab.update(col.name_toks)
    col_types.add(col.col_type)

  for tbl in all_tables:
    tbl_name_vocab.update(tbl.name_toks)

  # Build questions and SQL vocab
  for qa_pair in train_qas:
    all_actions_from_items.extend(transition_system.get_actions(transition_system.surface_code_to_ast(qa_pair.sql_tree)))
    question_vocab.update(qa_pair.question_toks)

  for qa_pair in val_qas:
    all_actions_from_items.extend(transition_system.get_actions(transition_system.surface_code_to_ast(qa_pair.sql_tree)))
    question_vocab.update(qa_pair.question_toks)

  # All tokens to be used to create src_vocab
  # Note that the order matters, i.e. 0th element will have token_id 0
  src_tokens = [PAD_TOKEN, "CLS"] + list(col_name_vocab | col_types | tbl_name_vocab | question_vocab)
  src_vocab = Vocab("source", src_tokens)


  # Now create the vocab of the decoder, which consists of the actions (ApplyRuleAction, ReduceAction, SpiderGenTokenAction)
  apply_rule_actions = [ApplyRuleAction(production) for production in transition_system.grammar.productions]
  reduce_action = [ReduceAction()]
  action_tokens = [PAD_TOKEN, "<START>", "<END>"] + list(set([repr(a) for a in apply_rule_actions + reduce_action + all_actions_from_items]))
  action_vocab = Vocab("target", action_tokens)

  return AllVocabs(src_vocab, action_vocab)


class DatasetItem(NamedTuple):
  """
  An item in the dataset
  Fields:
  - item = The original SpiderItem passed in
  - input_seq = the input sequence (an int tensor of input vocab token ids) -> Dim (src_block_size)
  - target_seq = The target (gold) sequence (and int tensor of target vocab token ids, i.e. the SQL Constructor actions) -> Dim (target_block_size)
  - target_actions = list of Action generated from the transion_sytem
  - action_mask = the locations/indexes of all ApplyRule actions in the target_seq as a 1/0 mask -> Dim (target_block_size) 
  - copy_mask = the locations/indexes of all GenToken-copy actions in the target_seq as a 1/0 mask -> Dim (target_block_size)
  - copy_target_mask = the locations/indexes of tokens in the input_seq that should be coped into the target_seq -> Dim (src_block_size)
  - frontier_fields = Frontier fields of each action in `target_actions`
  - frontier_fields_idx = token indexes of the frontier field of each action in the target_seq -> Dim (target_block_size)
  - frontier_typess_idx = token indexes of the type of each frontier field type
 """
  id: int
  item: SpiderItem
  input_seq: torch.Tensor
  target_seq: torch.Tensor
  target_actions: list[Action]
  action_mask: torch.Tensor
  copy_mask: torch.Tensor
  copy_target_mask: torch.Tensor
  frontier_fields: list[Union[Field, None]]
  frontier_fields_idx: torch.Tensor
  frontier_types_idx: torch.Tensor


def construct_input_sequence(item: SpiderItem):
  """Construct the input block from the provided item
  The input block consists of 3 parts
    The col name part: `CLS col_type ...col_name CLS col_type2 ...Col_name2`
    The table name part: `CLS ...tbl_name CLS ...tbl_name2`
    The question part: ` CLS ...question`
  Also returns the maps from the column/table in the database to the location of that column/table in the input sequence
  """
  input_seq: list[str] = ["CLS"]
  # Map from column/table index in the database to the location of this column/table in the input sequence
  column_position_map = dict[int, int]()
  table_position_map = dict[int, int]()
  # Map a word in the NL question to its location in the input sequence
  word_position_map = dict[str, int]()
  # input seq = question + col names + table names
  for word in item.qa_pair.question_toks:
    input_seq.append(word)
    word_position_map[word] = len(input_seq) - 1
  for col in item.db.columns:
    input_seq.append("CLS")
    column_position_map[col.col_index] = len(input_seq) - 1
    input_seq.extend(col.name_toks)
    input_seq.append(col.col_type)
  for tbl in item.db.tables:
    input_seq.append("CLS")
    table_position_map[tbl.table_index] = len(input_seq) - 1
    input_seq.extend(tbl.name_toks)
  return (input_seq, column_position_map, table_position_map, word_position_map)


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

  def __init__(self, all_items: list[SpiderItem], all_vocabs: AllVocabs, transition_system: SpiderTransitionSystem):
    self.vocabs = all_vocabs
    self.processed_items: list[DatasetItem] = []
    self.block_size = 0  # The maximum input sequence size
    self.target_size = 0  # The maximum target sequence size

    for i, item in enumerate(all_items):
      # Encoder item preprocessing
      # columns_tokens = [c.name_toks for c in item.db.columns[1:]]  # Skip the 0th-index "*" column
      # columns_types: list[ColumnType] = [c.col_type for c in item.db.columns[1:]]
      # tables_tokens = [t.name_toks for t in item.db.tables]
      # question_tokens = item.qa_pair.question_toks
      # input_parts = construct_input_parts(columns_tokens, columns_types, tables_tokens, question_tokens)
      input_seq, column_position_map, table_position_map, word_position_map = construct_input_sequence(item)
      input_sequence = torch.tensor(self.vocabs.src_vocab.encode(input_seq), dtype=torch.int64)  # Convert to tensor

      # Decoder item preprocessing
      frontier_fields: list[Union[Field, None]] = []

      target_actions = transition_system.get_actions(transition_system.surface_code_to_ast(item.qa_pair.sql_tree))
      target_tokens: list[str] = []
      action_mask = torch.zeros(len(target_actions))
      copy_mask = torch.ones(len(target_actions))
      copy_target_mask = torch.zeros(len(input_sequence))

      # Initiate a parsing process in order to retrieve frontier_field information
      parse_result = transition_system.parse()
      for pos, action in enumerate(target_actions):
        parse_result = parse_result.cont(Pos(pos), action)
        if isinstance(parse_result, Done):
          break
        assert isinstance(parse_result, Partial)
        frontier_fields.append(parse_result.frontier_field)
        # Inside the for loop is also an opportunity to fill out the masks
        if isinstance(action, ApplyRuleAction) or isinstance(action, ReduceAction):
          action_mask[pos] = 1
          copy_mask[pos] = 0
          target_tokens.append(repr(action))
        else:  # If not ApplyRule or Reduce, then must be GenToken
          assert isinstance(action, SpiderGenTokenAction)
          if str(action.token) == "[]":  # Special token that indicates end of primitve token generation
            continue
          # assert not isinstance(action.token, list), f"Expected {action.token} to be str. SQL {item.qa_pair.query}\nACTIONS {target_actions}"
          if isinstance(action, SpiderSingletonAction) or isinstance(action, SpiderIntAction):
            # These gen action tokens shall not be copied, but generated from the vocab, so we mark them as action
            action_mask[pos] = 1
            copy_mask[pos] = 0
            target_tokens.append(repr(action))
            continue
          # Else, every other type of GenToken action is copied from input, so we fill out the copy_mask and copy_target_mask
          if isinstance(action, SpiderColumnAction):
            column_index = int(action.token)
            column_position_in_input = column_position_map.get(column_index, None)
            if column_position_in_input:
              copy_target_mask[column_position_in_input] = 1
          elif isinstance(action, SpiderTableAction):
            table_index = int(action.token)
            table_position_in_input = table_position_map.get(table_index, None)
            if table_position_in_input:
              copy_target_mask[table_position_in_input] = 1
          elif isinstance(action, SpiderStringAction) or isinstance(action, SpiderObjectAction):
            token = str(action.token)
            word_position_in_input = word_position_map.get(token, None)
            if word_position_in_input: 
              copy_target_mask[word_position_in_input] = 1
          else:
            raise Exception(f"Unhandled action {action} from statement {item.qa_pair.question}:{
                            item.qa_pair.query}\ntree {'\n'.join(map(repr, target_actions))}")

      assert isinstance(parse_result, Done), "Action sequence is incomplete!"

      self.block_size = max(self.block_size, len(input_seq))
      self.target_size = max(self.target_size, len(target_tokens))

      target_sequence = torch.tensor(self.vocabs.action_vocab.encode(target_tokens), dtype=torch.int64)
      # Plus one to the tokenIds because we'll be using index 0 for None frontier field
      frontier_fields_idx = torch.Tensor([transition_system.grammar.field2id[f] + 1 if f else 0 for f in frontier_fields])
      frontier_field_types_idx = torch.Tensor([transition_system.grammar.type2id[f.type] + 1 if f else 0 for f in frontier_fields])
      # Pad the sequences so they matches the max length
      input_sequence = torch.nn.functional.pad(input_sequence, (0, BLOCK_SIZE - input_sequence.size(0)), value=0)
      target_sequence = torch.nn.functional.pad(target_sequence, (0, TGT_SIZE - target_sequence.size(0)), value=0)
      action_mask = torch.nn.functional.pad(action_mask, (0, TGT_SIZE - action_mask.size(0)), value=0)
      copy_mask = torch.nn.functional.pad(copy_mask, (0, TGT_SIZE - copy_mask.size(0)), value=0)
      frontier_fields_idx = torch.nn.functional.pad(frontier_fields_idx, (0, TGT_SIZE - frontier_fields_idx.size(0)), value=0)
      frontier_field_types_idx = torch.nn.functional.pad(frontier_field_types_idx, (0, TGT_SIZE - frontier_field_types_idx.size(0)), value=0)
      copy_target_mask = torch.nn.functional.pad(copy_target_mask, (0, BLOCK_SIZE - copy_target_mask.size(0)), value=0)
      assert input_sequence.shape == (BLOCK_SIZE,)
      assert target_sequence.shape == (TGT_SIZE,)

      self.processed_items.append(DatasetItem(
        id=i,
        item=item,
        input_seq=input_sequence,
        target_seq=target_sequence,
        target_actions=target_actions,
        frontier_fields=frontier_fields,
        frontier_fields_idx=frontier_fields_idx,
        frontier_types_idx=frontier_field_types_idx,
        action_mask=action_mask,
        copy_mask=copy_mask,
        copy_target_mask=copy_target_mask
      ))

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
# These numbers are found after running `get_global_block_size_and_tgt_size` and `get_vocab` isolatedly
BLOCK_SIZE = 1309
TGT_SIZE = 190
SRC_VOCAB_SIZE = 4195
TGT_VOCAB_SIZE = 1638

@dataclass
class ModelInput:
  id: torch.Tensor
  input: torch.Tensor
  target: torch.Tensor
  action_mask: torch.Tensor
  copy_mask: torch.Tensor
  copy_target_mask: torch.Tensor
  field_idx: torch.Tensor
  type_idx: torch.Tensor

def collate_function(batch: list[DatasetItem]):
  ids = [b.id for b in batch]
  input_seqs = [b.input_seq for b in batch]
  target_seqs = [b.target_seq for b in batch]
  action_masks = [b.action_mask for b in batch]
  copy_masks = [b.copy_mask for b in batch]
  copy_target_masks = [b.copy_target_mask for b in batch]
  field_seqs = [b.frontier_fields_idx for b in batch]
  type_seqs = [b.frontier_types_idx for b in batch]

  return ModelInput(
    id=torch.tensor(ids),
    input=torch.stack(input_seqs),
    target = torch.stack(target_seqs),
    action_mask = torch.stack(action_masks),
    copy_mask = torch.stack(copy_masks),
    copy_target_mask = torch.stack(copy_target_masks),
    field_idx = torch.stack(field_seqs),
    type_idx = torch.stack(type_seqs),
  )

def create_dataloaders(train_dataset: MyDataset, val_dataset: MyDataset, batch_size: int, num_workers: int = 4):
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_function)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_function)
  return (train_dataloader, val_dataloader)


def everything():
  """The entire preprocessing pipeline"""
  schema_lookup = create_database_schemas("spider/tables.json")
  train_qas, val_qas, train_items, val_items = create_training_items(
      "spider/train_spider.json", "spider/dev.json", schema_lookup
  )
  transition_system = SpiderTransitionSystem("grammars/Spider2.asdl", output_from=True)
  vocabs = create_vocabs(schema_lookup, train_qas, val_qas, transition_system)
  train_dataset = MyDataset(train_items, vocabs, transition_system)
  val_dataset = MyDataset(val_items, vocabs, transition_system)
  train_dataloader, val_dataloader = create_dataloaders(train_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=4)
  return (train_dataset, val_dataset, train_dataloader, val_dataloader, vocabs)


if __name__ == "__main__":
  spider_zip_path = download_spider_zip()
  extract_zip(spider_zip_path)

  if not os.path.exists("spider"):
    raise Exception("'spider' folder not found. Have you download and extracted the 'spider.zip' file?")

  train_dataset, val_dataset, train_dataloader, val_dataloader, vocabs = everything()

  block_size, tgt_size = get_global_block_size_and_tgt_size(train_dataset, val_dataset)
  print(f"{block_size=} {tgt_size=}")
  print(f"{len(vocabs.src_vocab.items)=} {len(vocabs.action_vocab.items)=}")

  dataset_item = train_dataset[15]
  print("ITEM")
  print(dataset_item.item.qa_pair.query)
  print(dataset_item.target_actions)
  print(dataset_item.action_mask)

  # with open("temptemp1.txt", "w") as f:
  #   for i in range(len(train_dataset)):
  #     f.write(f"{train_dataset[i]}\n\n")

  # with open("temptemp2.txt", "w") as f:
  #   for i, batch in enumerate(train_dataloader):
  #     f.write(f"{i}\n")
  #     f.write(f"{batch[0]} {batch[1]}\n")
