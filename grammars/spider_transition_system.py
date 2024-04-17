# coding=utf-8
from __future__ import absolute_import

from dataclasses import dataclass
from functools import total_ordering
from typing import List, Generator, Tuple, Union, Optional, Type, Dict, Any
import re

from grammars.spider import SpiderGrammar
from grammars.asdl import (
    ASDLGrammar,
    ASDLCompositeType,
    ASDLPrimitiveType,
    Field,
)
from grammars.asdl_ast import RealizedField, AbstractSyntaxTree
from grammars.transition_system import (
    GenTokenAction,
    TransitionSystem,
    ReduceAction,
    Action,
    MaskAction,
    ApplyRuleAction,
    UnkAction,
    ACTION_CLASS_ORDER,
    Pos,
    Done,
    Partial
)

from grammars.hypothesis import Hypothesis


class SpiderGenTokenAction(GenTokenAction):
  token: str

  def __eq__(self, other: "Action"):
    if isinstance(other, self.__class__):
      return self.token == other.token
    elif isinstance(other, Action):
      return False
    else:
      return NotImplemented

  def __lt__(self, other: "Action") -> bool:
    if isinstance(other, self.__class__):
      return self.token < other.token
    elif isinstance(other, SpiderGenTokenAction):
      return (
          SPIDER_GEN_TOKEN_ACTION_CLASS_ORDER[self.__class__]
          < SPIDER_GEN_TOKEN_ACTION_CLASS_ORDER[other.__class__]
      )
    elif isinstance(other, Action):
      return next(
          k for v, k in ACTION_CLASS_ORDER.items() if isinstance(self, v)
      ) < next(k for v, k in ACTION_CLASS_ORDER.items() if isinstance(other, v))
    else:
      return NotImplemented


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class SpiderStringAction(SpiderGenTokenAction):
  token: str

  def __repr__(self):
    return "StringPrimitive[%s]" % self.token


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class SpiderTableAction(SpiderGenTokenAction):
  token: str

  def __repr__(self):
    return "TablePrimitive[%s]" % self.token


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class SpiderColumnAction(SpiderGenTokenAction):
  token: str

  def __repr__(self):
    return "ColumnPrimitive[%s]" % self.token


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class SpiderSingletonAction(SpiderGenTokenAction):
  token: str

  def __repr__(self):
    return "SingletonPrimitive[%s]" % self.token


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class SpiderObjectAction(SpiderGenTokenAction):
  token: str

  def __repr__(self):
    return "ObjectPrimitive[%s]" % self.token


@dataclass(order=False, eq=False, unsafe_hash=True, frozen=True)
@total_ordering
class SpiderIntAction(SpiderGenTokenAction):
  token: str

  def __repr__(self):
    return "IntPrimitive[%s]" % self.token


SPIDER_GEN_TOKEN_ACTION_CLASS_ORDER: Dict[Type[SpiderGenTokenAction], int] = {
    v: k
    for k, v in enumerate(
        (
            SpiderStringAction,
            SpiderTableAction,
            SpiderColumnAction,
            SpiderSingletonAction,
            SpiderObjectAction,
            SpiderIntAction,
        )
    )
}


class SpiderTransitionSystem(TransitionSystem):
  def __init__(
      self,
      asdl_grammar_path: str,
      output_from: bool = False,
      use_table_pointer: bool = False,
      include_literals: bool = True,
      include_columns: bool = True,
  ) -> None:
    super(SpiderTransitionSystem, self).__init__(grammar=ASDLGrammar.from_text(open(asdl_grammar_path).read()))
    self.spider_grammar: SpiderGrammar = SpiderGrammar(
        output_from=output_from,
        use_table_pointer=use_table_pointer,
        include_literals=include_literals,
        include_columns=include_columns,
    )

  def compare_ast(self, hyp_ast: AbstractSyntaxTree, ref_ast: AbstractSyntaxTree):
    raise NotImplementedError

  def ast_to_surface_code(self, asdl_ast: AbstractSyntaxTree) -> dict:
    return asdl_ast_to_dict(self.grammar, asdl_ast)

  def surface_code_to_ast(self, code: dict) -> AbstractSyntaxTree:
    sql_query = self.spider_grammar.parse(code)
    self.spider_grammar.ast_wrapper.verify_ast(sql_query)
    asdl_ast = parsed_sql_query_to_asdl_ast(self.grammar, sql_query)
    asdl_ast.sanity_check()
    return asdl_ast

  def get_valid_continuation_types(self, hyp) -> Tuple[Type[Action], ...]:
      if hyp.tree:
          if self.grammar.is_composite_type(hyp.frontier_field.type):
              if hyp.frontier_field.cardinality == "single":
                  return (ApplyRuleAction,)
              else:  # optional, multiple
                  return ApplyRuleAction, ReduceAction
          else:
              if hyp.frontier_field.cardinality == "single":
                  return (SpiderGenTokenAction,)
              elif hyp.frontier_field.cardinality == "optional":
                  if hyp._value_buffer:
                      return (SpiderGenTokenAction,)
                  else:
                      return SpiderGenTokenAction, ReduceAction
              else:
                  return SpiderGenTokenAction, ReduceAction
      else:
          return (ApplyRuleAction,)

  def get_valid_continuating_productions(self, hyp):
    if hyp.tree:
        if self.grammar.is_composite_type(hyp.frontier_field.type):
            return self.grammar[hyp.frontier_field.type]
        else:
            raise ValueError
    else:
        return self.grammar[self.grammar.root_type]

  # def _tokenize(self, s: str) -> List[str]:
  #     # FIXME: dirty hack
  #     return self.tokenizer.tokenize(s[:-2] if s.endswith(".0") else s)

  def get_gen_token_action(
      self, primitive_type: ASDLPrimitiveType
  ) -> Union[
      Type[SpiderStringAction],
      Type[SpiderTableAction],
      Type[SpiderColumnAction],
      Type[SpiderSingletonAction],
      Type[SpiderObjectAction],
      Type[SpiderIntAction],
  ]:
    if primitive_type.name == "string":
      return SpiderStringAction
    elif primitive_type.name == "table":
      return SpiderTableAction
    elif primitive_type.name == "column":
      return SpiderColumnAction
    elif primitive_type.name == "singleton":
      return SpiderSingletonAction
    elif primitive_type.name == "object":
      return SpiderObjectAction
    elif primitive_type.name == "int":
      return SpiderIntAction
    else:
      raise ValueError("Invalid primitive type `{}`".format(primitive_type))

  def valid_action_predicate(
      self,
      action: Action,
      previous_action: Optional[Action],
      frontier_field: Optional[Field],
      allow_unk: bool,
  ) -> bool:
    """
    Filter function for determining the set of valid actions.
    id: id of action being filtered
    frontier_field: Field to decode
    Returns True if the filtered_action is a valid action, given the frontier field
    """
    # Setting the MASK as always valid allows to avoid NaNs in the output of the softmax
    if action == MaskAction():
      return True
    else:
      if frontier_field is None:
        # First decoding type-step: Only root apply-rule action is valid
        return (
            isinstance(action, ApplyRuleAction)
            and action.production.type == self.grammar.root_type
        )
      else:
        if isinstance(frontier_field.type, ASDLCompositeType):
          if action == ReduceAction():
            # Reduce action: for non-single fields only
            return frontier_field.cardinality != "single"
          elif isinstance(action, ApplyRuleAction):
            # Apply rule: true for productions of correct type
            return action.production.type == frontier_field.type
          else:
            return False
        elif isinstance(frontier_field.type, ASDLPrimitiveType):
          if action == ReduceAction():
            if frontier_field.cardinality == "single":
              # Reduce action: Cannot start a primitive field with a Reduce action
              return isinstance(
                  previous_action,
                  self.get_gen_token_action(
                      primitive_type=frontier_field.type
                  ),
              )
            else:
              return True
          elif action == UnkAction():
            # Unk action: not allowed if the target can be copied from the question
            return allow_unk
          elif frontier_field.type.name in ["column", "table"]:
            # Prevent generating tables and columns from vocabulary. Only copy!
            return False
          else:
            return isinstance(
                action,
                self.get_gen_token_action(
                    primitive_type=frontier_field.type
                ),
            )

  def get_primitive_field_actions(
      self, field: RealizedField
  ) -> List[Union[GenTokenAction, ReduceAction]]:
    assert isinstance(field.type, ASDLPrimitiveType)
    field_actions: List[Union[GenTokenAction, ReduceAction]] = []
    if field.type.name == "int":
      if field.name == "limit" and field.value is None:
        field_actions.append(ReduceAction())
      else:
        field_actions.append(
            self.get_gen_token_action(primitive_type=field.type)(
                token=field.value
            )
        )
        field_actions.append(ReduceAction())
    else:
      field_actions.extend(
          [
              self.get_gen_token_action(primitive_type=field.type)(token=token)
              for token in self._tokenize(str(field.value))
          ]
      )
      field_actions.append(ReduceAction())

    if (
        field.cardinality == "multiple"
        or field.cardinality == "optional"
        and not field_actions
    ):
      # reduce action
      field_actions.append(ReduceAction())

    return field_actions


def parsed_sql_query_to_asdl_ast(
    grammar: ASDLGrammar, query: dict
) -> AbstractSyntaxTree:
  # node should be composite
  production = grammar.get_prod_by_ctr_name(query["_type"])

  def _go(
      field: Field, field_value: Optional[dict]
  ) -> Union[
      None, str, Tuple[str, ...], AbstractSyntaxTree, Tuple[AbstractSyntaxTree, ...],
  ]:
    if field.cardinality == "single" or field.cardinality == "optional":
      if field_value is not None:  # sometimes it could be 0
        if grammar.is_composite_type(field.type):
          value = parsed_sql_query_to_asdl_ast(grammar, query=field_value)
        else:
          value = str(field_value)
      else:
        value = None
    # field with multiple cardinality
    elif field_value is not None:
      if grammar.is_composite_type(field.type):
        value = tuple(
            parsed_sql_query_to_asdl_ast(grammar, query=val)
            for val in field_value
        )
      else:
        value = tuple(str(val) for val in field_value)
    else:
      value = tuple()
    return value

  asdl_node = AbstractSyntaxTree(
      production=production,
      fields=tuple(
          RealizedField(
              name=field.name,
              type=field.type,
              cardinality=field.cardinality,
              value=_go(field, query[field.name] if field.name in query else None),
          )
          for field in production.fields
      ),
  )

  return asdl_node


def asdl_ast_to_dict(grammar: ASDLGrammar, asdl_ast: AbstractSyntaxTree) -> dict:
  query = {"_type": asdl_ast.production.constructor.name}

  for field in asdl_ast.fields:
    field_value = None

    if grammar.is_composite_type(field.type):
      if field.value and field.cardinality == "multiple":
        field_value = []
        for val in field.value:
          node = asdl_ast_to_dict(grammar, asdl_ast=val)
          field_value.append(node)
      elif field.value and field.cardinality in ("single", "optional"):
        field_value = asdl_ast_to_dict(grammar, asdl_ast=field.value)
    else:
      if field.value is not None:
        if field.type.name == "column":
          try:
            field_value = int(field.value)
          except ValueError:
            field_value = 0
        elif field.type.name == "string":
          field_value = field.value
        elif field.type.name == "singleton":
          field_value = field.value.lower() == "true"
        elif field.type.name == "table":
          try:
            field_value = int(field.value)
          except ValueError:
            field_value = 0
        elif field.type.name == "int":
          try:
            field_value = int(field.value)
          except ValueError:
            field_value = 0
        elif field.type.name == "object":
          field_value = field.value
        else:
          raise ValueError("unknown primitive field type")

    if field_value is not None:
      query[field.name] = field_value

  return query


def all_spider_gen_token_actions(
    actions: Tuple[Action, ...],
) -> Generator[SpiderGenTokenAction, None, None]:
  """Iterate over all tokens in primitive actions from sequence of actions."""
  for action in actions:
    if isinstance(action, SpiderGenTokenAction):
      yield action


def a2s(input):
  return "\n".join([repr(i) for i in input])

class DecodeHypothesis(Hypothesis):
  def __init__(self):
      super(DecodeHypothesis, self).__init__()

      self.action_infos = []
      self.code = None

  def clone_and_apply_action_info(self, action_info):
      action = action_info.action

      new_hyp = self.clone_and_apply_action(action)
      new_hyp.action_infos.append(action_info)

      return new_hyp

  def copy(self):
      new_hyp = DecodeHypothesis()
      if self.tree:
          new_hyp.tree = self.tree.copy()

      new_hyp.actions = list(self.actions)
      new_hyp.action_infos = list(self.action_infos)
      new_hyp.score = self.score
      new_hyp._value_buffer = list(self._value_buffer)
      new_hyp.t = self.t
      new_hyp.code = self.code

      new_hyp.update_frontier_info()

      return new_hyp
  
class ActionInfo(object):
  """sufficient statistics for making a prediction of an action at a time step"""

  def __init__(self, action=None):
    self.t = 0
    self.parent_t = -1
    self.action = action
    self.frontier_prod = None
    self.frontier_field = None

    # for GenToken actions only
    self.copy_from_src = False
    self.src_token_position = -1

  def __repr__(self, verbose=False):
    repr_str = '%s (t=%d, p_t=%d, frontier_field=%s)' % (repr(self.action),
                                                         self.t,
                                                         self.parent_t,
                                                         self.frontier_field.__repr__(True) if self.frontier_field else 'None')

    if verbose:
      verbose_repr = 'action_prob=%.4f, ' % self.action_prob
      if isinstance(self.action, SpiderGenTokenAction):
        verbose_repr += 'in_vocab=%s, ' \
                        'gen_copy_switch=%s, ' \
                        'p(gen)=%s, p(copy)=%s, ' \
                        'has_copy=%s, copy_pos=%s' % (self.in_vocab,
                                                      self.gen_copy_switch,
                                                      self.gen_token_prob, self.copy_token_prob,
                                                      self.copy_from_src, self.src_token_position)

      repr_str += '\n' + verbose_repr

    return repr_str


def get_action_infos(src_query, tgt_actions, force_copy=False):
  action_infos = []
  hyp = Hypothesis()
  for t, action in enumerate(tgt_actions):
    action_info = ActionInfo(action)
    action_info.t = t
    if hyp.frontier_node:
      action_info.parent_t = hyp.frontier_node.created_time
      action_info.frontier_prod = hyp.frontier_node.production
      action_info.frontier_field = hyp.frontier_field.field

    if isinstance(action, GenTokenAction):
      try:
        tok_src_idx = src_query.index(str(action.token))
        action_info.copy_from_src = True
        action_info.src_token_position = tok_src_idx
      except ValueError:
        if force_copy:
          raise ValueError('cannot copy primitive token %s from source' % action.token)

    hyp.apply_action(action)
    action_infos.append(action_info)

  return action_infos


def main():
  from preprocess import create_database_schemas, create_training_items, everything
  ts = SpiderTransitionSystem("grammars/Spider2.asdl", output_from=True)
  # # print("FIELDS\n", a2s(ts.grammar.fields))
  # gen_tok_acts = [ts.get_gen_token_action(a)(num) for a in ts.grammar.primitive_types for num in range(10)]
  # # print("ACTIONS?", a2s(ts.grammar.composite_types))

  # schema_lookup = create_database_schemas("spider/tables.json")
  # _, _, train_items, _ = create_training_items(
  #   "spider/train_spider.json", "spider/dev.json", schema_lookup
  # )
  _, _, _, _, vocabs = everything()


if __name__ == "__main__":
  main()
