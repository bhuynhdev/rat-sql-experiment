from typing import Any
import torch
from preprocess import AllVocabs, ModelInput
from grammars.spider_transition_system import ActionInfo, DecodeHypothesis, SpiderTransitionSystem, SpiderGenTokenAction, SpiderTableAction, SpiderColumnAction, SpiderIntAction, SpiderObjectAction, SpiderSingletonAction, SpiderStringAction
from grammars.transition_system import ApplyRuleAction, ReduceAction
from transformer_tree_decoder import Transformer1, DEVICE
from preprocess import TGT_SIZE, PRIMITIVE_IDX_TO_TOK
import torch.nn.functional as F
import re


def beam_search(model: Transformer1, transition_system: SpiderTransitionSystem, batch: ModelInput, question: str, vocabs: AllVocabs, col_map: dict[int, int], tbl_map: dict[int, int], beam_size: int = 2, decode_max_time_step: int = 1000):
  """Given an input NL query, use information from the model to return a list of potential hypothesis
  Parameters:
  - col_map: Map from the position in the input to a table

  """
  new_hyp_meta: list[Any] = []
  t = 0
  hypotheses = [DecodeHypothesis()]
  hyp_states = [[]]
  completed_hypotheses: list[DecodeHypothesis] = []
  batch_size = batch.input.size(0)
  
  target_seq = batch.target
  frontier_field_seq = batch.field_idx
  frontier_type_seq = batch.type_idx

  # Shift right the target_idx to add the <START> token
  start_tokens = torch.ones((batch_size, 1), device=DEVICE, dtype=torch.int64)
  start_frontier_field_tokens = torch.zeros((batch_size, 1), device=DEVICE, dtype=torch.int64)  # 0s for None frontier field
  start_frontier_type_tokens = torch.zeros((batch_size, 1), device=DEVICE, dtype=torch.int64)  # 0s for None frontier type
  dec_input = torch.cat((start_tokens, target_seq[:, :-1]), dim=1)
  frontier_field_dec_input = torch.cat((start_frontier_field_tokens, frontier_field_seq[:, :-1]), dim=1)
  frontier_type_dec_input = torch.cat((start_frontier_type_tokens, frontier_type_seq[:, :-1]), dim=1)
  
  encoder_emb = model.encode(batch.input)

  while len(completed_hypotheses) < beam_size and t < decode_max_time_step:
    hyp_num = len(hypotheses)

    # Create a batch from the hypotheses
    new_dec_inputs: list[torch.Tensor] = []
    new_field_inputs: list[torch.Tensor] = []
    new_type_inputs: list[torch.Tensor] = []
    for _, hyp in enumerate(hypotheses):
      action_tm1 = hyp.actions[-1]
      chosen_action_token = vocabs.action_vocab.encode([repr(action_tm1)])[0]
      chosen_frontier_field_token = transition_system.grammar.field2id[hyp.frontier_field.field] + 1
      chosen_frontier_type_token = transition_system.grammar.type2id[hyp.frontier_field.type] + 1
      new_dec_input = dec_input.clone()
      new_dec_input[:, t + 1] = chosen_action_token
      new_field_input = frontier_field_dec_input.clone()
      new_field_input[:, t + 1] = chosen_frontier_field_token
      new_type_input = frontier_type_dec_input.clone()
      new_type_input[:, t + 1] = chosen_frontier_type_token
      new_dec_inputs.append(new_dec_input)
      new_field_inputs.append(new_field_input)
      new_type_inputs.append(new_type_input)

    dec_input = torch.stack(new_dec_inputs)
    frontier_field_dec_input = torch.stack(new_field_inputs)
    frontier_type_dec_input = torch.stack(new_type_inputs)

    decoder_action_emb = model.decoder_token_emb(dec_input)  # (B, T, E)
    decoder_pos_emb = model.positional_embedding_tgt  # (B, T, E)
    decoder_field_emb = model.decoder_field_emb(frontier_field_dec_input)
    decoder_type_emb = model.decoder_type_emb(frontier_type_dec_input)
    decoder_input = torch.cat((decoder_action_emb + decoder_pos_emb, decoder_field_emb, decoder_type_emb), dim=-1)

    assert decoder_input.shape == (batch_size, TGT_SIZE, model.decoder_hidden_size)
    # "memory" is bascially encoder_ouput that is remembered/cached/re-used throughout all decoder calculation
    decoder_output, memory = model.decode(encoder_emb, decoder_input)

    # Convert target embeddings to probability of taking each action in the vocab
    action_probs: torch.Tensor = model.tgt_lm_head(decoder_output)  # (B, T, C) where C = tgt_vocab_size
    action_probs = F.log_softmax(action_probs, dim=-1)  # (B, T, C)
    primitive_probs: torch.Tensor = model.primitive_lm_head(decoder_output)  # (B, T, 7)
    primitive_probs = F.log_softmax(primitive_probs, dim=-1)

    copy_logits: torch.Tensor = model.pointer_network(query=decoder_output, keys=memory)  # (B, TGT_SIZE, BLOCK_SIZE)
    copy_probs = F.log_softmax(copy_logits, dim=-1)
        
    # For each current hypothesis, generate a new hypothesis stemming from it
    for hyp_id, hyp in enumerate(hypotheses):
      next_action_types = transition_system.get_valid_continuation_types(hyp)
      for action_type in next_action_types:
        if action_type == ApplyRuleAction:
          next_productions = transition_system.get_valid_continuating_productions(hyp)
          for production in next_productions:
            # Convert production to token id
            action_id = vocabs.action_vocab.encode([repr(ApplyRuleAction(production))])[0]
            prod_score = action_probs[:, hyp_id, action_id].item()
            new_hyp_score = hyp.score + prod_score

            meta_entry = {'action_type': 'apply_rule', 'prod_id': action_id,
                          'score': prod_score, 'new_hyp_score': new_hyp_score,
                          'prev_hyp_id': hyp_id}
            new_hyp_meta.append(meta_entry)
        elif action_type == ReduceAction:
          action_id = vocabs.action_vocab.encode([repr(ReduceAction())])[0]
          prod_score = action_probs[:, hyp_id, action_id].item()
          new_hyp_score = hyp.score + prod_score

          meta_entry = {'action_type': 'apply_rule', 'prod_id': action_id,
                        'score': prod_score, 'new_hyp_score': new_hyp_score,
                        'prev_hyp_id': hyp_id}
        elif action_type == SpiderGenTokenAction:
          # See which primitive to use - get top 4
          # For each primitive, determine which token to copy, using the copy logits - get top 2
          top_k_primitives = torch.topk(primitive_probs[:, hyp_id], dim=-1, k=4)
          for i, primitive_id in enumerate(top_k_primitives.indices[0, :]):
            primitive_id = primitive_id.item()
            primitive_score = top_k_primitives.values[0, i]
            top_k_copy_pos = torch.topk(copy_probs[0, hyp_id], dim=-1, k=2)
            for copy_pos in top_k_copy_pos:
              meta_entry = {'action_type': 'gen_token', 'primitive_id': primitive_id, 'token_pos': copy_pos,
                            'score': primitive_score, 'new_hyp_score': hyp.score + primitive_score,
                            'prev_hyp_id': hyp_id}

    new_hyp_scores = torch.cat([x['new_hyp_score'] for x in new_hyp_meta])
    top_new_hyp_scores, meta_ids = torch.topk(new_hyp_scores, k=min(new_hyp_scores.size(0), beam_size - len(completed_hypotheses)))

    live_hyp_ids = []
    new_hypotheses = []
    # Convert the hypothesis metadatas into the tree
    for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
      action_info = ActionInfo()
      hyp_meta_entry = new_hyp_meta[meta_id]
      prev_hyp_id: int = hyp_meta_entry['prev_hyp_id']
      prev_hyp = hypotheses[prev_hyp_id]

      action_type_str = hyp_meta_entry['action_type']
      if action_type_str == 'apply_rule':
          # ApplyRule action
        prod_id = hyp_meta_entry['prod_id']
        production_name = vocabs.action_vocab.decode([prod_id])[0]  # Will be of format "ApplyRule[<inner_production>]" or "Reduce"
        if production_name == "Reduce":
          action = ReduceAction()
        else:
          inner_production_name = re.findall(r"\[(.*?)\]", production_name)[0]
          # Convert to name to the actual production object
          production = transition_system.grammar.get_prod_by_ctr_name(inner_production_name)
          action = ApplyRuleAction(production)
      else:  # action_type_str == 'gen_token':
          # Use the primitive id to determine what kind of SpiderGenToken there are
        spider_gen_token_type = PRIMITIVE_IDX_TO_TOK[hyp_meta_entry['primitive_id']]
        if spider_gen_token_type == "TablePrimitive":
          table = tbl_map[hyp_meta_entry['token_pos']]
          action = SpiderTableAction(table)
        elif spider_gen_token_type == "ColumnPrimitive":
          col = col_map[hyp_meta_entry['token_pos']]
          action = SpiderColumnAction(col)
        elif spider_gen_token_type == "StringPrimitive":
          str = question[hyp_meta_entry['token_pos']]
          action = SpiderStringAction(str)
        elif spider_gen_token_type == "ObjectPrimitive":
          str = question[hyp_meta_entry['token_pos']]
          action = SpiderObjectAction(str)
        else:
          action = SpiderStringAction("Unknon")
        if 'token_pos' in hyp_meta_entry:
          action_info.copy_from_src = True
          action_info.src_token_position = hyp_meta_entry['token_pos']

      action_info.action = action
      action_info.t = t

      if t > 0:
        action_info.parent_t = prev_hyp.frontier_node.created_time
        action_info.frontier_prod = prev_hyp.frontier_node.production
        action_info.frontier_field = prev_hyp.frontier_field.field

      new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
      new_hyp.score = new_hyp_score

      if new_hyp.completed:
        completed_hypotheses.append(new_hyp)
      else:
        new_hypotheses.append(new_hyp)
        live_hyp_ids.append(prev_hyp_id)

    if live_hyp_ids:
      hypotheses = new_hypotheses
      t += 1
    else:
      break
