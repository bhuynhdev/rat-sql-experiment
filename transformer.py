import math
from typing import Iterator, cast

import torch
import torch.nn as nn
from torch.nn import functional as F

import preprocess
from preprocess import BLOCK_SIZE, SRC_VOCAB_SIZE, TGT_SIZE, TGT_VOCAB_SIZE, BATCH_SIZE, DatasetItem
from util import UnpackedSequential

DROP_OUT = 0.2
device = "cpu"
if torch.backends.mps.is_available():
  device = "mps"  # Apple Metal Performance Shader (M1 chip)
if torch.cuda.is_available():
  device = "cuda"
DEVICE = device


class SingleHeadAttention(nn.Module):
  def __init__(self, emb_size: int, head_size: int):
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    self.emb_size = emb_size
    self.head_size = head_size
    # Each head has its own W_Q, W_K, and W_V matrixes for transform the each tok emd to its corresponding q, k, v vectors
    self.query_matrix = nn.Linear(emb_size, head_size, bias=False)
    self.key_matrix = nn.Linear(emb_size, head_size, bias=False)
    self.value_matrix = nn.Linear(emb_size, head_size, bias=False)
    # tril_mask is a static non-learned parameter, so need to use `register_buffer`
    self.register_buffer("tril_mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    self.dropout = nn.Dropout(DROP_OUT)

  def forward(self, x: torch.Tensor, should_mask: bool):
    """
    Parameters:
      x: the input embedding (after summing token_emb and position_emb) --> dimension (B, T, E)
      should_mask: should this Attention block use masked attention (decoder should use mask, encoder shouldn't)
    """
    query = self.query_matrix(x)  # (B, T, D) with D = head_size
    key = self.key_matrix(x)  # (B, T, D)
    value = self.value_matrix(x)  # (B, T, D)
    # `q @ k.T` will produce the itcanitial affinity matrix, basically how strong each query relates to each key
    # dimension: (B, T, D) @ (B, D, T) = (B, T, T)
    # Note The original "Attention is all you need paper" also scales down the affinity scores by multiplying `sqrt(head_size)`
    affinity = (query @ key.transpose(-2, -1)) * (math.sqrt(self.head_size))  # tranpose(-2, -1) avoid transposing the Batch dimension
    if should_mask:
      affinity = affinity.masked_fill(self.tril_mask == 0, float("-inf"))
    weight = F.softmax(affinity, dim=-1)  # (B, T, T)
    weight = self.dropout(weight)
    # The output is the embeddings where each token's embedding have been tweaked
    # to also include information about other related tokens
    out = weight @ value  # (B, T, D)
    return out


class SingleHeadCrossAttention(nn.Module):
  """Cross attention module to be used in the Transformer decoder
  Mostly the same, just that now it also uses the Encoder's embedding to generate the key and value
  """

  def __init__(self, emb_size: int, head_size: int):
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    self.emb_size = emb_size
    self.head_size = head_size
    # Each head has its own W_Q, W_K, and W_V matrixes for transform the each tok emd to its corresponding q, k, v vectors
    self.query_matrix = nn.Linear(emb_size, head_size, bias=False)
    self.key_matrix = nn.Linear(emb_size, head_size, bias=False)
    self.value_matrix = nn.Linear(emb_size, head_size, bias=False)
    # tril_mask is a static non-learned parameter, so need to use `register_buffer`
    self.register_buffer("tril_mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    self.dropout = nn.Dropout(DROP_OUT)

  def forward(self, x1: torch.Tensor, x2: torch.Tensor, should_mask: bool):
    """
    Parameters:
      x1: the first embedding (i.e. the decoder embedding) --> dimension (B, T1, E)
      x2: the second embedding (i.e. the encoder's embedding in the case of Transformer decoder) --> dimension (B, T2, E)
      should_mask: should this Attention block use masked attention (decoder should use mask, encoder shouldn't)
    """
    query = self.query_matrix(x1)  # (B, T1, D) with D = head_size
    key = self.key_matrix(x2)  # (B, T2, D)
    value = self.value_matrix(x2)  # (B, T2, D)
    # `q @ k.T` will produce the affinity matrix, basically how strong each query relates to each key
    # dimension: (B, T1, D) @ (B, D, T2) = (B, T1, T2)
    # Note The original "Attention is all you need paper" also scales down the affinity scores by multiplying `sqrt(head_size)`
    affinity = (query @ key.transpose(-2, -1)) * (math.sqrt(self.head_size))  # tranpose(-2, -1) avoid transposing the Batch dimension
    if should_mask:
      affinity = affinity.masked_fill(self.tril_mask == 0, float("-inf"))
    weight = F.softmax(affinity, dim=-1)  # (B, T1, T2)
    weight = self.dropout(weight)
    # The output is the embeddings where each token's embedding have been tweaked
    # to also include information about other related tokens
    out = weight @ value  # (B, T1, T2) @ (B, T2, D) = (B, T1, D)
    return out


class MultiHeadAttention(nn.Module):
  def __init__(self, emb_size: int, num_head: int, is_masked: bool):
    super().__init__()  # pyright: ignore[reportUnknownMemberType]
    self.emb_size = emb_size
    self.is_masked_attention = is_masked
    # Each head size is emb_size / num_head so that at the end, when we concat all vectors from each head, we still get a vector of emb_size
    self.heads = nn.ModuleList([SingleHeadAttention(emb_size, emb_size // num_head) for _ in range(num_head)])
    self.dropout = nn.Dropout(DROP_OUT)

  def forward(self, x: torch.Tensor):
    out = torch.cat([sa(x, should_mask=self.is_masked_attention) for sa in self.heads], dim=-1)
    out = self.dropout(out)
    return out


class MultiHeadCrossAttention(nn.Module):
  def __init__(self, emb_size: int, num_head: int, is_masked: bool):
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    self.emb_size = emb_size
    self.is_masked_attention = is_masked
    # Each head size is emb_size / num_head so that at the end, when we concat all vectors from each head, we still get a vector of emb_size
    self.heads = nn.ModuleList([SingleHeadCrossAttention(emb_size, emb_size // num_head) for _ in range(num_head)])
    self.dropout = nn.Dropout(DROP_OUT)

  def forward(self, x1: torch.Tensor, x2: torch.Tensor):
    out = torch.cat([sa(x1, x2, should_mask=self.is_masked_attention) for sa in self.heads], dim=-1)
    out = self.dropout(out)
    return out


class PositionWiseFeedForward(nn.Module):
  """After self-attention block is a Feed forward neural net (section 3.3)
  Feed-Forward Layer is a position-wise transformation that consists of linear transformation, ReLU, and another linear transformation.
  https://vaclavkosar.com/ml/Feed-Forward-Self-Attendion-Key-Value-Memory
  """

  def __init__(self, emb_size: int):
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    self.feed_forward = nn.Sequential(
      nn.Linear(emb_size, 4 * emb_size),
      nn.ReLU(),
      nn.Linear(4 * emb_size, emb_size),
      nn.Dropout(DROP_OUT)
    )

  def forward(self, x: torch.Tensor):
    return self.feed_forward(x)


class EncoderBlock(nn.Module):
  """A Transformer Encoder block: A self-attention followed by feedforward net"""

  def __init__(self, emb_size: int, num_attention_heads: int):
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    self.self_attention = MultiHeadAttention(emb_size, num_attention_heads, is_masked=False)
    self.feed_forward = PositionWiseFeedForward(emb_size)
    self.layer_norm1 = nn.LayerNorm(emb_size)  # Layer norm for the self-attention sublayer
    self.layer_norm2 = nn.LayerNorm(emb_size)  # Layer norm for the feed-forward sublayer

  def forward(self, x: torch.Tensor):
    # Addition sicnce additive residual connection
    x = x + self.self_attention(self.layer_norm1(x))
    x = x + self.feed_forward(self.layer_norm2(x))
    return x


class DecoderBlock(nn.Module):
  def __init__(self, emb_size: int, num_attention_heads: int):
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    self.masked_self_attention = MultiHeadAttention(emb_size, num_attention_heads, is_masked=False)
    self.cross_attention = MultiHeadCrossAttention(emb_size, num_attention_heads, is_masked=False)
    self.feed_forward = PositionWiseFeedForward(emb_size)
    self.layer_norm1 = nn.LayerNorm(emb_size)  # Layer norm for the masked-self-attention sublayer
    self.layer_norm2 = nn.LayerNorm(emb_size)  # Layer norm for the cross-attention sublayer
    self.layer_norm3 = nn.LayerNorm(emb_size)  # Layer norm for the feed-forward sublayer

  def forward(self, x: torch.Tensor, encoder_embedding: torch.Tensor):
    x = x + self.masked_self_attention(self.layer_norm1(x))
    x = x + self.cross_attention(self.layer_norm2(x), self.layer_norm2(encoder_embedding))
    x = x + self.feed_forward(self.layer_norm3(x))
    return x, encoder_embedding


# # Positional Encoding
# Use the sine and cosine positional encoding scheme
# Credit: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# Adjusted so that input `x` can be of shape `[batch_size, seq_len, embedding_dim]`
def compute_pos_encoding(block_size: int, d_model: int):
  positions = torch.arange(block_size).unsqueeze(1)
  div_terms = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
  position_encodings = torch.zeros(block_size, d_model, device=DEVICE)  # (B, T, E)
  position_encodings[:, 0::2] = torch.sin(positions * div_terms)
  position_encodings[:, 1::2] = torch.cos(positions * div_terms)
  return position_encodings


class Transformer1(nn.Module):
  def __init__(self, emb_size: int = 256):
    """
    Parameters:
      emb_size: the size of each word embeddings. For example: GloVe embeddings is 300, BERT is 768
    """
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    self.emb_size = emb_size
    # 4 encoder blocks
    self.encoder_layers = nn.Sequential(
      EncoderBlock(emb_size, num_attention_heads=4),
      EncoderBlock(emb_size, num_attention_heads=4),
      EncoderBlock(emb_size, num_attention_heads=4),
      EncoderBlock(emb_size, num_attention_heads=4),
      nn.LayerNorm(emb_size),
    )
    # ENCODER COMPONENTS
    self.encoder_token_emb = nn.Embedding(SRC_VOCAB_SIZE, emb_size)
    # Position embedding table: Convert each token's position in its block to a position embedding
    # Since it is using the sine and cosine positional encoding scheme, it's actually static
    self.register_buffer("positional_embedding_src", compute_pos_encoding(BLOCK_SIZE, emb_size))
    self.register_buffer("positional_embedding_tgt", compute_pos_encoding(TGT_SIZE, emb_size))

    # DECODER COMPONENTS
    # Embedding lookup table: Convert token_ids to that token's corresponding embeddings
    self.decoder_token_emb = nn.Embedding(TGT_VOCAB_SIZE, emb_size)
    # Target language modeling head: Transform back from the embedding dimension to the tgt_vocab_size dimension
    # So that we can get the distribution and know which target token to choose
    self.tgt_lm_head = nn.Linear(self.emb_size, TGT_VOCAB_SIZE)
    self.decoder_layers = UnpackedSequential(
      DecoderBlock(emb_size, num_attention_heads=4),
      DecoderBlock(emb_size, num_attention_heads=4),
      DecoderBlock(emb_size, num_attention_heads=4),
      DecoderBlock(emb_size, num_attention_heads=4),
    )

  def forward(self, input_idx: torch.Tensor, target_idx: torch.Tensor):
    """
    Parameters:
      target_idx: the list of target tokens across the batches. Dimension (B, T)
    """
    # First, encode
    encoder_emb = self.encode(input_idx)  # (B, T, E)
    # Shift right the target_idx to add the <START> token
    start_tokens = torch.ones((BATCH_SIZE, 1), device=DEVICE, dtype=torch.int64)
    shifted_input = torch.cat((start_tokens, target_idx[:, :-1]), dim=1)
    # Now, decode
    y, _ = self.decode(encoder_emb, shifted_input)
    # Convert target embeddings to target probs
    predicted_target_probs = self.tgt_lm_head(y)  # (B, T, C) where C = tgt_vocab_size
    assert predicted_target_probs.shape == (BATCH_SIZE, TGT_SIZE, TGT_VOCAB_SIZE)
    # Cross_entropy requires the "Class" dimension to be the 2nd dimension
    B, T, C = predicted_target_probs.shape
    predicted_target_probs = predicted_target_probs.view(B * T, C)
    target_idx = target_idx.view(B * T)
    # Calculate loss
    loss = F.cross_entropy(predicted_target_probs, target_idx, ignore_index=0)
    return y, loss

  def encode(self, input_batch: torch.Tensor):
    # Input batch is of shape (B, T) (i.e. (batch size, block_size))
    token_emb = self.encoder_token_emb(input_batch)  # (B, T, E) where E=emb_size
    # the position_embedding_table takes input the position of each token in the sequence (i.e. the T dimension)
    position_emb = self.positional_embedding_src  # (T, E)
    x = token_emb + position_emb  # (B, T, E)
    assert x.shape == (BATCH_SIZE, BLOCK_SIZE, self.emb_size), f"Expected {(BATCH_SIZE, BLOCK_SIZE, self.emb_size)}. Got {x.shape}"
    # Feed this x through layers of Transformer Self-Attention blocks
    x = self.encoder_layers(x)
    return x

  def decode(self, encoder_emb: torch.Tensor, target_batch: torch.Tensor):
    # Target batch should be of shape (B, T) (i.e (batch_size, target_size))
    token_emb = self.decoder_token_emb(target_batch)  # (B, T, E) where E=emb_size
    position_emb = self.positional_embedding_tgt  # (T, E)
    y = token_emb + position_emb
    assert y.shape == (BATCH_SIZE, TGT_SIZE, self.emb_size)
    y = self.decoder_layers((y, encoder_emb))  # (B, T, E)
    return y

  def generate(self, input_idx: torch.Tensor, max_generated_tokens: int = 45):
    with torch.no_grad():
      predicted_tokens: list[list[int]] = [list() for _ in range(BATCH_SIZE)]  # Predicted token across the batches
      encoder_emb = self.encode(input_idx)
      # Feed the <START> token as the first chosen token to the entire batch
      # The <START> token has index 1
      target_tokens = torch.ones((BATCH_SIZE, 1), device=DEVICE, dtype=torch.int64)  # (B)
      target_tokens = torch.nn.functional.pad(target_tokens, (0, TGT_SIZE - target_tokens.size(1)), value=0)
      for i in range(max_generated_tokens):
        assert target_tokens.shape == (BATCH_SIZE, TGT_SIZE), f"Expected {(BATCH_SIZE, TGT_SIZE)} got {target_tokens.shape}"
        token_emb = self.decoder_token_emb(target_tokens)  # (B, T, E) where E=emb_size
        position_emb = self.positional_embedding_tgt  # (T, E)
        y = token_emb + position_emb
        assert y.shape == (BATCH_SIZE, TGT_SIZE, self.emb_size)
        y, _ = self.decoder_layers((y, encoder_emb))
        tgt_probs = self.tgt_lm_head(y)  # (B, T, C) where C = tgt_vocab_size
        tgt_probs_this_time_step = tgt_probs[:,i,:] # (B, 1, C)
        chosen_tokens = torch.argmax(tgt_probs_this_time_step, dim=-1)  # (B, 1)
        if i < max_generated_tokens - 1:
          # Add the predicted tokens as new input target tokens to be used to generate next word
          # Note that in the last generatetion, we don't need to add the predicted tokens to the input any more
          target_tokens[:, i + 1] = chosen_tokens

        for j, chosen_token in enumerate(chosen_tokens):
          predicted_tokens[j].append(int(chosen_token.item()))
      return predicted_tokens

def main():
  # # Train
  m1 = Transformer1()
  print(f"Parameter count: {sum(dict((p.data_ptr(), p.numel()) for p in m1.parameters()).values())}") # https://stackoverflow.com/a/62764464
  m1 = m1.to(DEVICE)

  print("BOOTSTRAPPING THE DATALOADER")
  _, _, train_dataloader, _, token_lookup_tables = preprocess.everything()

  m1.eval()
  train_dataloader_iter = cast(Iterator[DatasetItem], iter(train_dataloader))
  batch = next(train_dataloader_iter)

  input, target = batch # (B, block_size)
  print("SAMPLE GENERATING")
  print("Sample input", preprocess.toks_decode(input.tolist()[0], token_lookup_tables, "source")) # type: ignore
  print("Sample target", preprocess.toks_decode(target.tolist()[0], token_lookup_tables, "target")) # type: ignore

  y_batch = m1.generate(input.to(DEVICE))
  predicted = preprocess.toks_decode(y_batch[0], token_lookup_tables, "target")
  print(predicted)

  MODEL_NAME = "transform3"
  
  
  print("START TRAINING")
  # Train the network
  # Create an optimizer
  m1.train()
  optimizer = torch.optim.AdamW(m1.parameters(), lr=5e-5)

  for epoch in range(20000):
    try:
      batch = next(train_dataloader_iter)
    except StopIteration:
      # Reset the dataloader
      train_dataloader_iter = iter(train_dataloader)
      batch = next(train_dataloader_iter)
    input = batch[0].to(DEVICE)  # (B, block_size)
    target = batch[1].to(DEVICE)  # (B, TGT_SIZE)
    _, loss = m1(input, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(epoch, loss)

  # After training
  # Save the model
  print("TRAINING FINISHED. SAVING THE MODEL")
  torch.save(m1.state_dict(), f"{MODEL_NAME}.pt") # pyright: ignore[reportUnknownMemberType]

  # Inference
  m1_trained = Transformer1()
  m1_trained = m1_trained.to(DEVICE)
  m1_trained.load_state_dict(torch.load(f"{MODEL_NAME}.pt")) # pyright: ignore[reportUnknownMemberType]
  m1_trained.eval()

  batch = next(train_dataloader_iter)

  input_seq, target_seq = batch
  y_batch = m1_trained.generate(input_seq.to(DEVICE), max_generated_tokens=TGT_SIZE)

  # Print result to cmd (just first 2 batch)
  for i in range(2):
    print("Inference: Input words", preprocess.toks_decode(input_seq.tolist()[i], token_lookup_tables, "source")) # type: ignore
    print("Inference: Target tokens", target_seq.tolist()[i])  # type: ignore
    print("Inference: Target words", preprocess.toks_decode(target_seq.tolist()[i], token_lookup_tables, "target"))  # type: ignore
    print("Inference: Output tokens", y_batch[i])
    print("Inference: Output words", preprocess.toks_decode(y_batch[i], token_lookup_tables, "target"))

  # Write result as txt
  with open(f"{MODEL_NAME}-result.txt", "w") as f:
    for i in range(BATCH_SIZE):
      f.write(f"{i}\n")
      f.write(f"input_words: {preprocess.toks_decode(input_seq[i, :].tolist(), token_lookup_tables, "source")}\n") # type: ignore
      f.write(f"target_words: {preprocess.toks_decode(target_seq[i, :].tolist(), token_lookup_tables, "target")}\n") # type: ignore
      f.write(f"target_tokens: {target_seq[i, :].tolist()}\n") # type: ignore
      f.write(f"predicted_words: {preprocess.toks_decode(y_batch[i], token_lookup_tables, "target")}\n") # type: ignore
      f.write(f"predicted_tokens: {y_batch[i]}\n")

if __name__ == "__main__":
  preprocess.download_spider_zip()
  main()
