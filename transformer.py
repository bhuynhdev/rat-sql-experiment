import torch
import torch.nn as nn
import math
import preprocess
from typing import Iterator, Union, cast
from preprocess import BLOCK_SIZE, TGT_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, DatasetItem
from torch.nn import functional as F

DROP_OUT = 0.2
BATCH_SIZE = 64
device = "cpu"
if torch.backends.mps.is_available():
  device = "mps"  # Apple Metal Performance Shader (M1 chip)
if torch.cuda.is_available():
  device = "cuda"
DEVICE = device


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


class MultiHeadAttention(nn.Module):
  def __init__(self, emb_size: int, num_head: int, is_masked: bool):
    super().__init__()
    self.emb_size = emb_size
    self.is_masked_attention = is_masked
    # Each head size is emb_size / num_head so that at the end, when we concat all vectors from each head, we still get a vector of emb_size
    self.heads = nn.ModuleList([SingleHeadAttention(emb_size, emb_size // num_head) for _ in range(num_head)])
    self.dropout = nn.Dropout(DROP_OUT)

  def forward(self, x: torch.Tensor):
    out = torch.cat([sa(x, should_mask=self.is_masked_attention) for sa in self.heads], dim=-1)
    out = self.dropout(out)
    return out


class PositionWiseFeedForward(nn.Module):
  """After self-attention block is a Feed forward neural net (section 3.3)
  Feed-Forward Layer is a position-wise transformation that consists of linear transformation, ReLU, and another linear transformation.
  https://vaclavkosar.com/ml/Feed-Forward-Self-Attendion-Key-Value-Memory
  """

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


class EncoderBlock(nn.Module):
  """A Transformer Encoder block: A self-attention followed by feedforward net"""

  def __init__(self, emb_size: int, num_attention_heads: int):
    super().__init__()
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
    super().__init__()
    self.masked_self_attention = MultiHeadAttention(emb_size, num_attention_heads, is_masked=False)
    self.cross_attention = MultiHeadAttention(emb_size, num_attention_heads, is_masked=False)
    self.feed_forward = PositionWiseFeedForward(emb_size)
    self.layer_norm1 = nn.LayerNorm(emb_size)  # Layer norm for the masked-self-attention sublayer
    self.layer_norm2 = nn.LayerNorm(emb_size)  # Layer norm for the cross-attention sublayer
    self.layer_norm3 = nn.LayerNorm(emb_size)  # Layer norm for the feed-forward sublayer

  def forward(self, x: torch.Tensor):
    x = x + self.masked_self_attention(self.layer_norm1(x))
    x = x + self.cross_attention(self.layer_norm2(x))
    x = x + self.feed_forward(self.layer_norm3(x))
    return x


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
    super().__init__()
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
    self.register_buffer("positional_embedding", compute_pos_encoding(BLOCK_SIZE, emb_size))

    # DECODER COMPONENTS
    self.decoder_hidden_state_size = 2 * emb_size
    # Embedding lookup table: Convert token_ids to that token's corresponding embeddings
    self.decoder_token_emb = nn.Embedding(TGT_VOCAB_SIZE, emb_size)
    # Target language modeling head: Transform back from the embedding dimension to the tgt_vocab_size dimension
    # So that we can get the distribution and know which target token to choose
    self.tgt_lm_head = nn.Linear(self.decoder_hidden_state_size, TGT_VOCAB_SIZE)
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
    x = self.encode(input_idx)  # (B, T, E)
    # Average the last hidden state of the encoder as the context
    context_emb = torch.mean(x, dim=1)  # (B, E)
    # Feed the <START> token as the first chosen token to the entire batch
    # The <START> token has index 1
    chosen_tokens = torch.ones(x.size(0), device=DEVICE, dtype=torch.int64)  # (B)
    ys: list[torch.Tensor] = []
    # Initialize the first hidden state as 0s
    hidden_state = torch.zeros((BATCH_SIZE, self.decoder_hidden_state_size), device=DEVICE)  # (B, H) where H = dec_hidden_state
    for i in range(TGT_SIZE):
      # hidden_state: (B, H)
      # tgt_probs: (B, C) where C = tgt_vocab_size
      hidden_state, tgt_probs = self.decode(context_emb, input_tokenIds=chosen_tokens, prev_hidden_state=hidden_state)
      ys.append(tgt_probs)
      if target_idx is not None:
        # Teacher forcing
        chosen_tokens = target_idx[:, i]
      else:
        # Greedily select the token with highest prob from the distribution
        chosen_tokens = torch.argmax(tgt_probs, dim=1)  # (B)

    # Note that ys is collected by looping over max_len_target, so when stacked, the first dimension is max_len_target
    y = torch.stack(ys)  # (T, B, C) where C = tgt_vocab_size
    assert y.shape == (TGT_SIZE, BATCH_SIZE, TGT_VOCAB_SIZE)
    if target_idx is None:
      return ys, None
    # Cross_entropy requires the "Class" dimension to be the 2nd dimension
    T, B, C = y.shape
    y = y.view(B * T, C)
    target_idx = target_idx.view(B * T)
    # Calculate loss
    loss = F.cross_entropy(y, target_idx, ignore_index=0)
    return y, loss

  def encode(self, input_batch: torch.Tensor):
    # Input batch is of shape (B, T) (i.e. (batch size, block_size))
    token_emb = self.encoder_token_emb(input_batch)  # (B, T, E) where E=emb_size
    # the position_embedding_table takes input the position of each token in the sequence (i.e. the T dimension)
    position_emb = self.positional_embedding  # (T, E)
    x = token_emb + position_emb  # (B, T, E)
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
    temp1 = self.decoder_context_linear(context_emb)  # (B, dec_hidden_size)
    temp2 = self.decoder_hiddenstate_linear(prev_hidden_state)  # (B, dec_hiden_size)
    tok_emb = self.decoder_token_emb(input_tokenIds)  # (B, E)
    temp3 = self.decoder_token_emb_linear(tok_emb)  # (B, dec_hidden_size)
    z = temp1 + temp2 + temp3  # (B, dec_hidden_size)
    hidden_state = torch.tanh(z)  # (B, dec_hidden_size)
    assert hidden_state.shape == (BATCH_SIZE, self.decoder_hidden_state_size)

    tgt_distribution: torch.Tensor = self.tgt_lm_head(hidden_state)  # (B, tgt_vocab_size)
    # Do NOT run softmax here as the Pytorch Cross Entropy Loss function expects unnormalized numbers
    # tgt_probs = F.softmax(tgt_distribution, dim=-1) # (B, tgt_vocab_size)
    return hidden_state, tgt_distribution

  def generate(self, input_idx: torch.Tensor, max_generated_tokens: int = 20):
    with torch.no_grad():
      encoder_last_hidden_state = self.encode(input_idx)  # (B, T, E)
      # Average all input tokens embs across the encoder last hidden state as the context
      context_emb = torch.mean(encoder_last_hidden_state, dim=1)  # (B, E)
      # Feed the <START> token as the first chosen token to the entire batch
      # The <START> token has index 1
      chosen_tokens = torch.ones(input_idx.size(0), device=DEVICE, dtype=torch.int64)  # (B)
      predicted_tokens: list[list[int]] = [list() for _ in range(BATCH_SIZE)] # Predicted token across the batches
      # Initialize the first hidden state as 0s
      hidden_state = torch.zeros((BATCH_SIZE, self.decoder_hidden_state_size), device=DEVICE)  # (B, H) where H = dec_hidden_state
      for _ in range(max_generated_tokens):
        # hidden_state: dimension (B, H)
        # tgt_probs: dimension (B, tgt_vocab_size)
        hidden_state, tgt_probs = self.decode(context_emb, chosen_tokens, prev_hidden_state=hidden_state)
        # Greedily select the token with highest prob from the distribution
        chosen_tokens = torch.argmax(tgt_probs, dim=1)  # (B)
        for i, chosen_token in enumerate(chosen_tokens):
          predicted_tokens[i].append(int(chosen_token.item()))
    return predicted_tokens


def main():
  # # Train
  # m1 = Transformer1()
  # m1 = m1.to(DEVICE)

  # # print("BOOTSTRAPPING THE DATALOADER")
  _, _, train_dataloader, _, token_lookup_tables = preprocess.everything()

  # # m1.eval()
  train_dataloader_iter = cast(Iterator[DatasetItem], iter(train_dataloader))
  # batch = next(train_dataloader_iter)

  # input, target = batch # (B, block_size)
  # print("SAMPLE GENERATING")
  # print("Sample input", preprocess.toks_decode(input.tolist()[0], token_lookup_tables, "source"))
  # print("Sample target", preprocess.toks_decode(target.tolist()[0], token_lookup_tables, "target"))

  # first_batch_predicted_tokens = m1.generate(input.to(DEVICE))
  # predicted = preprocess.toks_decode(first_batch_predicted_tokens, token_lookup_tables, "target")
  # print(predicted)

  # # Train the network
  # # Create an optimizer
  # m1.train()
  # optimizer = torch.optim.AdamW(m1.parameters(), lr=5e-4)

  # print("START TRAINING")

  # for epoch in range(100):
  #   try:
  #     batch = next(train_dataloader_iter)
  #   except StopIteration:
  #     # Reset the dataloader
  #     train_dataloader_iter = iter(train_dataloader)
  #     batch = next(train_dataloader_iter)
  #   input = batch[0].to(DEVICE)  # (B, block_size)
  #   target = batch[1].to(DEVICE)  # (B, TGT_SIZE)
  #   _, loss = m1(input, target)
  #   optimizer.zero_grad(set_to_none=True)
  #   loss.backward()
  #   optimizer.step()
  #   print(epoch, loss)

  # # After training
  # # Save the model
  # torch.save(m1.state_dict(), "transform2.pt")

  # Inference
  m1_trained = Transformer1()
  m1_trained = m1_trained.to(DEVICE)
  m1_trained.load_state_dict(torch.load("transform2.pt"))
  m1_trained.eval()

  batch = next(train_dataloader_iter)

  input_seq, target_seq = batch
  print("Inference: Input words", preprocess.toks_decode(input_seq.tolist()[0], token_lookup_tables, "source"))
  print("Inference: Target tokens", target_seq.tolist()[0])
  print("Inference: Target words", preprocess.toks_decode(target_seq.tolist()[0], token_lookup_tables, "target"))

  y_batch = m1_trained.generate(input_seq.to(DEVICE), max_generated_tokens=TGT_SIZE)
  for y in y_batch:
    print("Inference: Output tokens", y)
    print("Inference: Output words", preprocess.toks_decode(y, token_lookup_tables, "target"))


if __name__ == "__main__":
  preprocess.download_spider_zip()
  main()
