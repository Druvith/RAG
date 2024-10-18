import torch
from torch import nn
import math
import transformers
from transformers import BertTokenizer, BertModel as HFBertModel
import time
import os
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embeddings(nn.Module):
    """Constructs the embeddings for given input tensor (after tokenization)."""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)    # (Vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)   # (Block_size, hidden_dim)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x: torch.Tensor):
        """Forward pass for embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S), where B is batch size and S is sequence length.

        Returns:
            torch.Tensor: Embeddings of shape (B, S, D), where D is the hidden size.
        """
        seq_length = x.size(1)
        assert x.max() < self.word_embeddings.num_embeddings, "Input contains token IDs outside the vocab range."
        assert x.min() >= 0, "Input contains negative token IDs."

        # Create position IDs and token type IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        token_type_ids = torch.zeros_like(x, dtype=torch.long, device=x.device)

        # Get embeddings
        word_embeddings = self.word_embeddings(x)  # (B, S, D)
        position_embeddings = self.position_embeddings(position_ids)  # (B, S, D)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # (B, S, D)

        # Sum the embeddings and apply layer normalization and dropout
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)  # (B, S, D)
    
class Attention(nn.Module):
  """ Vanilla multi-head attention without attention mask (since BERT is bidirection Encoder only transformer) """
  def __init__(self, config):
    super().__init__()
    self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
    self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
    self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
    assert config.hidden_size % config.num_attention_heads == 0
    self.head_size = config.hidden_size // config.num_attention_heads
    self.num_attention_heads = config.num_attention_heads
    self.dropout = nn.Dropout(0.1)
  def forward(self, x: torch.Tensor):
    bz, seq_len, _ = x.size()
    query = self.query(x)
    key = self.key(x)
    value = self.value(x)
    q = query.view(bz, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)   # size -> (B, nh, S, hz)
    k = key.view(bz, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)     # size -> (B, nh, S, hz)
    v = value.view(bz, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)   # size -> (B, nh, S, hz)
    att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_size)
    att = nn.functional.softmax(att, dim=-1)  # size -> (B, nh, S, S)
    att = att @ v  # size -> (B, nh, S, hz)
    outputs = att.transpose(1,2).contiguous().view(bz, seq_len, _)  # size -> (B, S, hidden_size)
    return outputs
  
class AttentionOutput(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)
    self.dropout = nn.Dropout(0.1)
  def forward(self, attention_outputs, x):
    """ Add and norm part of the encoder architecture """
    hidden_states = self.dense(attention_outputs)       # size -> (B, S, D)
    hidden_states = self.LayerNorm(hidden_states + x)
    return self.dropout(hidden_states)
  
class Intermediate(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
    self.intermediate_act_fn = nn.GELU()

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    x = self.intermediate_act_fn(hidden_states)
    return x        #shape: (B,S, I_size)
  
class BertOutput(nn.Module):        # Output MLP + Layer Norm of BERT Encoder
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)
    self.intermediate_act_fn = nn.GELU()
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    hidden_states = self.intermediate_act_fn(self.dense(x))     # (B, S, H)
    hidden_states = self.LayerNorm(hidden_states)
    return self.dropout(hidden_states)
  
class BertAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attention = Attention(config)
    self.output = AttentionOutput(config)

  def forward(self, hidden_states):
    """ Concatenation of BERT Attention and attention outputs (replication from the BERT architecture)"""
    attention_outputs = self.attention(hidden_states)
    attention_outputs = self.output(attention_outputs, hidden_states)
    return attention_outputs
  
class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attention = BertAttention(config)
    self.intermediate = Intermediate(config)
    self.output = BertOutput(config)

  def forward(self, hidden_states):
    """ BERT Encoder """
    attention_outputs = self.attention(hidden_states)
    intermediate_outputs = self.intermediate(attention_outputs)
    layer_outputs = self.output(intermediate_outputs)
    return layer_outputs
  
class BertPooler(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
    self.dropout = nn.Dropout(0.1)
    self.activation = nn.Tanh()
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    first_token_tensor = hidden_states[:, 0]
    return self.activation(self.dropout(self.dense(first_token_tensor)))    # size: (B, 1, dim)


class Config:
  """ L(number of transformer blocks) = 12,H (Hidden dim) = 768, A(Attention heads) = 12, total params = 110M"""
  vocab_size: int = 28996
  hidden_size: int = 1024
  num_hidden_layers: int = 24
  num_attention_heads: int = 16
  intermediate_size: int = 4096
  max_position_embeddings: int = 512
  type_vocab_size: int = 2        # 0 : Question, 1: Answer. (Not required for our application)
  layer_norm_eps: float = 1e-12   # to avoid division by 0

class CustomBertModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embeddings = Embeddings(config)
    self.encoder = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    self.pooler = BertPooler(config)

  def forward(self, x):
    x = self.embeddings(x)
    for layer in self.encoder:
      """ each layer of encoder """
      x = layer(x)
      output = self.pooler(x)
    return output

  def device(self, device):
    return next(self.parameters()).device()

  @classmethod
  def from_pretrained(cls, config, device):
    """Load pre-trained BERT-base weights from Hugging Face, excluding the pooler."""
    from transformers import BertModel as HFBertModel
    import torch

    logger.info("Starting to load pre-trained Hugging Face model...")
    model_hf = HFBertModel.from_pretrained('google-bert/bert-large-cased')
    logger.info("Pre-trained BERT-base model loaded successfully.")

    # Get the state dictionary
    sd_hf = model_hf.state_dict()
    logger.debug(f"Original state dict keys: {list(sd_hf.keys())[:10]}...")

    # Create an instance of the custom model
    model = cls(config)
    logger.info("Custom model instance created.")

    # Log the current device
    logger.info(f"Loading weights with strict=False on device: {device}.")

    # Load the filtered state dictionary with strict=False
    try:
        model.load_state_dict(sd_hf, strict=False)
        logger.info("Weights loaded successfully with strict=False.")
    except RuntimeError as e:
        logger.error(f"Error while loading state dict: {e}")
        raise

    # Log model parameters for debugging
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")

    # Move the model to the specified device with logging
    try:
        logger.info(f"Moving model to {device}...")
        model = model.to(device)
        logger.info(f"Model moved to {device} successfully.")
    except RuntimeError as e:
        logger.error(f"Error moving model to {device}: {e}")
        raise

    return model


