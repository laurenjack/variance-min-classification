from dataclasses import dataclass
import torch
from torch import nn, Tensor


VOCAB_SIZE = 151936
HEAD_DIM = 128
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1_000_000

@dataclass
class QwenConfig:
  layers: int = 36
  hidden_size: int = 4096
  intermediate_size: int = 12288
  attn_heads: int = 32
  kv_heads: int = 8
  max_position_emb: int = 40960
  tie_word_embeddings: bool = False


class RMSNorm:

  def __init__(self, d: int):
    self.d = d
    self.weight = nn.Parameter(torch.ones(d))

  def forward(self, x: Tensor):
    var = torch.mean(x ** 2, dim=-1, keepdim=True)
    return self.weight * x / (var + RMS_NORM_EPS) ** 0.5


class Attention:

  def __init__(self, c: QwenConfig):
    self.d = c.hidden_size
    self.h = c.head_dim
    self.num_heads = self.d // HEAD_DIM
    self.attn_W = nn.Linear(self.d, 3 * self.d)
    self.out_W = nn.Linear(self.d, self.d)
    # Q, K norms
    self.q_norm = RMSNorm(HEAD_DIM)
    self.k_norm = RMSNorm(HEAD_DIM)

  def forward(self, x: Tensor):
    b, t, d = x.shape
    # Apply all 3 at once
    x = self.attn_W(x)
    QKV = x[:, :self.d], x[:, self.d : 2*self.d], x[:, 2*self.d:3*self.d]
    # Switch sequence dimension and number of heads for the attention matrix multiply
    Q, K, V = [T.reshape(b, t, self.num_heads, HEAD_DIM).transpose(1, 2) for T in  QKV]
    # sequence length dim on end for Kt
    Kt = K.transpose(2, 3)
    qkt = Q @ Kt / self.h ** 0.5
    # Softmax over the K keys
    A = torch.softmax(qkt, dim=-1) # Shape is [b, num_heads, T, T]
    out = A @ V # Shape is [b, num_heads, T, head_dims]
    out = out.transpose(1, 2)
    # Reshape to concantenate the heads
    out = out.reshape(b, t, d)
    return self.out_W(out)

    


    




class Qwen3:

  def __init__(self, c: QwenConfig):
    self.embeddings = nn.Embedding(VOCAB_SIZE, c.hidden_size)

    
