"""Encoder-decoder Transformer for translation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = residual + x

        # Pre-norm feedforward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x

        return x


class TransformerDecoderLayer(nn.Module):
    """Pre-norm Transformer decoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pre-norm causal self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=False,  # We provide explicit mask
        )
        x = residual + x

        # Pre-norm cross-attention
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = residual + x

        # Pre-norm feedforward
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x

        return x


class TransformerModel(nn.Module):
    """Encoder-decoder Transformer for translation.

    Architecture:
    - Shared source/target vocabulary (joint BPE)
    - Learned positional embeddings
    - 6 encoder layers, 6 decoder layers
    - 8 attention heads (head_dim = d_model // 8)
    - d_ff = 4 * d_model
    - Pre-norm (LayerNorm before attention/FFN)
    - Dropout = 0.0 (hardcoded per paper)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff_multiplier: int = 4,
        max_seq_len: int = 512,
        pad_idx: int = 0,
    ):
        """Initialize Transformer model.

        Args:
            vocab_size: Size of shared vocabulary.
            d_model: Embedding dimension.
            n_layers: Number of encoder and decoder layers.
            n_heads: Number of attention heads.
            d_ff_multiplier: FFN hidden dimension = d_ff_multiplier * d_model.
            max_seq_len: Maximum sequence length for positional embeddings.
            pad_idx: Padding token index.
        """
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        d_ff = d_ff_multiplier * d_model

        # Shared embedding for encoder and decoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection (tied to embedding weights)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get position indices."""
        return torch.arange(seq_len, device=device)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode source sequence.

        Args:
            src: [batch_size, src_len] source token indices.
            src_key_padding_mask: [batch_size, src_len] True where padded.

        Returns:
            [batch_size, src_len, d_model] encoder output.
        """
        batch_size, src_len = src.shape
        device = src.device

        # Embed tokens + positions
        positions = self._get_positions(src_len, device)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        x = self.encoder_norm(x)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode target sequence.

        Args:
            tgt: [batch_size, tgt_len] target token indices.
            memory: [batch_size, src_len, d_model] encoder output.
            tgt_key_padding_mask: [batch_size, tgt_len] True where padded.
            memory_key_padding_mask: [batch_size, src_len] True where padded.

        Returns:
            [batch_size, tgt_len, vocab_size] logits.
        """
        batch_size, tgt_len = tgt.shape
        device = tgt.device

        # Embed tokens + positions
        positions = self._get_positions(tgt_len, device)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)

        # Causal mask
        tgt_mask = self._generate_causal_mask(tgt_len, device)

        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(
                x, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        x = self.decoder_norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)
        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            src: [batch_size, src_len] source token indices.
            tgt: [batch_size, tgt_len] target token indices (input, without last token).

        Returns:
            [batch_size, tgt_len, vocab_size] logits.
        """
        # Create padding masks
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)

        # Encode
        memory = self.encode(src, src_key_padding_mask)

        # Decode
        logits = self.decode(
            tgt, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 128,
        bos_idx: int = 2,
        eos_idx: int = 3,
    ) -> torch.Tensor:
        """Generate translations using greedy decoding.

        Args:
            src: [batch_size, src_len] source token indices.
            max_len: Maximum generation length.
            bos_idx: Beginning of sentence token index.
            eos_idx: End of sentence token index.

        Returns:
            [batch_size, gen_len] generated token indices.
        """
        batch_size = src.shape[0]
        device = src.device

        # Create padding mask and encode
        src_key_padding_mask = (src == self.pad_idx)
        memory = self.encode(src, src_key_padding_mask)

        # Start with BOS token
        tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Decode
            logits = self.decode(
                tgt, memory,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Update finished flag
            finished = finished | (next_token.squeeze(-1) == eos_idx)

            # Append next token
            tgt = torch.cat([tgt, next_token], dim=1)

            # Stop if all sequences are finished
            if finished.all():
                break

        return tgt


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
