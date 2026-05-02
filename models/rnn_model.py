"""
models/rnn_model.py
===================
Basic (Elman) RNN implementations for:
  • Task 1 – Language modelling / text generation   (RNNLanguageModel)
  • Task 2 – Machine translation via Seq2Seq        (RNNSeq2Seq)

Architecture notes
------------------
RNN cells suffer from vanishing gradients over long sequences; they serve as
the baseline against which the LSTM is compared.  Gradient clipping (config.py
GRAD_CLIP) is essential for stable training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import random

import config as C


# ═══════════════════════════════════════════════════════════════════════════════
# Shared embedding builder
# ═══════════════════════════════════════════════════════════════════════════════

def _make_embedding(pretrained_matrix: Optional[Tensor],
                    vocab_size: int,
                    embed_type: str) -> Tuple[nn.Embedding, int]:
    """
    Build an nn.Embedding from a pretrained matrix or from scratch.

    Returns (embedding_layer, embed_dim).
    One-hot embeddings are frozen (no gradients); Word2Vec embeddings are
    fine-tuned by default.
    """
    if pretrained_matrix is not None:
        embed_dim = pretrained_matrix.shape[1]
        emb = nn.Embedding.from_pretrained(
            pretrained_matrix.float(),
            freeze=(embed_type == "onehot"),
            padding_idx=C.SPECIAL_TOKENS.index(C.PAD_TOKEN),
        )
    else:
        embed_dim = C.W2V_EMBED_DIM
        emb = nn.Embedding(vocab_size, embed_dim,
                           padding_idx=C.SPECIAL_TOKENS.index(C.PAD_TOKEN))
    return emb, embed_dim


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1 – RNN Language Model
# ═══════════════════════════════════════════════════════════════════════════════

class RNNLanguageModel(nn.Module):
    """
    Multi-layer Elman RNN language model.

    Forward pass
    ------------
    Input  : (batch, seq_len)  token indices
    Output : (batch, seq_len, vocab_size)  logits over vocabulary
    Hidden : (num_layers, batch, hidden_dim)

    The embedding layer is shared with the output projection (weight tying)
    when embed_dim == hidden_dim and embedding_type == 'word2vec'.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_matrix: Optional[Tensor] = None,
                 embed_type: str = C.EMBEDDING_TYPE,
                 hidden_dim: int = C.HIDDEN_DIM,
                 num_layers: int = C.NUM_LAYERS,
                 dropout: float = C.DROPOUT):
        super().__init__()
        self.embedding, embed_dim = _make_embedding(embed_matrix, vocab_size,
                                                    embed_type)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, vocab_size)

        # Weight tying: share embedding ↔ output projection weights when dims match
        if embed_dim == hidden_dim and embed_type == "word2vec":
            self.fc.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, p in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.zeros_(self.fc.bias)

    def forward(self,
                src: Tensor,
                hidden: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        # src: (batch, seq_len)
        emb = self.drop(self.embedding(src))           # (B, T, E)
        out, hidden = self.rnn(emb, hidden)            # (B, T, H), (L, B, H)
        out = self.drop(out)
        logits = self.fc(out)                          # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(self.rnn.num_layers, batch_size,
                           self.rnn.hidden_size, device=device)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2 – RNN Seq2Seq (Encoder + Decoder + wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class RNNEncoder(nn.Module):
    """
    Encodes a source sequence into a context vector (final hidden state).

    Input  : (batch, src_len)
    Output : (batch, src_len, hidden_dim),  hidden (num_layers, batch, H)
    """

    def __init__(self,
                 src_vocab_size: int,
                 src_embed_matrix: Optional[Tensor] = None,
                 embed_type: str = C.EMBEDDING_TYPE,
                 hidden_dim: int = C.HIDDEN_DIM,
                 num_layers: int = C.NUM_LAYERS,
                 dropout: float = C.DROPOUT):
        super().__init__()
        self.embedding, embed_dim = _make_embedding(src_embed_matrix,
                                                    src_vocab_size, embed_type)
        self.rnn  = nn.RNN(embed_dim, hidden_dim, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        emb = self.drop(self.embedding(src))
        outputs, hidden = self.rnn(emb)
        return outputs, hidden


class RNNDecoder(nn.Module):
    """
    Auto-regressive decoder conditioned on encoder hidden state.

    Produces one token at a time; call repeatedly for inference.
    """

    def __init__(self,
                 tgt_vocab_size: int,
                 tgt_embed_matrix: Optional[Tensor] = None,
                 embed_type: str = C.EMBEDDING_TYPE,
                 hidden_dim: int = C.HIDDEN_DIM,
                 num_layers: int = C.NUM_LAYERS,
                 dropout: float = C.DROPOUT):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding, embed_dim = _make_embedding(tgt_embed_matrix,
                                                    tgt_vocab_size, embed_type)
        self.rnn  = nn.RNN(embed_dim, hidden_dim, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self,
                tgt_tok: Tensor,
                hidden: Tensor
                ) -> Tuple[Tensor, Tensor]:
        # tgt_tok: (batch,)  single time step
        emb = self.drop(self.embedding(tgt_tok.unsqueeze(1)))  # (B, 1, E)
        out, hidden = self.rnn(emb, hidden)                    # (B, 1, H)
        logit = self.fc(self.drop(out.squeeze(1)))             # (B, V)
        return logit, hidden


class RNNSeq2Seq(nn.Module):
    """
    Full encoder-decoder RNN translation model with teacher forcing.

    Training   : call forward(src, tgt, teacher_forcing_ratio)
    Inference  : call translate(src, max_len, bos_idx, eos_idx)
    """

    def __init__(self, encoder: RNNEncoder, decoder: RNNDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                teacher_forcing_ratio: float = C.TEACHER_FORCING
                ) -> Tensor:
        """
        src : (batch, src_len)
        tgt : (batch, tgt_len)  — includes BOS at position 0
        Returns (batch, tgt_len - 1, tgt_vocab_size) logits
        """
        batch, tgt_len = tgt.shape
        V = self.decoder.tgt_vocab_size
        outputs = torch.zeros(batch, tgt_len - 1, V, device=src.device)

        _, hidden = self.encoder(src)

        # Align encoder layers to decoder if depths differ
        dec_layers = self.decoder.rnn.num_layers
        hidden = hidden[-dec_layers:]

        tok = tgt[:, 0]                         # BOS token
        for t in range(tgt_len - 1):
            logit, hidden = self.decoder(tok, hidden)
            outputs[:, t] = logit
            use_teacher = random.random() < teacher_forcing_ratio
            tok = tgt[:, t + 1] if use_teacher else logit.argmax(dim=-1)

        return outputs

    @torch.no_grad()
    def translate(self,
                  src: Tensor,
                  max_len: int,
                  bos_idx: int,
                  eos_idx: int) -> Tensor:
        """Greedy decoding – returns (batch, max_len) predicted token ids."""
        self.eval()
        batch = src.size(0)
        _, hidden = self.encoder(src)
        dec_layers = self.decoder.rnn.num_layers
        hidden = hidden[-dec_layers:]

        tok     = torch.full((batch,), bos_idx, dtype=torch.long,
                             device=src.device)
        results = []
        for _ in range(max_len):
            logit, hidden = self.decoder(tok, hidden)
            tok = logit.argmax(dim=-1)
            results.append(tok)
            if (tok == eos_idx).all():
                break

        return torch.stack(results, dim=1)           # (batch, len)
