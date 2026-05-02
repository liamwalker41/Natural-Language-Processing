"""
models/lstm_model.py
====================
LSTM implementations for:
  • Task 1 – Language modelling / text generation   (LSTMLanguageModel)
  • Task 2 – Machine translation via Seq2Seq        (LSTMSeq2Seq)

Compared to the RNN baseline, LSTM adds:
  - Cell state  (c_t) that carries long-range information
  - Input / forget / output gates that learn what to remember and discard
  - Significantly better gradient flow → can model longer dependencies

The API is intentionally identical to rnn_model.py to allow easy switching
via config.ARCHITECTURE.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import config as C
from models.rnn_model import _make_embedding   # shared helper


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1 – LSTM Language Model
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMLanguageModel(nn.Module):
    """
    Multi-layer LSTM language model.

    Forward pass
    ------------
    Input  : (batch, seq_len)  token indices
    Output : (batch, seq_len, vocab_size)  logits over vocabulary
    Hidden : Tuple[ (num_layers, batch, H), (num_layers, batch, H) ]  (h, c)

    Features
    --------
    - Variational dropout applied consistently across time steps
    - Optional weight tying between embedding and output projection
    """

    def __init__(self,
                 vocab_size: int,
                 embed_matrix: Optional[Tensor] = None,
                 embed_type: str = C.EMBEDDING_TYPE,
                 hidden_dim: int = C.HIDDEN_DIM,
                 num_layers: int = C.NUM_LAYERS,
                 dropout: float = C.DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding, embed_dim = _make_embedding(embed_matrix, vocab_size,
                                                    embed_type)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, vocab_size)

        # Weight tying
        if embed_dim == hidden_dim and embed_type == "word2vec":
            self.fc.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # Initialise forget gate bias to 1 – standard best practice
                n = p.size(0)
                p.data[n // 4: n // 2].fill_(1.0)
        nn.init.zeros_(self.fc.bias)

    def forward(self,
                src: Tensor,
                hidden: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        emb = self.drop(self.embedding(src))        # (B, T, E)
        out, hidden = self.lstm(emb, hidden)        # (B, T, H)
        out = self.drop(out)
        logits = self.fc(out)                       # (B, T, V)
        return logits, hidden

    def init_hidden(self,
                    batch_size: int,
                    device: torch.device) -> Tuple[Tensor, Tensor]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                        device=device)
        c = torch.zeros_like(h)
        return h, c

    @staticmethod
    def detach_hidden(hidden: Tuple[Tensor, Tensor]
                      ) -> Tuple[Tensor, Tensor]:
        """Truncated BPTT: detach hidden state from the computation graph."""
        return hidden[0].detach(), hidden[1].detach()


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2 – LSTM Seq2Seq (Encoder + Decoder + wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMEncoder(nn.Module):
    """
    LSTM encoder – maps source token ids to a context vector.

    Input  : (batch, src_len)
    Output : encoder_outputs (batch, src_len, H),
             hidden tuple  ( (L, B, H), (L, B, H) )
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
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        emb = self.drop(self.embedding(src))
        outputs, hidden = self.lstm(emb)
        return outputs, hidden


class LSTMDecoder(nn.Module):
    """
    LSTM decoder with optional additive (Bahdanau-style) attention.

    One-step forward:
      tgt_tok : (batch,)           — current input token id
      hidden  : (h, c) tuple       — from previous step
      enc_out : (batch, src_len, H) — encoder outputs (for attention)

    Returns logit (batch, V) and new hidden state.
    """

    def __init__(self,
                 tgt_vocab_size: int,
                 tgt_embed_matrix: Optional[Tensor] = None,
                 embed_type: str = C.EMBEDDING_TYPE,
                 hidden_dim: int = C.HIDDEN_DIM,
                 num_layers: int = C.NUM_LAYERS,
                 dropout: float = C.DROPOUT,
                 use_attention: bool = True):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.use_attention  = use_attention
        self.hidden_dim     = hidden_dim

        self.embedding, embed_dim = _make_embedding(tgt_embed_matrix,
                                                    tgt_vocab_size, embed_type)

        # Attention
        if use_attention:
            self.attn_Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.attn_Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.attn_v  = nn.Linear(hidden_dim, 1, bias=False)
            lstm_in = embed_dim + hidden_dim   # embed + context vector
        else:
            lstm_in = embed_dim

        self.lstm = nn.LSTM(lstm_in, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, tgt_vocab_size)

    def _attend(self, hidden: Tensor, enc_out: Tensor) -> Tensor:
        """
        Additive attention.
        hidden  : (batch, H)           — top-layer decoder hidden state
        enc_out : (batch, src_len, H)
        Returns context : (batch, H)
        """
        # (batch, src_len, H)
        score = self.attn_v(
            torch.tanh(
                self.attn_Wa(enc_out) +
                self.attn_Ua(hidden.unsqueeze(1))
            )
        ).squeeze(-1)                              # (batch, src_len)
        weights = torch.softmax(score, dim=-1)     # (batch, src_len)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context                             # (batch, H)

    def forward(self,
                tgt_tok: Tensor,
                hidden: Tuple[Tensor, Tensor],
                enc_out: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        emb = self.drop(self.embedding(tgt_tok.unsqueeze(1)))  # (B, 1, E)

        if self.use_attention and enc_out is not None:
            context = self._attend(hidden[0][-1], enc_out)     # (B, H)
            emb = torch.cat([emb, context.unsqueeze(1)], dim=-1)  # (B,1,E+H)

        out, hidden = self.lstm(emb, hidden)
        logit = self.fc(self.drop(out.squeeze(1)))             # (B, V)
        return logit, hidden


class LSTMSeq2Seq(nn.Module):
    """
    Full LSTM encoder-decoder with teacher forcing and attention.

    Training  : forward(src, tgt, teacher_forcing_ratio)
    Inference : translate(src, max_len, bos_idx, eos_idx)
    """

    def __init__(self, encoder: LSTMEncoder, decoder: LSTMDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                teacher_forcing_ratio: float = C.TEACHER_FORCING
                ) -> Tensor:
        batch, tgt_len = tgt.shape
        V = self.decoder.tgt_vocab_size
        outputs = torch.zeros(batch, tgt_len - 1, V, device=src.device)

        enc_out, hidden = self.encoder(src)

        # Trim encoder layers to match decoder depth
        dec_layers = self.decoder.lstm.num_layers
        h = hidden[0][-dec_layers:]
        c = hidden[1][-dec_layers:]
        hidden = (h, c)

        tok = tgt[:, 0]
        for t in range(tgt_len - 1):
            logit, hidden = self.decoder(tok, hidden, enc_out)
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
        """Greedy decoding – returns (batch, max_len) predicted ids."""
        self.eval()
        batch = src.size(0)
        enc_out, hidden = self.encoder(src)
        dec_layers = self.decoder.lstm.num_layers
        h = hidden[0][-dec_layers:]
        c = hidden[1][-dec_layers:]
        hidden = (h, c)

        tok     = torch.full((batch,), bos_idx, dtype=torch.long,
                             device=src.device)
        results = []
        for _ in range(max_len):
            logit, hidden = self.decoder(tok, hidden, enc_out)
            tok = logit.argmax(dim=-1)
            results.append(tok)
            if (tok == eos_idx).all():
                break

        return torch.stack(results, dim=1)
