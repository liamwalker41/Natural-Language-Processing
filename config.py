"""
config.py
=========
Central configuration for all tasks, architectures, and embedding types.
Modify this file to switch between:
  - Tasks:        'text_gen' | 'translation'
  - Architecture: 'lstm'     | 'rnn'
  - Embeddings:   'word2vec' | 'onehot'
"""

import os
from dataclasses import dataclass, field
from typing import Literal


# ── Task / Architecture / Embedding selectors ────────────────────────────────

TASK:           Literal["text_gen", "translation"] = "text_gen"
ARCHITECTURE:   Literal["lstm", "rnn"]             = "lstm"
EMBEDDING_TYPE: Literal["word2vec", "onehot"]      = "word2vec"


# ── Directories ───────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
W2V_CACHE_DIR   = os.path.join(DATA_DIR, "w2v_cache")

for _d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, W2V_CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)


# ── Vocabulary / Tokenisation ─────────────────────────────────────────────────

MIN_FREQ        = 2          # minimum token frequency to enter the vocabulary
MAX_VOCAB_SIZE  = 20_000     # cap vocabulary to keep one-hot tractable
UNK_TOKEN       = "<unk>"
PAD_TOKEN       = "<pad>"
BOS_TOKEN       = "<bos>"
EOS_TOKEN       = "<eos>"
SPECIAL_TOKENS  = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


# ── Embedding ─────────────────────────────────────────────────────────────────

# Word2Vec training (on the corpus itself — avoids large download)
W2V_EMBED_DIM   = 300
W2V_WINDOW      = 5
W2V_MIN_COUNT   = MIN_FREQ
W2V_EPOCHS      = 10
W2V_WORKERS     = 4

# One-hot embedding dimension equals vocab size (set dynamically at runtime)
# EMBED_DIM is resolved in load_data.py after vocab is built.


# ── Model Architecture ────────────────────────────────────────────────────────

HIDDEN_DIM      = 256    # hidden state size for RNN / LSTM cells
NUM_LAYERS      = 2      # number of stacked RNN / LSTM layers
DROPOUT         = 0.5    # dropout probability (applied between layers)
BIDIRECTIONAL   = False  # encoder only; decoder is always unidirectional


# ── Text Generation (Task 1) ──────────────────────────────────────────────────

SEQ_LEN         = 35     # BPTT sequence length


# ── Machine Translation (Task 2) ──────────────────────────────────────────────

MAX_SRC_LEN     = 50     # maximum source sentence length (tokens)
MAX_TGT_LEN     = 50     # maximum target sentence length (tokens)
TEACHER_FORCING = 0.5    # probability of using teacher forcing during training
SRC_LANG        = "en"
TGT_LANG        = "de"


# ── Training ──────────────────────────────────────────────────────────────────

BATCH_SIZE      = 64
EPOCHS          = 15
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 5.0    # gradient clipping max norm
PATIENCE        = 3      # early-stopping patience (epochs without improvement)
SEED            = 42


# ── Prediction / Generation ───────────────────────────────────────────────────

GEN_MAX_LEN     = 100    # maximum tokens to generate
GEN_TEMPERATURE = 1.0    # sampling temperature (1.0 = no change)
GEN_TOP_K       = 0      # top-k sampling (0 = disabled)
GEN_GREEDY      = False  # if True, always pick the argmax token


# ── Device ────────────────────────────────────────────────────────────────────

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Checkpoint naming helper ──────────────────────────────────────────────────

def ckpt_path(task: str = TASK, arch: str = ARCHITECTURE,
              emb: str = EMBEDDING_TYPE) -> str:
    """Return the checkpoint file path for the current run configuration."""
    return os.path.join(CHECKPOINT_DIR, f"{task}_{arch}_{emb}_best.pt")
