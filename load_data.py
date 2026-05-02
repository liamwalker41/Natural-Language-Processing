"""
load_data.py
============
Handles all data loading, vocabulary construction, embedding initialisation,
and DataLoader creation for both tasks:

  Task 1 – Text Generation  : WikiText-2 (via torchtext / HuggingFace datasets)
  Task 2 – Machine Translation: OPUS-100 en↔de subset (HuggingFace datasets)

Embedding strategies
--------------------
  'word2vec'  – Word2Vec model trained on the corpus with gensim; embedding
                matrix is used to initialise a frozen (or fine-tuned) nn.Embedding.
  'onehot'    – Identity / one-hot matrix; vocab is capped at MAX_VOCAB_SIZE to
                keep memory feasible.  The embedding layer is NOT trained.
"""

from __future__ import annotations

import os
import re
import pickle
import logging
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# Gensim Word2Vec
from gensim.models import Word2Vec

import config as C

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """Bidirectional token ↔ index mapping with special tokens."""

    def __init__(self, special_tokens: List[str] = C.SPECIAL_TOKENS):
        self.special_tokens = special_tokens
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        for tok in special_tokens:
            self._add(tok)

    # ── construction ──────────────────────────────────────────────────────────

    def _add(self, token: str) -> int:
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]

    def build_from_counter(self, counter: Counter,
                           min_freq: int = C.MIN_FREQ,
                           max_size: int = C.MAX_VOCAB_SIZE) -> None:
        for token, freq in counter.most_common(max_size - len(self.token2idx)):
            if freq >= min_freq:
                self._add(token)

    # ── properties ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.token2idx)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[C.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[C.UNK_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.token2idx[C.BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[C.EOS_TOKEN]

    # ── encoding / decoding ───────────────────────────────────────────────────

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        skip = set(self.special_tokens) if skip_special else set()
        return " ".join(
            self.idx2token.get(i, C.UNK_TOKEN)
            for i in indices
            if self.idx2token.get(i, C.UNK_TOKEN) not in skip
        )

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

_CLEAN_RE = re.compile(r"[^a-zA-ZäöüÄÖÜß\s'-]")


def simple_tokenize(text: str, lower: bool = True) -> List[str]:
    """Very lightweight whitespace tokeniser (no external deps required)."""
    if lower:
        text = text.lower()
    text = _CLEAN_RE.sub(" ", text)
    return text.split()


# ═══════════════════════════════════════════════════════════════════════════════
# Word2Vec embedding helpers
# ═══════════════════════════════════════════════════════════════════════════════

def train_word2vec(sentences: List[List[str]],
                   cache_path: Optional[str] = None) -> Word2Vec:
    """Train a Word2Vec model on the supplied tokenised sentences."""
    if cache_path and os.path.exists(cache_path):
        log.info("Loading cached Word2Vec model from %s", cache_path)
        return Word2Vec.load(cache_path)

    log.info("Training Word2Vec on %d sentences …", len(sentences))
    model = Word2Vec(
        sentences=sentences,
        vector_size=C.W2V_EMBED_DIM,
        window=C.W2V_WINDOW,
        min_count=C.W2V_MIN_COUNT,
        workers=C.W2V_WORKERS,
        epochs=C.W2V_EPOCHS,
        seed=C.SEED,
    )
    if cache_path:
        model.save(cache_path)
        log.info("Word2Vec model saved to %s", cache_path)
    return model


def build_embedding_matrix(vocab: Vocabulary,
                            w2v_model: Optional[Word2Vec],
                            embed_type: str) -> Tensor:
    """
    Return an (vocab_size × embed_dim) tensor used to initialise nn.Embedding.

    embed_type == 'word2vec' : rows filled from w2v_model; unknown tokens get
                               a random unit vector.
    embed_type == 'onehot'   : identity matrix; embed_dim == vocab_size.
    """
    V = len(vocab)

    if embed_type == "onehot":
        log.info("Building one-hot embedding matrix  (%d × %d)", V, V)
        return torch.eye(V)  # embed_dim == vocab_size

    # ── Word2Vec ──
    assert w2v_model is not None, "w2v_model required for 'word2vec' embeddings"
    D = C.W2V_EMBED_DIM
    matrix = np.zeros((V, D), dtype=np.float32)
    hits = 0
    for token, idx in vocab.token2idx.items():
        if token in w2v_model.wv:
            matrix[idx] = w2v_model.wv[token]
            hits += 1
        else:
            matrix[idx] = np.random.normal(scale=0.6, size=(D,))
    log.info("Word2Vec coverage: %d / %d tokens (%.1f%%)", hits, V, 100 * hits / V)
    return torch.tensor(matrix)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1 – Text Generation  (WikiText-2)
# ═══════════════════════════════════════════════════════════════════════════════

class TextGenDataset(Dataset):
    """
    Flat token sequence split into (input, target) pairs of length SEQ_LEN.
    target[i] == input[i + 1]  (standard language-model objective).
    """

    def __init__(self, token_ids: List[int], seq_len: int = C.SEQ_LEN):
        self.seq_len = seq_len
        # Trim so that len divides evenly
        n = (len(token_ids) - 1) // seq_len * seq_len
        self.data = torch.tensor(token_ids[:n + 1], dtype=torch.long)

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start = idx * self.seq_len
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


def load_wikitext2(embedding_type: str = C.EMBEDDING_TYPE):
    """
    Download (or load from cache) WikiText-2, build vocab, train Word2Vec
    if needed, and return DataLoaders + vocab + embedding matrix.

    Returns
    -------
    train_loader, val_loader, test_loader, vocab, embed_matrix
    """
    from datasets import load_dataset

    cache_file = os.path.join(C.DATA_DIR, "wikitext2_vocab.pkl")

    # ── raw text ──────────────────────────────────────────────────────────────
    log.info("Loading WikiText-2 …")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def _tokenise_split(split_name: str) -> List[List[str]]:
        sentences = []
        for item in ds[split_name]:
            text = item["text"].strip()
            if text:
                toks = simple_tokenize(text)
                if toks:
                    sentences.append(toks)
        return sentences

    train_sents = _tokenise_split("train")
    val_sents   = _tokenise_split("validation")
    test_sents  = _tokenise_split("test")

    # ── vocabulary ────────────────────────────────────────────────────────────
    if os.path.exists(cache_file):
        log.info("Loading cached vocabulary …")
        vocab = Vocabulary.load(cache_file)
    else:
        log.info("Building vocabulary …")
        counter: Counter = Counter()
        for sent in train_sents:
            counter.update(sent)
        vocab = Vocabulary()
        vocab.build_from_counter(counter)
        vocab.save(cache_file)
    log.info("Vocabulary size: %d", len(vocab))

    # ── Word2Vec ──────────────────────────────────────────────────────────────
    w2v_model = None
    if embedding_type == "word2vec":
        w2v_cache = os.path.join(C.W2V_CACHE_DIR, "wikitext2_w2v.model")
        w2v_model = train_word2vec(train_sents, cache_path=w2v_cache)

    embed_matrix = build_embedding_matrix(vocab, w2v_model, embedding_type)

    # ── encode to flat id sequences ───────────────────────────────────────────
    def _encode_flat(sentences: List[List[str]]) -> List[int]:
        ids = []
        for s in sentences:
            ids.extend(vocab.encode(s))
        return ids

    train_ids = _encode_flat(train_sents)
    val_ids   = _encode_flat(val_sents)
    test_ids  = _encode_flat(test_sents)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def _loader(ids, shuffle):
        ds_obj = TextGenDataset(ids)
        return DataLoader(ds_obj, batch_size=C.BATCH_SIZE,
                          shuffle=shuffle, drop_last=True)

    train_loader = _loader(train_ids, shuffle=True)
    val_loader   = _loader(val_ids,   shuffle=False)
    test_loader  = _loader(test_ids,  shuffle=False)

    log.info("Train batches: %d | Val: %d | Test: %d",
             len(train_loader), len(val_loader), len(test_loader))

    return train_loader, val_loader, test_loader, vocab, embed_matrix


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2 – Machine Translation  (OPUS-100 en-de)
# ═══════════════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """
    Pairs of (src_ids, tgt_ids) with BOS/EOS wrapping on the target side.
    """

    def __init__(self, src_seqs: List[List[int]], tgt_seqs: List[List[int]],
                 src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                 max_src: int = C.MAX_SRC_LEN,
                 max_tgt: int = C.MAX_TGT_LEN):
        assert len(src_seqs) == len(tgt_seqs)
        self.pairs = []
        for s, t in zip(src_seqs, tgt_seqs):
            s = s[:max_src]
            t = [tgt_vocab.bos_idx] + t[:max_tgt - 2] + [tgt_vocab.eos_idx]
            self.pairs.append((s, t))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.pairs[idx]


def _collate_translation(batch, src_pad: int, tgt_pad: int):
    """Pad a batch of variable-length sequence pairs."""
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    max_s = max(src_lens)
    max_t = max(tgt_lens)

    src_tensor = torch.full((len(batch), max_s), src_pad, dtype=torch.long)
    tgt_tensor = torch.full((len(batch), max_t), tgt_pad, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_tensor[i, :len(s)] = torch.tensor(s)
        tgt_tensor[i, :len(t)] = torch.tensor(t)

    return src_tensor, tgt_tensor


def load_translation_data(embedding_type: str = C.EMBEDDING_TYPE,
                          max_samples: int = 50_000):
    """
    Download OPUS-100 en-de, build separate src/tgt vocabularies, and return
    DataLoaders + vocabs + embedding matrices.

    Returns
    -------
    train_loader, val_loader, test_loader,
    src_vocab, tgt_vocab,
    src_embed_matrix, tgt_embed_matrix
    """
    from datasets import load_dataset

    src_cache = os.path.join(C.DATA_DIR, "translation_src_vocab.pkl")
    tgt_cache = os.path.join(C.DATA_DIR, "translation_tgt_vocab.pkl")

    log.info("Loading OPUS-100 en-de …")
    # opus_books is a lighter alternative; opus100 has 1 M pairs
    try:
        ds = load_dataset("opus100", "en-de")
    except Exception:
        log.warning("opus100 unavailable, falling back to opus_books")
        ds = load_dataset("opus_books", "en-de")

    def _extract(split):
        rows = ds[split]["translation"] if "translation" in ds[split].features \
            else ds[split]
        src, tgt = [], []
        for row in rows:
            s = simple_tokenize(row.get("en", row.get(C.SRC_LANG, "")))
            t = simple_tokenize(row.get("de", row.get(C.TGT_LANG, "")))
            if 1 <= len(s) <= C.MAX_SRC_LEN and 1 <= len(t) <= C.MAX_TGT_LEN:
                src.append(s)
                tgt.append(t)
        return src, tgt

    train_src, train_tgt = _extract("train")
    # cap training set for speed
    train_src, train_tgt = train_src[:max_samples], train_tgt[:max_samples]

    # Some datasets don't have val/test; create splits manually
    if "validation" in ds:
        val_src, val_tgt     = _extract("validation")
        test_src, test_tgt   = _extract("test") if "test" in ds else (val_src[:500], val_tgt[:500])
    else:
        n = len(train_src)
        val_src,  val_tgt  = train_src[n - 2000: n - 1000], train_tgt[n - 2000: n - 1000]
        test_src, test_tgt = train_src[n - 1000:],          train_tgt[n - 1000:]
        train_src, train_tgt = train_src[:n - 2000], train_tgt[:n - 2000]

    # ── vocabularies ──────────────────────────────────────────────────────────
    def _build_vocab(sentences, cache):
        if os.path.exists(cache):
            return Vocabulary.load(cache)
        ctr: Counter = Counter()
        for s in sentences:
            ctr.update(s)
        v = Vocabulary()
        v.build_from_counter(ctr)
        v.save(cache)
        return v

    src_vocab = _build_vocab(train_src, src_cache)
    tgt_vocab = _build_vocab(train_tgt, tgt_cache)
    log.info("Src vocab: %d | Tgt vocab: %d", len(src_vocab), len(tgt_vocab))

    # ── Word2Vec ──────────────────────────────────────────────────────────────
    src_w2v = tgt_w2v = None
    if embedding_type == "word2vec":
        src_w2v = train_word2vec(
            train_src,
            cache_path=os.path.join(C.W2V_CACHE_DIR, "trans_src_w2v.model"))
        tgt_w2v = train_word2vec(
            train_tgt,
            cache_path=os.path.join(C.W2V_CACHE_DIR, "trans_tgt_w2v.model"))

    src_embed = build_embedding_matrix(src_vocab, src_w2v, embedding_type)
    tgt_embed = build_embedding_matrix(tgt_vocab, tgt_w2v, embedding_type)

    # ── encode ────────────────────────────────────────────────────────────────
    def _encode_pair(srcs, tgts):
        return (
            [src_vocab.encode(s) for s in srcs],
            [tgt_vocab.encode(t) for t in tgts],
        )

    def _make_loader(srcs, tgts, shuffle):
        s_ids, t_ids = _encode_pair(srcs, tgts)
        dataset = TranslationDataset(s_ids, t_ids, src_vocab, tgt_vocab)
        collate = lambda b: _collate_translation(b, src_vocab.pad_idx,
                                                 tgt_vocab.pad_idx)
        return DataLoader(dataset, batch_size=C.BATCH_SIZE,
                          shuffle=shuffle, collate_fn=collate, drop_last=False)

    train_loader = _make_loader(train_src, train_tgt, shuffle=True)
    val_loader   = _make_loader(val_src,   val_tgt,   shuffle=False)
    test_loader  = _make_loader(test_src,  test_tgt,  shuffle=False)

    log.info("Translation — Train: %d | Val: %d | Test: %d",
             len(train_loader), len(val_loader), len(test_loader))

    return (train_loader, val_loader, test_loader,
            src_vocab, tgt_vocab, src_embed, tgt_embed)
