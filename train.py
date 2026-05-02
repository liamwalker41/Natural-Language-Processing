"""
train.py
========
Unified training entry-point.  Configure everything via config.py or
the CLI flags below.

Usage examples
--------------
# LSTM + Word2Vec, text generation (defaults)
python train.py

# RNN + One-hot, text generation
python train.py --arch rnn --emb onehot --task text_gen

# LSTM + Word2Vec, machine translation
python train.py --arch lstm --emb word2vec --task translation

# Override epochs / lr
python train.py --arch lstm --emb word2vec --task text_gen --epochs 20 --lr 5e-4
"""

import argparse
import math
import os
import time
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config as C

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NLP model trainer")
    p.add_argument("--task",   choices=["text_gen", "translation"],
                   default=C.TASK)
    p.add_argument("--arch",   choices=["lstm", "rnn"], default=C.ARCHITECTURE)
    p.add_argument("--emb",    choices=["word2vec", "onehot"],
                   default=C.EMBEDDING_TYPE)
    p.add_argument("--epochs", type=int,   default=C.EPOCHS)
    p.add_argument("--lr",     type=float, default=C.LEARNING_RATE)
    p.add_argument("--batch",  type=int,   default=C.BATCH_SIZE)
    p.add_argument("--hidden", type=int,   default=C.HIDDEN_DIM)
    p.add_argument("--layers", type=int,   default=C.NUM_LAYERS)
    p.add_argument("--dropout",type=float, default=C.DROPOUT)
    p.add_argument("--seed",   type=int,   default=C.SEED)
    p.add_argument("--device", default=C.DEVICE)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Model builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_text_gen_model(vocab_size, embed_matrix, arch, emb_type,
                         hidden_dim, num_layers, dropout):
    from models.lstm_model import LSTMLanguageModel
    from models.rnn_model  import RNNLanguageModel
    cls = LSTMLanguageModel if arch == "lstm" else RNNLanguageModel
    return cls(
        vocab_size=vocab_size,
        embed_matrix=embed_matrix,
        embed_type=emb_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_translation_model(src_vocab_size, tgt_vocab_size,
                             src_embed, tgt_embed, arch, emb_type,
                             hidden_dim, num_layers, dropout):
    if arch == "lstm":
        from models.lstm_model import LSTMEncoder, LSTMDecoder, LSTMSeq2Seq
        enc = LSTMEncoder(src_vocab_size, src_embed, emb_type,
                          hidden_dim, num_layers, dropout)
        dec = LSTMDecoder(tgt_vocab_size, tgt_embed, emb_type,
                          hidden_dim, num_layers, dropout, use_attention=True)
        return LSTMSeq2Seq(enc, dec)
    else:
        from models.rnn_model import RNNEncoder, RNNDecoder, RNNSeq2Seq
        enc = RNNEncoder(src_vocab_size, src_embed, emb_type,
                         hidden_dim, num_layers, dropout)
        dec = RNNDecoder(tgt_vocab_size, tgt_embed, emb_type,
                         hidden_dim, num_layers, dropout)
        return RNNSeq2Seq(enc, dec)


# ═══════════════════════════════════════════════════════════════════════════════
# Training loops
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch_text_gen(model, loader, optimizer, criterion,
                         device, grad_clip) -> float:
    """One training epoch for the language model.  Returns average loss."""
    model.train()
    total_loss = 0.0
    is_lstm = hasattr(model, "init_hidden") and hasattr(
        model.init_hidden(1, device), "__len__")

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        # Initialise / detach hidden state
        if batch_idx == 0:
            hidden = model.init_hidden(batch_size, device)
        else:
            if isinstance(hidden, tuple):
                hidden = model.detach_hidden(hidden) \
                    if hasattr(model, "detach_hidden") \
                    else (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()

        optimizer.zero_grad()
        logits, hidden = model(x, hidden)    # (B, T, V)

        # Flatten for cross-entropy
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch_text_gen(model, loader, criterion, device) -> float:
    """Evaluation epoch for the language model.  Returns average loss."""
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        hidden = model.init_hidden(x.size(0), device)
        logits, _ = model(x, hidden)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch_translation(model, loader, optimizer, criterion,
                            device, grad_clip, pad_idx) -> float:
    """One training epoch for seq2seq.  Returns average loss."""
    model.train()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        # output: (batch, tgt_len-1, V)
        output = model(src, tgt)
        # Flatten: ignore BOS token in target (shifted by 1)
        output_flat = output.reshape(-1, output.size(-1))
        target_flat = tgt[:, 1:].reshape(-1)
        loss = criterion(output_flat, target_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch_translation(model, loader, criterion, device, pad_idx) -> float:
    model.eval()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt, teacher_forcing_ratio=0.0)
        output_flat = output.reshape(-1, output.size(-1))
        target_flat = tgt[:, 1:].reshape(-1)
        loss = criterion(output_flat, target_flat)
        total_loss += loss.item()
    return total_loss / len(loader)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, epoch, val_loss, path, extra=None):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    log.info("Resumed from checkpoint (epoch %d, val_loss %.4f)",
             ckpt["epoch"], ckpt["val_loss"])
    return ckpt["epoch"], ckpt["val_loss"]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    log.info("Device: %s | Task: %s | Arch: %s | Emb: %s",
             device, args.task, args.arch, args.emb)

    ckpt_file = C.ckpt_path(args.task, args.arch, args.emb)

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.task == "text_gen":
        from load_data import load_wikitext2
        train_loader, val_loader, test_loader, vocab, embed_matrix = \
            load_wikitext2(args.emb)

        model = build_text_gen_model(
            vocab_size=len(vocab),
            embed_matrix=embed_matrix,
            arch=args.arch,
            emb_type=args.emb,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        pad_idx   = vocab.pad_idx
        extra_save = {"vocab_path": os.path.join(C.DATA_DIR,
                                                  "wikitext2_vocab.pkl")}

        def train_fn(ep):
            return train_epoch_text_gen(model, train_loader, optimizer,
                                        criterion, device, C.GRAD_CLIP)
        def val_fn():
            return eval_epoch_text_gen(model, val_loader, criterion, device)

    else:  # translation
        from load_data import load_translation_data
        (train_loader, val_loader, test_loader,
         src_vocab, tgt_vocab, src_embed, tgt_embed) = \
            load_translation_data(args.emb)

        model = build_translation_model(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            src_embed=src_embed,
            tgt_embed=tgt_embed,
            arch=args.arch,
            emb_type=args.emb,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
        ).to(device)

        pad_idx   = tgt_vocab.pad_idx
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        extra_save = {
            "src_vocab_path": os.path.join(C.DATA_DIR,
                                           "translation_src_vocab.pkl"),
            "tgt_vocab_path": os.path.join(C.DATA_DIR,
                                           "translation_tgt_vocab.pkl"),
        }

        def train_fn(ep):
            return train_epoch_translation(model, train_loader, optimizer,
                                           criterion, device, C.GRAD_CLIP,
                                           pad_idx)
        def val_fn():
            return eval_epoch_translation(model, val_loader, criterion,
                                          device, pad_idx)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=C.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=1, verbose=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_ctr  = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_fn(epoch)
        val_loss   = val_fn()
        elapsed    = time.time() - t0

        train_ppl = math.exp(min(train_loss, 20))
        val_ppl   = math.exp(min(val_loss,   20))
        log.info(
            "Epoch %02d/%02d | %.1fs | "
            "Train loss %.4f  ppl %.2f | Val loss %.4f  ppl %.2f",
            epoch, args.epochs, elapsed,
            train_loss, train_ppl, val_loss, val_ppl,
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            save_checkpoint(model, optimizer, epoch, val_loss,
                            ckpt_file, extra_save)
        else:
            patience_ctr += 1
            log.info("No improvement (%d/%d)", patience_ctr, C.PATIENCE)
            if patience_ctr >= C.PATIENCE:
                log.info("Early stopping triggered.")
                break

    log.info("Training complete.  Best val loss: %.4f  (ppl %.2f)",
             best_val_loss, math.exp(min(best_val_loss, 20)))


if __name__ == "__main__":
    main()
