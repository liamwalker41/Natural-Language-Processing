"""
evaluate.py
===========
Evaluation routines for both tasks.

Task 1 – Text Generation
  Metric: Perplexity  = exp(average cross-entropy loss)
          Lower is better; a model that perfectly predicts the next token
          achieves perplexity == 1.

Task 2 – Machine Translation
  Metric: Corpus-level BLEU score (sacrebleu)
          Measures n-gram overlap between hypotheses and references.
          Industry standard; range 0–100 (higher is better).

Usage examples
--------------
# Evaluate text-gen LSTM with Word2Vec on test split
python evaluate.py --task text_gen --arch lstm --emb word2vec --split test

# Evaluate translation RNN with one-hot on validation split
python evaluate.py --task translation --arch rnn --emb onehot --split val
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from typing import List, Optional

import torch
import torch.nn as nn

import config as C
from load_data import Vocabulary

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NLP model evaluator")
    p.add_argument("--task",  choices=["text_gen", "translation"],
                   default=C.TASK)
    p.add_argument("--arch",  choices=["lstm", "rnn"], default=C.ARCHITECTURE)
    p.add_argument("--emb",   choices=["word2vec", "onehot"],
                   default=C.EMBEDDING_TYPE)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--show_examples", type=int, default=5,
                   help="Number of qualitative examples to display")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Perplexity  (Task 1)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_perplexity(model: nn.Module,
                       loader,
                       device: torch.device,
                       pad_idx: int) -> float:
    """
    Compute token-level perplexity on a DataLoader.
    Perplexity = exp( -1/N * Σ log P(w_t | context) )
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total_loss  = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            hidden = model.init_hidden(x.size(0), device)
            logits, _ = model(x, hidden)
            # Mask padding
            mask = y != pad_idx
            n_tok = mask.sum().item()
            loss  = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            total_loss   += loss.item()
            total_tokens += n_tok

    avg_nll    = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_nll, 20))
    return perplexity


# ═══════════════════════════════════════════════════════════════════════════════
# BLEU  (Task 2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bleu(model: nn.Module,
                 loader,
                 tgt_vocab: Vocabulary,
                 device: torch.device,
                 max_len: int = C.MAX_TGT_LEN) -> float:
    """
    Corpus-level BLEU score using sacrebleu.
    Translates every batch in `loader` and compares to gold references.
    Returns BLEU score (0–100).
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError("Install sacrebleu:  pip install sacrebleu")

    model.eval()
    hypotheses: List[str] = []
    references: List[str] = []

    with torch.no_grad():
        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(device)
            pred_ids  = model.translate(
                src_batch,
                max_len=max_len,
                bos_idx=tgt_vocab.bos_idx,
                eos_idx=tgt_vocab.eos_idx,
            )  # (batch, max_len)

            for pred_row, ref_row in zip(pred_ids, tgt_batch):
                # Truncate at EOS
                pred_list = pred_row.tolist()
                if tgt_vocab.eos_idx in pred_list:
                    pred_list = pred_list[:pred_list.index(tgt_vocab.eos_idx)]

                ref_list = ref_row.tolist()
                if tgt_vocab.eos_idx in ref_list:
                    ref_list = ref_list[:ref_list.index(tgt_vocab.eos_idx)]
                # Remove BOS from reference
                if ref_list and ref_list[0] == tgt_vocab.bos_idx:
                    ref_list = ref_list[1:]

                hypotheses.append(tgt_vocab.decode(pred_list))
                references.append(tgt_vocab.decode(ref_list))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


# ═══════════════════════════════════════════════════════════════════════════════
# Qualitative examples
# ═══════════════════════════════════════════════════════════════════════════════

def show_text_gen_examples(model: nn.Module, vocab: Vocabulary,
                           device: torch.device, n: int = 5):
    """Display n seed→continuation examples for text generation."""
    from predict import generate_text
    log.info("─" * 60)
    log.info("Text Generation Examples")
    log.info("─" * 60)
    seeds = [
        "the history of",
        "scientists have discovered",
        "the government decided to",
        "in the beginning of",
        "one of the most important",
    ]
    for seed in seeds[:n]:
        generated = generate_text(model, vocab, device, seed,
                                  max_len=20, temperature=0.8)
        log.info("Seed : %s", seed)
        log.info("Gen  : %s", generated)
        log.info("")


def show_translation_examples(model: nn.Module,
                               src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                               loader, device: torch.device, n: int = 5):
    """Display n source→prediction vs gold examples."""
    log.info("─" * 60)
    log.info("Translation Examples")
    log.info("─" * 60)
    model.eval()
    count = 0
    with torch.no_grad():
        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(device)
            pred_ids  = model.translate(src_batch, max_len=C.MAX_TGT_LEN,
                                        bos_idx=tgt_vocab.bos_idx,
                                        eos_idx=tgt_vocab.eos_idx)
            for i in range(min(src_batch.size(0), n - count)):
                src_str  = src_vocab.decode(src_batch[i].tolist())
                pred_list = pred_ids[i].tolist()
                if tgt_vocab.eos_idx in pred_list:
                    pred_list = pred_list[:pred_list.index(tgt_vocab.eos_idx)]
                pred_str = tgt_vocab.decode(pred_list)

                ref_list = tgt_batch[i].tolist()
                if tgt_vocab.eos_idx in ref_list:
                    ref_list = ref_list[:ref_list.index(tgt_vocab.eos_idx)]
                if ref_list and ref_list[0] == tgt_vocab.bos_idx:
                    ref_list = ref_list[1:]
                ref_str = tgt_vocab.decode(ref_list)

                log.info("SRC  : %s", src_str)
                log.info("PRED : %s", pred_str)
                log.info("REF  : %s", ref_str)
                log.info("")
                count += 1
                if count >= n:
                    return


# ═══════════════════════════════════════════════════════════════════════════════
# Model loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_from_checkpoint(ckpt_path: str, task: str, arch: str,
                                emb_type: str, device: torch.device):
    """Reconstruct model + vocab(s) from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)

    if task == "text_gen":
        vocab = Vocabulary.load(ckpt["vocab_path"])
        from load_data import load_wikitext2
        _, _, _, _, embed_matrix = load_wikitext2(emb_type)
        from train import build_text_gen_model
        model = build_text_gen_model(
            vocab_size=len(vocab),
            embed_matrix=embed_matrix,
            arch=arch,
            emb_type=emb_type,
            hidden_dim=C.HIDDEN_DIM,
            num_layers=C.NUM_LAYERS,
            dropout=0.0,   # no dropout at eval
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model, vocab

    else:  # translation
        src_vocab = Vocabulary.load(ckpt["src_vocab_path"])
        tgt_vocab = Vocabulary.load(ckpt["tgt_vocab_path"])
        from load_data import load_translation_data
        _, _, _, _, _, src_embed, tgt_embed = load_translation_data(emb_type)
        from train import build_translation_model
        model = build_translation_model(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            src_embed=src_embed,
            tgt_embed=tgt_embed,
            arch=arch,
            emb_type=emb_type,
            hidden_dim=C.HIDDEN_DIM,
            num_layers=C.NUM_LAYERS,
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model, src_vocab, tgt_vocab


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device(args.device)
    ckpt   = C.ckpt_path(args.task, args.arch, args.emb)

    if not os.path.exists(ckpt):
        log.error("Checkpoint not found: %s\nRun train.py first.", ckpt)
        return

    log.info("Loading checkpoint: %s", ckpt)

    results_file = os.path.join(
        C.RESULTS_DIR,
        f"eval_{args.task}_{args.arch}_{args.emb}_{args.split}.txt"
    )

    if args.task == "text_gen":
        model, vocab = load_model_from_checkpoint(ckpt, args.task, args.arch,
                                                  args.emb, device)
        from load_data import load_wikitext2
        _, val_loader, test_loader, _, _ = load_wikitext2(args.emb)
        loader = test_loader if args.split == "test" else val_loader

        ppl = compute_perplexity(model, loader, device, vocab.pad_idx)
        log.info("=" * 60)
        log.info("Task: text_gen | Arch: %s | Emb: %s | Split: %s",
                 args.arch, args.emb, args.split)
        log.info("Perplexity: %.2f", ppl)
        log.info("=" * 60)

        if args.show_examples > 0:
            show_text_gen_examples(model, vocab, device, args.show_examples)

        with open(results_file, "w") as f:
            f.write(f"task={args.task}  arch={args.arch}  "
                    f"emb={args.emb}  split={args.split}\n")
            f.write(f"perplexity={ppl:.4f}\n")

    else:  # translation
        model, src_vocab, tgt_vocab = load_model_from_checkpoint(
            ckpt, args.task, args.arch, args.emb, device)
        from load_data import load_translation_data
        _, val_loader, test_loader, _, _, _, _ = \
            load_translation_data(args.emb)
        loader = test_loader if args.split == "test" else val_loader

        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx,
                                        reduction="sum")
        # Cross-entropy loss as a sanity check alongside BLEU
        total_loss   = 0.0
        total_tokens = 0
        model.eval()
        with torch.no_grad():
            for src_b, tgt_b in loader:
                src_b, tgt_b = src_b.to(device), tgt_b.to(device)
                out  = model(src_b, tgt_b, teacher_forcing_ratio=0.0)
                mask = tgt_b[:, 1:] != tgt_vocab.pad_idx
                loss = criterion(out.reshape(-1, out.size(-1)),
                                 tgt_b[:, 1:].reshape(-1))
                total_loss   += loss.item()
                total_tokens += mask.sum().item()
        val_ppl = math.exp(min(total_loss / max(total_tokens, 1), 20))

        bleu = compute_bleu(model, loader, tgt_vocab, device)

        log.info("=" * 60)
        log.info("Task: translation | Arch: %s | Emb: %s | Split: %s",
                 args.arch, args.emb, args.split)
        log.info("Perplexity : %.2f", val_ppl)
        log.info("BLEU score : %.2f", bleu)
        log.info("=" * 60)

        if args.show_examples > 0:
            show_translation_examples(model, src_vocab, tgt_vocab, loader,
                                      device, args.show_examples)

        with open(results_file, "w") as f:
            f.write(f"task={args.task}  arch={args.arch}  "
                    f"emb={args.emb}  split={args.split}\n")
            f.write(f"perplexity={val_ppl:.4f}\n")
            f.write(f"bleu={bleu:.4f}\n")

    log.info("Results saved → %s", results_file)


if __name__ == "__main__":
    main()
