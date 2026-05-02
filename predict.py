"""
predict.py
==========
Interactive inference for both tasks.

Task 1 – Text Generation
  Given a seed phrase, the model samples the next N tokens.
  Supports greedy, top-k, and temperature sampling.

Task 2 – Machine Translation
  Translates an English sentence to the target language using
  greedy decoding (or beam search if enabled).

Usage examples
--------------
# Text generation – interactive mode
python predict.py --task text_gen --arch lstm --emb word2vec

# Text generation – single seed
python predict.py --task text_gen --arch lstm --emb word2vec \
    --seed "the history of science" --max_len 50 --temperature 0.8

# Machine translation – interactive mode
python predict.py --task translation --arch lstm --emb word2vec

# Machine translation – single sentence
python predict.py --task translation --arch rnn --emb onehot \
    --sentence "The cat sits on the mat."
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import config as C
from load_data import Vocabulary

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NLP model predictor")
    p.add_argument("--task",  choices=["text_gen", "translation"],
                   default=C.TASK)
    p.add_argument("--arch",  choices=["lstm", "rnn"], default=C.ARCHITECTURE)
    p.add_argument("--emb",   choices=["word2vec", "onehot"],
                   default=C.EMBEDDING_TYPE)
    p.add_argument("--device", default=C.DEVICE)

    # Text generation args
    p.add_argument("--seed",        type=str, default=None,
                   help="Seed phrase for text generation")
    p.add_argument("--max_len",     type=int, default=C.GEN_MAX_LEN)
    p.add_argument("--temperature", type=float, default=C.GEN_TEMPERATURE)
    p.add_argument("--top_k",       type=int, default=C.GEN_TOP_K)
    p.add_argument("--greedy",      action="store_true")

    # Translation args
    p.add_argument("--sentence", type=str, default=None,
                   help="English sentence to translate")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_next_token(logits: Tensor,
                       temperature: float = 1.0,
                       top_k: int = 0,
                       greedy: bool = False) -> int:
    """
    Sample the next token index from a logits vector.

    Parameters
    ----------
    logits      : raw logits, shape (vocab_size,)
    temperature : divide logits before softmax; <1 → sharper, >1 → flatter
    top_k       : restrict sampling to the top-k most probable tokens
    greedy      : if True, always pick the argmax (overrides sampling)
    """
    if greedy:
        return int(logits.argmax().item())

    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        # Zero out everything except the top-k
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold  = values[-1]
        logits     = logits.masked_fill(logits < threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1 – Text Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_text(model,
                  vocab: Vocabulary,
                  device: torch.device,
                  seed: str,
                  max_len: int = C.GEN_MAX_LEN,
                  temperature: float = C.GEN_TEMPERATURE,
                  top_k: int = C.GEN_TOP_K,
                  greedy: bool = C.GEN_GREEDY) -> str:
    """
    Generate text continuation from `seed`.

    The seed is tokenised and fed through the model to prime the hidden state,
    then tokens are sampled one at a time until `max_len` new tokens are
    produced or <eos> is encountered.

    Returns
    -------
    Full string:  seed tokens + generated continuation
    """
    from load_data import simple_tokenize

    model.eval()
    with torch.no_grad():
        seed_tokens = simple_tokenize(seed)
        seed_ids    = vocab.encode(seed_tokens)

        if not seed_ids:
            log.warning("Seed produced 0 tokens after tokenisation.")
            return seed

        # Prime the model's hidden state with the seed
        seed_tensor = torch.tensor(seed_ids, dtype=torch.long,
                                   device=device).unsqueeze(0)  # (1, T)
        hidden      = model.init_hidden(1, device)
        _, hidden   = model(seed_tensor, hidden)

        generated: List[int] = list(seed_ids)
        tok_id    = seed_ids[-1]          # start generating from last seed tok

        for _ in range(max_len):
            inp     = torch.tensor([[tok_id]], dtype=torch.long, device=device)
            logits, hidden = model(inp, hidden)
            logits  = logits[0, -1]       # (V,)

            # Block special tokens
            for special in C.SPECIAL_TOKENS:
                if special in vocab.token2idx and special != C.UNK_TOKEN:
                    logits[vocab.token2idx[special]] = float("-inf")

            tok_id = _sample_next_token(logits, temperature, top_k, greedy)

            if tok_id == vocab.eos_idx:
                break
            generated.append(tok_id)

    return vocab.decode(generated, skip_special=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2 – Machine Translation
# ═══════════════════════════════════════════════════════════════════════════════

def translate_sentence(model,
                       sentence: str,
                       src_vocab: Vocabulary,
                       tgt_vocab: Vocabulary,
                       device: torch.device,
                       max_len: int = C.MAX_TGT_LEN) -> str:
    """
    Translate a single English sentence to the target language.

    Tokenises `sentence` → encodes → greedy decode → detokenise.
    """
    from load_data import simple_tokenize

    model.eval()
    with torch.no_grad():
        tokens = simple_tokenize(sentence, lower=True)
        ids    = src_vocab.encode(tokens)
        if not ids:
            return "[empty input after tokenisation]"

        src = torch.tensor(ids, dtype=torch.long,
                           device=device).unsqueeze(0)   # (1, src_len)

        pred_ids = model.translate(
            src,
            max_len=max_len,
            bos_idx=tgt_vocab.bos_idx,
            eos_idx=tgt_vocab.eos_idx,
        )  # (1, pred_len)

        pred_list = pred_ids[0].tolist()
        if tgt_vocab.eos_idx in pred_list:
            pred_list = pred_list[:pred_list.index(tgt_vocab.eos_idx)]

    return tgt_vocab.decode(pred_list)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint loader (shared with evaluate.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _load(task, arch, emb, device):
    from evaluate import load_model_from_checkpoint
    ckpt = C.ckpt_path(task, arch, emb)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Train the model first with: python train.py --task {task} "
            f"--arch {arch} --emb {emb}"
        )
    return load_model_from_checkpoint(ckpt, task, arch, emb, device)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device(args.device)

    log.info("Loading model  task=%s  arch=%s  emb=%s …",
             args.task, args.arch, args.emb)

    # ── Text Generation ───────────────────────────────────────────────────────
    if args.task == "text_gen":
        result = _load(args.task, args.arch, args.emb, device)
        model, vocab = result

        def _gen(seed_text: str) -> str:
            return generate_text(model, vocab, device,
                                 seed=seed_text,
                                 max_len=args.max_len,
                                 temperature=args.temperature,
                                 top_k=args.top_k,
                                 greedy=args.greedy)

        if args.seed:
            print("\n" + "─" * 60)
            print("Input :", args.seed)
            print("Output:", _gen(args.seed))
            print("─" * 60)
        else:
            print("\nText Generation Interactive Mode")
            print("Type a seed phrase and press Enter.  Ctrl-C to quit.\n")
            while True:
                try:
                    seed_text = input("Seed > ").strip()
                    if seed_text:
                        print("Output:", _gen(seed_text))
                        print()
                except (KeyboardInterrupt, EOFError):
                    print("\nBye!")
                    break

    # ── Machine Translation ───────────────────────────────────────────────────
    else:
        result = _load(args.task, args.arch, args.emb, device)
        model, src_vocab, tgt_vocab = result

        def _translate(sent: str) -> str:
            return translate_sentence(model, sent, src_vocab, tgt_vocab, device,
                                      max_len=args.max_len)

        if args.sentence:
            print("\n" + "─" * 60)
            print("EN :", args.sentence)
            print("DE :", _translate(args.sentence))
            print("─" * 60)
        else:
            print("\nTranslation Interactive Mode (EN → DE)")
            print("Type an English sentence and press Enter.  Ctrl-C to quit.\n")
            while True:
                try:
                    sent = input("EN > ").strip()
                    if sent:
                        print("DE :", _translate(sent))
                        print()
                except (KeyboardInterrupt, EOFError):
                    print("\nBye!")
                    break


if __name__ == "__main__":
    main()
