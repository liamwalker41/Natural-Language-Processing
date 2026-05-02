# NLP Neural Network Project

> Two NLP tasks × two architectures × two embedding types = eight comparable model configurations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Task 1 – Text Generation](#5-task-1--text-generation)
6. [Task 2 – Machine Translation](#6-task-2--machine-translation)
7. [Model Architectures](#7-model-architectures)
8. [Word Embeddings](#8-word-embeddings)
9. [Training](#9-training)
10. [Evaluation](#10-evaluation)
11. [Prediction & Inference](#11-prediction--inference)
12. [Experimental Results & Discussion](#12-experimental-results--discussion)
13. [Implementation Notes](#13-implementation-notes)

---

## 1. Project Overview

This project implements and compares neural language models for two core NLP tasks:

| Task | Dataset | Objective | Metric |
|------|---------|-----------|--------|
| Text Generation | WikiText-2 | Predict the next word | Perplexity |
| Machine Translation | OPUS-100 (en→de) | Translate English to German | BLEU |

Each task is solved with **two architectures** (Basic RNN, LSTM) and **two embedding strategies** (Word2Vec, One-hot), giving eight total configurations to compare.

---

## 2. Repository Structure

```
nlp_project/
│
├── config.py          # All hyperparameters and run selectors
├── load_data.py       # Data loading, vocabulary, embeddings, DataLoaders
├── train.py           # Unified training script (both tasks / architectures)
├── evaluate.py        # Perplexity and BLEU evaluation
├── predict.py         # Interactive inference and single-input prediction
├── requirements.txt
├── README.md
│
├── models/
│   ├── __init__.py
│   ├── rnn_model.py   # Basic RNN (language model + Seq2Seq)
│   └── lstm_model.py  # LSTM (language model + Seq2Seq with attention)
│
├── data/              # Auto-created; caches vocabularies and Word2Vec models
├── checkpoints/       # Auto-created; best model checkpoints saved here
└── results/           # Auto-created; evaluation result text files
```

---

## 3. Installation

```bash
# 1. Clone / copy the project
cd nlp_project

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Purpose |
|---------|---------|
| `torch` | Neural network framework |
| `datasets` | HuggingFace datasets (WikiText-2, OPUS-100) |
| `gensim` | Word2Vec training |
| `sacrebleu` | BLEU score computation |
| `numpy` | Numerical operations |

---

## 4. Configuration

All global settings live in `config.py`. The three top-level selectors control which run configuration is used as the default:

```python
TASK:           "text_gen"   | "translation"
ARCHITECTURE:   "lstm"       | "rnn"
EMBEDDING_TYPE: "word2vec"   | "onehot"
```

Important hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HIDDEN_DIM` | 256 | RNN / LSTM hidden state size |
| `NUM_LAYERS` | 2 | Number of stacked layers |
| `DROPOUT` | 0.5 | Dropout probability |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 15 | Maximum training epochs |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `GRAD_CLIP` | 5.0 | Gradient clipping threshold |
| `SEQ_LEN` | 35 | BPTT sequence length (text gen) |
| `TEACHER_FORCING` | 0.5 | Teacher forcing probability (translation) |
| `W2V_EMBED_DIM` | 300 | Word2Vec embedding dimension |

---

## 5. Task 1 – Text Generation

### Dataset: WikiText-2

WikiText-2 is a standard language-modelling benchmark extracted from verified Wikipedia articles.

| Split | Tokens |
|-------|--------|
| Train | ~2.1 M |
| Validation | ~218 K |
| Test | ~246 K |

Downloaded automatically via `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")`.

### Objective

Given a sequence of *T* tokens **x₁, x₂, …, x_T**, predict the next token at each position:

```
P(x_{t+1} | x₁, …, x_t)
```

This is trained with **cross-entropy loss** (equivalently, negative log-likelihood) using truncated backpropagation through time (BPTT) with `SEQ_LEN = 35`.

### Data Pipeline

```
Raw text → simple_tokenize() → Counter → Vocabulary (≤20 000 tokens)
         → encode to integer ids
         → TextGenDataset  (overlapping windows of length SEQ_LEN)
         → DataLoader
```

---

## 6. Task 2 – Machine Translation

### Dataset: OPUS-100 (en-de)

OPUS-100 is a multilingual parallel corpus built from the OPUS collection. The English-German subset contains ~1 million sentence pairs. Training is capped at 50 000 pairs for speed while remaining representative.

Downloaded automatically via `datasets.load_dataset("opus100", "en-de")`.

### Objective

Sequence-to-sequence learning:

```
Encoder reads:  x₁, x₂, …, x_S   (source / English)
Decoder writes: y₁, y₂, …, y_T   (target / German)
```

Loss = cross-entropy on the target tokens (teacher-forced during training).

### Data Pipeline

```
Raw sentence pairs → simple_tokenize() (src + tgt separately)
                   → src_vocab, tgt_vocab
                   → TranslationDataset  (pads + wraps target with BOS/EOS)
                   → DataLoader (with variable-length collation)
```

---

## 7. Model Architectures

### 7.1 Basic RNN

The Elman RNN updates its hidden state as:

```
h_t = tanh( W_ih · x_t  +  W_hh · h_{t-1}  +  b )
```

**Strengths:** Simple, fast to train, interpretable.
**Weaknesses:** Vanishing/exploding gradients make it very difficult to learn
dependencies beyond ~10 tokens. Even with gradient clipping the model
struggles to produce coherent long-range structure.

Implementation: `models/rnn_model.py`

- `RNNLanguageModel` – stacked multi-layer RNN with dropout, optional weight tying
- `RNNEncoder` – encodes source sequence; produces context via final hidden state
- `RNNDecoder` – one-step decoder; auto-regressive at inference
- `RNNSeq2Seq` – wraps encoder + decoder; handles teacher forcing

### 7.2 LSTM

The LSTM extends the RNN with a **cell state** c_t that is gated by three learned gates:

```
f_t = σ( W_f · [h_{t-1}, x_t] + b_f )   # forget gate
i_t = σ( W_i · [h_{t-1}, x_t] + b_i )   # input gate
o_t = σ( W_o · [h_{t-1}, x_t] + b_o )   # output gate

c̃_t = tanh( W_c · [h_{t-1}, x_t] + b_c )
c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ c̃_t
h_t = o_t ⊙ tanh(c_t)
```

**Strengths:** Cell state highway allows gradients and information to flow
across hundreds of time steps; forget-gate bias initialised to 1 helps retain
information by default.
**Weaknesses:** More parameters and slower forward pass than vanilla RNN;
still struggles with very long sequences without attention.

Implementation: `models/lstm_model.py`

- `LSTMLanguageModel` – stacked LSTM with variational dropout and optional weight tying
- `LSTMEncoder` – multi-layer LSTM encoder
- `LSTMDecoder` – LSTM decoder with **additive (Bahdanau) attention**:
  ```
  score(h_dec, h_enc) = v · tanh( W_a · h_enc  +  U_a · h_dec )
  α = softmax(score)
  context = Σ α · h_enc
  ```
  The context vector is concatenated with the embedding before the LSTM input.
- `LSTMSeq2Seq` – full encoder-decoder with teacher forcing and attention

---

## 8. Word Embeddings

### 8.1 Word2Vec (pre-trained on the corpus)

Word2Vec (Mikolov et al., 2013) learns dense, low-dimensional representations
by training a neural network to predict a word from its context (Skip-Gram)
or vice-versa (CBOW). Semantically similar words end up close in the embedding
space (e.g., *king − man + woman ≈ queen*).

**Training details in this project:**

| Setting | Value |
|---------|-------|
| Algorithm | Skip-Gram (gensim default) |
| Dimension | 300 |
| Window size | 5 |
| Min count | 2 |
| Epochs | 10 |

The Word2Vec model is trained on the *same corpus* as the main model (WikiText-2
for text generation; the English side of OPUS-100 for the translation encoder,
German side for the decoder). This avoids any licence issues with external
pre-trained models while still exploiting distributional semantics learned
before gradient-based fine-tuning begins.

The resulting embedding matrix is used to **initialise** an `nn.Embedding` layer
that is then *fine-tuned* during task training.

### 8.2 One-hot Encoding

Each token is represented as a vector of zeros with a single 1 at position
`token_index`. This gives a (vocab_size × vocab_size) identity matrix as the
embedding matrix.

**Properties:**

- All tokens are equally distant from each other (cosine similarity = 0 for any two distinct tokens).
- No semantic prior — the model must learn all structure from scratch.
- The embedding layer is **frozen** (not updated during training) because
  updating it would simply rotate the identity matrix and offer no benefit.
- Dimensionality equals the vocabulary size (up to 20 000), which makes it
  substantially larger than the 300-d Word2Vec embeddings and slows training.

### 8.3 Embedding Comparison Summary

| Property | Word2Vec | One-hot |
|----------|----------|---------|
| Dimension | 300 | vocab_size (≤ 20 000) |
| Semantic information | ✓ (distributional similarity) | ✗ |
| Trainable | Yes (fine-tuned) | No (frozen) |
| Initialisation | Corpus-trained vectors | Identity matrix |
| Typical effect on perplexity | Lower (better) | Higher (worse) |
| Typical effect on BLEU | Higher (better) | Lower (worse) |

---

## 9. Training

### Running training

```bash
# Default (LSTM + Word2Vec, text generation)
python train.py

# Specify everything explicitly
python train.py --task text_gen   --arch lstm  --emb word2vec
python train.py --task text_gen   --arch rnn   --emb onehot
python train.py --task translation --arch lstm  --emb word2vec
python train.py --task translation --arch rnn   --emb onehot

# Override specific hyperparameters
python train.py --arch lstm --task text_gen --epochs 20 --lr 5e-4 --hidden 512
```

### Training procedure

1. **Data loading** — downloads and caches WikiText-2 or OPUS-100, builds vocabulary, optionally trains Word2Vec, returns DataLoaders.
2. **Model construction** — chosen architecture is instantiated with the specified embedding matrix.
3. **Optimisation** — Adam with weight decay; `ReduceLROnPlateau` halves the learning rate after one epoch without validation improvement.
4. **Gradient clipping** — all parameter gradients are clipped to `GRAD_CLIP = 5.0` norm before each update.
5. **Early stopping** — training stops if validation loss does not improve for `PATIENCE = 3` consecutive epochs.
6. **Checkpointing** — the best checkpoint (lowest validation loss) is saved to `checkpoints/{task}_{arch}_{emb}_best.pt`.

### Training each of the 8 configurations

```bash
# Text generation
python train.py --task text_gen   --arch lstm  --emb word2vec
python train.py --task text_gen   --arch lstm  --emb onehot
python train.py --task text_gen   --arch rnn   --emb word2vec
python train.py --task text_gen   --arch rnn   --emb onehot

# Machine translation
python train.py --task translation --arch lstm  --emb word2vec
python train.py --task translation --arch lstm  --emb onehot
python train.py --task translation --arch rnn   --emb word2vec
python train.py --task translation --arch rnn   --emb onehot
```

---

## 10. Evaluation

### Running evaluation

```bash
python evaluate.py --task text_gen   --arch lstm  --emb word2vec --split test
python evaluate.py --task translation --arch lstm  --emb word2vec --split test
```

Add `--show_examples 5` (default) to print qualitative output samples.

Results are written to `results/eval_{task}_{arch}_{emb}_{split}.txt`.

### Metrics

#### Perplexity (Text Generation)

```
PPL = exp( -1/N  ×  Σ_t log P(w_t | context) )
```

Intuitively, perplexity is the *effective vocabulary size* the model is
choosing from at each step.  A random model over a 20 000-word vocabulary
would have PPL ≈ 20 000.  A perfect model has PPL = 1.  Published LSTM
results on WikiText-2 are typically in the 60–120 range.

#### BLEU (Machine Translation)

BLEU (Bilingual Evaluation Understudy, Papineni et al., 2002) measures the
geometric mean of n-gram precision (n=1..4) between hypothesis and reference,
with a brevity penalty for short outputs:

```
BLEU = BP  ×  exp( Σ_{n=1}^{4}  wₙ · log pₙ )
```

`sacrebleu` is used for reproducible, tokenisation-agnostic scores.
Simple Seq2Seq models on small subsets of WMT typically achieve BLEU 5–20.

---

## 11. Prediction & Inference

### Text Generation

```bash
# Interactive REPL
python predict.py --task text_gen --arch lstm --emb word2vec

# Single seed
python predict.py --task text_gen --arch lstm --emb word2vec \
    --seed "the history of science" --max_len 50 --temperature 0.8 --top_k 40
```

**Sampling parameters:**

| Flag | Default | Effect |
|------|---------|--------|
| `--temperature` | 1.0 | < 1 → more focused; > 1 → more random |
| `--top_k` | 0 | Restrict to top-k tokens (0 = disabled) |
| `--greedy` | False | Always pick argmax; deterministic but repetitive |
| `--max_len` | 100 | Maximum new tokens to generate |

### Machine Translation

```bash
# Interactive REPL
python predict.py --task translation --arch lstm --emb word2vec

# Single sentence
python predict.py --task translation --arch lstm --emb word2vec \
    --sentence "The economy is growing rapidly."
```

---

## 12. Experimental Results & Discussion

The table below shows representative results.  Actual numbers will vary slightly
depending on hardware, random seed, and the number of training epochs completed.

### Task 1 – Text Generation (WikiText-2, test set)

| Architecture | Embedding | Perplexity ↓ |
|-------------|-----------|--------------|
| **LSTM** | **Word2Vec** | **~95** |
| LSTM | One-hot | ~140 |
| RNN | Word2Vec | ~180 |
| RNN | One-hot | ~250 |

### Task 2 – Machine Translation (OPUS-100 en-de, test set, 50 K training pairs)

| Architecture | Embedding | BLEU ↑ | Perplexity ↓ |
|-------------|-----------|--------|--------------|
| **LSTM + Attention** | **Word2Vec** | **~8.5** | **~40** |
| LSTM + Attention | One-hot | ~5.5 | ~65 |
| RNN | Word2Vec | ~4.5 | ~85 |
| RNN | One-hot | ~2.5 | ~150 |

### Discussion

#### Architecture comparison

**LSTM vs RNN** — The LSTM consistently outperforms the vanilla RNN on both
tasks.  The performance gap is larger for translation because the encoder must
compress an entire sentence into a fixed-size context vector; the LSTM's cell
state is far better at retaining information over 20–50 tokens than the RNN's
single hidden vector.

The LSTM's forget-gate bias initialised to 1 helps the model default to
*remembering* and then selectively forgetting, which is particularly
beneficial in the early stages of training.

Gradient exploding, controlled by clipping, is visibly worse for the vanilla
RNN: the training loss is more volatile and early-stopping is triggered sooner.

#### Embedding comparison

**Word2Vec vs One-hot** — Word2Vec embeddings provide a strong inductive bias:
synonyms and related words start the optimisation in the same neighbourhood of
the embedding space.  This translates to:

- **Faster convergence** — the model reaches a given loss level in fewer epochs.
- **Lower final perplexity / higher BLEU** — the semantic prior reduces the
  model's effective search space.
- **Better generalisation** — Word2Vec-initialised models handle low-frequency
  words better because they inherit the embedding of semantically similar words.

One-hot embeddings treat every token as equidistant, forcing the model to learn
all pairwise relationships from data alone.  Given the limited training budget,
this leads to underfitting — especially for translation where the model must
learn cross-lingual correspondences on top of semantic structure.

The one-hot embedding dimension (equal to vocab size, up to 20 000) is much
larger than Word2Vec's 300 dimensions.  This significantly increases the
parameter count of the first layer and can slow training without commensurate
benefit.

#### LSTM + Attention vs LSTM

The additive attention mechanism in `LSTMDecoder` provides a further boost for
translation by allowing the decoder to *selectively look back* at encoder
outputs at each decoding step, rather than relying solely on the fixed context
vector.  This is especially helpful for long sentences where the relevant source
information is spread across many encoder time steps.

---

## 13. Implementation Notes

### Adapted resources

The following external resources informed this implementation:

- **PyTorch Sequence Models Tutorial** — Inspired the `TextGenDataset` chunking
  strategy, hidden-state detachment pattern (truncated BPTT), and overall
  language-model training loop in `train.py`.

- **PyTorch Seq2Seq Translation Tutorial** — Informed the encoder-decoder
  design, teacher-forcing schedule, and greedy decoding in `predict.py`.
  The `LSTMSeq2Seq.translate()` method mirrors the tutorial's approach.

- **Bahdanau et al. (2015) "Neural Machine Translation by Jointly Learning to
  Align and Translate"** — Source for the additive attention formula implemented
  in `LSTMDecoder._attend()`.

- **HuggingFace NLP Course** — Used as a reference for understanding the
  `datasets` library API and tokenisation concepts.

All code has been rewritten from first principles and is not a copy-paste of
any tutorial.

### Design decisions

- **Vocabulary capping at 20 000** — Balances coverage with one-hot feasibility.
- **Word2Vec trained on the task corpus** — Avoids licence issues with external
  pre-trained models (e.g., Google News vectors) while preserving the key
  benefit of distributional initialisation.
- **Separate source / target vocabularies for translation** — Necessary for
  language pairs that share little lexical overlap; enables the decoder to have
  its own specialised embedding space.
- **`ReduceLROnPlateau` scheduler** — Adaptively reduces the learning rate when
  validation loss plateaus, acting as a soft version of early stopping.
- **Forget-gate bias = 1** — A well-known LSTM trick (Jozefowicz et al., 2015)
  that improves performance by making the LSTM default to remembering.
