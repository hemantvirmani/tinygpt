# 🧠 TinyGPT

This repo is a learning-first GPT project. The goal is to understand how GPTs work by building a toy GPT model and progressively evolving it into a more serious implementation.

Goal is to understand how GPT learns, not just how to run it.

We start with a minimal, readable model in `tinygpt.py` and will grow the codebase as we learn: better data pipelines, cleaner training loops, stronger modeling choices, and improved evaluation.

Tokenization uses GPT-2's `tiktoken` encoding.

## 🧪 FemtoGPT

FemtoGPT will use the Shakespeare dataset and will be a smaller version of GPT. It will use a super basic tokenizer and move away from GPT-2.

This will be a practice project for Karpathy's "Let's build the GPT Tokenizer".

```text
https://www.youtube.com/watch?v=zduSFxRajkE
```

## 💾 Checkpoints

Training saves a checkpoint every 500 steps to `checkpoints/` as `tinygpt_step_<N>.pt`.

## 🎛️ Reproducibility

We use a fixed random seed (`G_SEED`) so experiments are easier to compare. This seeds both CPU and CUDA (when available).

## 🎯 Current Focus

- Build a working toy GPT end-to-end.
- Keep the code simple enough to explain line-by-line.
- Iterate in small, understandable steps.

## 🧩 TODO (Living List, not in order of implementation)

- Add dropout layer
- multi head attention: different heads learn different patterns
- Temperature sampling, Top-k / Top-p sampling
- Fine Tuning
- RLHF

## 📂 Dataset

Uses Shakespeare (Karpathy's dataset) as starting point.

### prepare_dataset.py

`prepare_dataset.py` builds a larger mixed dataset (WikiText-103 + OpenWebText sample), cleans it, and saves:

- `dataset.txt` (with `<|endoftext|>` separators)
- `tokens.npy` (token IDs as NumPy array)
- `dataset.bin` (binary token stream)

Tokenizer: `tiktoken` with GPT-2 encoding, and `<|endoftext|>` is allowed as a special token.

## 📈 Expected Results

Train Loss: \~1.2 -- 2.0\
Val Loss: \~1.5 -- 2.5

## 🚀 Learning Roadmap

Run → Understand → Control → Scale → Customize

## 🙌 Credits

Inspired by Andrej Karpathy and GPT architectures. Used ChatGPT for the code

## 📝 My Notes
3/18: tinygpt.py - Loss: train 5.1001 | val 5.1144
    G_BATCH_SIZE = 32
    G_BLOCK_SIZE = 256
    G_N_EMBD = 512
    G_MAX_ITERS = 10000
    G_LR = 1e-4

3/17: Femtogpt.py - Training Loss = 3.829 and Validation Loss: 4.6121
    G_BATCH_SIZE = 64
    G_BLOCK_SIZE = 64
    G_N_EMBD = 128
    G_MAX_ITERS = 2000
    G_LR = 3e-4

## 🧪 Test Creation Plan

- Use `pytest` as the test runner.
- Add a tiny fixture dataset under `tests/fixtures/` to avoid network calls.
- Add unit tests for:
  - `build_state` (tokenizer, data shape, vocab size)
  - `get_batch` (shapes, device, x/y shift)
  - `SelfAttention` mask behavior and shape
  - `Block` shape preservation
  - `TinyGPT` forward output shape
- Add training loop smoke test:
  - One or two steps produce finite loss and optimizer step runs
- Add checkpoint tests:
  - Checkpoints are created at `save_every`
  - Checkpoint contents include `step`, `model_state`, `optimizer_state`, `vocab_size`
- Add resume tests:
  - Resume loads without error and continues from saved step
- Add generation tests:
  - `generateText` returns a non-empty string and length grows with `max_tokens`
- Add CLI smoke test for `--checkpoint`
- Add a GitHub Actions workflow to run tests on CPU

## 📋 Test Plan Backlog

Keep this plan here so we can pick it up next week.

- State and data setup
- Batching shapes, device placement, and x/y shift correctness
- Self-attention masking and shape preservation
- Block and model forward shapes
- Training loop: loss is finite and optimizer step runs
- Checkpoint save content and cadence
- Resume logic restores model and optimizer state
- Text generation length and basic sanity checks
- CLI smoke test for `--checkpoint`
- CPU/GPU device handling
