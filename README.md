# 🧠 TinyGPT

This repo is a learning-first GPT project. The goal is to understand how GPTs work by building a toy GPT model and progressively evolving it into a more serious implementation.

Goal is to understand how GPT learns, not just how to run it.

We start with a minimal, readable model in `tinygpt.py` and will grow the codebase as we learn: better data pipelines, cleaner training loops, stronger modeling choices, and improved evaluation.

## 💾 Checkpoints

Training saves a checkpoint every 500 steps to `checkpoints/` as `tinygpt_step_<N>.pt`.

## 🎛️ Reproducibility

We use a fixed random seed (`G_SEED`) so experiments are easier to compare. This seeds both CPU and CUDA (when available).

## 🎯 Current Focus

- Build a working toy GPT end-to-end.
- Keep the code simple enough to explain line-by-line.
- Iterate in small, understandable steps.

## 🧩 TODO (Living List, not in order of implementation)

- Add a proper dataset loader with train/val splits, track train & val loss
- Improve Reproducibility (torch.manual_seed), Logging
- Add dropout layer
- multi head attention: different heads learn different patterns
- Temperature sampling, Top-k / Top-p sampling
- Improve tokenization and larger dataset - Use WikiText?
- Experiment with optimizer and Learning Rates
- Fine Tuning
- RLHF

## ðŸ“‚ Dataset

Uses Tiny Shakespeare by default.

## ðŸ“ˆ Expected Results

Train Loss: \~1.2 -- 2.0\
Val Loss: \~1.5 -- 2.5

## ðŸš€ Learning Roadmap

Run â†’ Understand â†’ Control â†’ Scale â†’ Customize

## ðŸ™Œ Credits

Inspired by Andrej Karpathy and GPT architectures. Used ChatGPT for the code

## 📝 My Notes
Training Loss = 2.67 with these params on 3/16/2026:
    G_BATCH_SIZE = 16
    G_BLOCK_SIZE = 64
    G_N_EMBD = 128
    G_MAX_ITERS = 5000
    G_LR = 1e-3

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
- Add CLI smoke test for `--resume`
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
- CLI smoke test for `--resume`
- CPU/GPU device handling
