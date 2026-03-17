# TinyGPT

This repo is a learning-first GPT project. The goal is to understand how GPTs work by building a toy GPT model and progressively evolving it into a more serious implementation.

Goal is to understand how GPT learns, not just how to run it.

We start with a minimal, readable model in `tinygpt.py` and will grow the codebase as we learn: better data pipelines, cleaner training loops, stronger modeling choices, and improved evaluation.

## Current Focus

- Build a working toy GPT end-to-end.
- Keep the code simple enough to explain line-by-line.
- Iterate in small, understandable steps.

## TODO (Living List, not in order of implementation)

- Add a proper dataset loader with train/val splits, track train & val loss
- Improve Reproducibility (torch.manual_seed), Logging
- Add dropout layer
- multi head attention: different heads learn different patterns
- Temperature sampling, Top-k / Top-p sampling
- Improve tokenization and larger dataset - Use WikiText?
- Experiment with optimizer and Learning Rates
- Fine Tuning
- RLHF


## 📂 Dataset

Uses Tiny Shakespeare by default.

## 📈 Expected Results

Train Loss: \~1.2 -- 2.0\
Val Loss: \~1.5 -- 2.5

# 🚀 Learning Roadmap

Run → Understand → Control → Scale → Customize

## 🙌 Credits

Inspired by Andrej Karpathy and GPT architectures. Used ChatGPT for the code
