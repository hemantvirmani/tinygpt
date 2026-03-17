# TinyGPT

This repo is a learning-first GPT project. The goal is to understand how GPTs work by building a toy GPT model and progressively evolving it into a more serious implementation.

We start with a minimal, readable model in `tinygpt.py` and will grow the codebase as we learn: better data pipelines, cleaner training loops, stronger modeling choices, and improved evaluation.

## Current Focus

- Build a working toy GPT end-to-end.
- Keep the code simple enough to explain line-by-line.
- Iterate in small, understandable steps.

## TODO (Living List)

- Add a proper dataset loader with train/val splits.
- Add logging for loss curves and samples.
- Improve tokenization and dataset preparation.
- Add attention masking tests and shape checks.
- Add basic eval metrics and checkpoints.
- Support longer contexts and larger models.
- Experiment with optimizer and LR schedules.
- Document training commands and expected outputs.


## 📂 Dataset

Uses Tiny Shakespeare by default.

## 📈 Expected Results

Train Loss: \~1.2 -- 2.0\
Val Loss: \~1.5 -- 2.5

# 🚀 Learning Roadmap

Run → Understand → Control → Scale → Customize

## 🙌 Credits

Inspired by Andrej Karpathy and GPT architectures.
