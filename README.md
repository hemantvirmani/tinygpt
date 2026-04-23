# FemtoGPT & TinyGPT

This repo is a learning-first GPT project. My goal is to understand how GPTs work by building a toy GPT model and progressively evolving it into a more serious implementation. I want to understand how GPT learns, not just how to run it.

We start with a minimal, readable model in `femtoGPT.py` (0.2M parameter model) which is based on Andrej Karpathy's minimal GPT, utilizing a character-level tokenizer and trained on the Shakespeare dataset. I used [this youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY) to develop this model. Its character based tokenizer is custom-made inside its code only, as described in the video too. It has all features including Multihead attention, etc.

`tinygpt.py` is a GPT-2/nano-gpt like transformer model with **163.04M parameters**. It uses tiktoken's GPT-2 tokenizer.

## Checkpoints

Training saves a checkpoint every 1000 steps in tinyGPT. Did not bother to save femtoGPT model yet.

## Reproducibility

We use a fixed random seed (`G_SEED`) so experiments are easier to compare. This seeds both CPU and CUDA (when available).

## Current Focus

- Build a working toy GPT end-to-end.
- Keep the code simple enough to explain line-by-line.
- Iterate in small, understandable steps.

## Performance Optimizations

### Hardware Performance Optimizations
- TensorFloat-32 (TF32) precision: set `torch.set_float32_matmul_precision('high')` for 19-bit internal matmul precision. [DONE]
- bfloat16 mixed precision: use `torch.autocast` for faster, lower-precision training without float16 overflow issues. [DONE]
- Kernel fusion via `torch.compile`: fuses ops into a single CUDA kernel to reduce overhead. [DONE]
- FlashAttention: use `torch.nn.functional.scaled_dot_product_attention` to avoid materializing the NxN attention matrix. [DONE]

### Algorithmic Optimizations
- AdamW optimizer with GPT-3 hyperparameters: beta1=0.9, beta2=0.95, epsilon=1e-8. [DONE]
- Weight decay 0.1 on 2D parameters only (exclude biases and LayerNorm scales). [DONE]
- Cosine LR schedule: linear warmup to max LR, then decay to 10% of max. [DONE]
- Global grad clip 1.0 to prevent loss spikes. [DONE]
- Fused AdamW (`fused=True`) for faster updates. [DONE]
- Gradient accumulation to reach large effective batch sizes without OOM. [DONE]


## TODO (Living List, not in order of implementation)

- Temperature sampling, Top-k / Top-p sampling
- Fine Tuning
- RLHF

## Dataset

**FemtoGPT** — Uses Shakespeare (Karpathy's dataset).

**TinyGPT** — Streams directly from [FineWeb-Edu sample-100BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (~100B tokens of high-quality educational web text) during training. No offline dataset.bin required. The previous offline dataset (~1.25B tokens from WikiText-103 + 2% OpenWebText + 10% FineWeb-Edu sample-10BT) is still available on HuggingFace/Kaggle for reference.

### prepare_dataset.py

`prepare_dataset.py` builds an offline mixed dataset (WikiText-103 + OpenWebText sample + FineWeb-Edu sample), cleans it, and saves `dataset.txt` and `dataset.bin`. Kept as a fallback; set `G_USE_STREAMING = False` in `tinygpt.py` to use it.

Tokenizer: `tiktoken` with GPT-2 encoding.

## Results

### Target
Train Loss: ~2.0–2.5  
Val Loss: ~2.0–2.5

### FemtoGPT

| Date | Params | Config | Train Loss | Val Loss |
|------|--------|--------|-----------|---------|
| 3/19 | 10.8M | batch=64, ctx=256, embd=384, layers=6, heads=6, iters=2000, lr=3e-4 | 1.3349 | 1.6529 |

FemtoGPT loss is already in the target range.

### TinyGPT (163.04M params, 12 layers, 12 heads, embd=768, ctx=1024)

| Attempt | Steps | Dataset | Eff. Batch | Train Loss | Val Loss | Notes |
|---------|-------|---------|-----------|-----------|---------|-------|
| 1 (3/27) | 100k | Offline ~1.25B tokens (10% FineWeb-Edu sample-10BT + WikiText + OWT) | 16 | 3.2916 | 3.6472 | Val loss diverging — overfitting on finite dataset |
| 2 (4/7) | 100k | Streaming FineWeb-Edu sample-100BT | 32 | 3.4189 | 3.4164 | Train ≈ val, no overfitting. Larger effective batch stabilizes training |
| 3 (in progress) | — | Streaming FineWeb-Edu sample-100BT | 32 | — | — | Target: 600k steps (~10B tokens) |

**Key takeaway from attempt 1→2:** Switching from a finite offline dataset to streaming eliminated the train/val gap entirely. The model now sees fresh data every step and cannot overfit.

**Expected trajectory (attempt 3):**

| Steps | Est. Train Loss |
|-------|----------------|
| 100k  | ~3.3–3.4 |
| 200k  | ~3.0–3.1 |
| 400k  | ~2.7–2.8 |
| 600k  | ~2.5–2.6 |

## Learning Roadmap

Run → Understand → Control → Scale → Customize

## Credits

* Inspired by Andrej Karpathy and GPT architectures.
* For TinyGPT, I used ChatGPT (not codex) for helping with the starting code.
* Both ChatGPT and Gemini have answered a bunch of stupid questions to it.
* Thanks to Kaggle and RunPod for GPU access.
