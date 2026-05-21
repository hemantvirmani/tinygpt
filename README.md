# FemtoGPT & TinyGPT

This repo is meant to build hands-on intuition for attention mechanisms, embeddings, fine-tuning, etc. By the end of it, we will understand working of GPTs by building a toy model and an evolved model.

We start with a minimal, readable model in `femtogpt.py` (~10M parameter model) based on Andrej Karpathy's minimal GPT, using a character-level tokenizer trained on the Shakespeare dataset. Reference: [Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY). It includes all core features including multi-head attention.

`tinygpt.py` is a GPT-2/nanoGPT-class decoder-only Transformer with **163.04M parameters**. It uses tiktoken's GPT-2 tokenizer and was trained on FineWeb-Edu (sample-100BT).

## Checkpoints

TinyGPT saves a checkpoint every 1000 steps.

## Inference and Hosted Model

### Pretrained TinyGPT

Both formats are hosted on Hugging Face at [hemantvirmani/tinyGPT](https://huggingface.co/hemantvirmani/tinyGPT/):

- **PyTorch-native weights** (`tinygpt_pretrained_weights.pt`) — load with `tinygpt.load_model_for_inference()`
- **HuggingFace format** (`model.safetensors` + `config.json`) — load with `GPT2LMHeadModel.from_pretrained("hemantvirmani/tinyGPT")`

**Inference scripts:**

- [infer_pytorch.py](infer_pytorch.py) — loads the PyTorch-native model and runs the prompt suite used for the sample generations.
- [infer_hf.py](infer_hf.py) — loads the HuggingFace-format model from `tinygpt_pretrained_model_hf/` and runs the same prompt suite.

**Export scripts:**

- [export_to_hf.py](export_to_hf.py) — converts `tinygpt_pretrained_weights.pt` → `tinygpt_pretrained_model_hf/` (HuggingFace format).

- [pretraining_output_1.0.txt](gpt2-training-artifacts/pretraining_output_1.0.txt) contains sample outputs from PyTorch-native inference with `temperature=1.0` and `max_tokens=500`.
- [pretraining_output_0.7.txt](gpt2-training-artifacts/pretraining_output_0.7.txt) contains the better sample outputs from PyTorch-native inference with `temperature=0.7` and `max_tokens=500`.

Temperature controls sampling randomness: lower values usually make generations more conservative and coherent, while higher values add variety but can drift more.

### Instruction-Tuned TinyGPT (Alpaca)

Fine-tuned on Alpaca Cleaned 52K. Upload to Hugging Face pending.

- **PyTorch-native weights** (`tinygpt_finetuned_checkpoint_alpaca.pt`) — _not yet uploaded_
- **HuggingFace format** (`tinygpt_pretrained_model_hf_alpaca/`) — _not yet exported or uploaded_

**Export script:** [export_to_hf_alpaca.py](export_to_hf_alpaca.py) — converts `tinygpt_finetuned_checkpoint_alpaca.pt` → `tinygpt_pretrained_model_hf_alpaca/` (HuggingFace format).

## Reproducibility

A fixed random seed (`G_SEED`) seeds both CPU and CUDA so experiments are easier to compare.

## Performance Optimizations

### Hardware Performance Optimizations

- TensorFloat-32 (TF32) precision: `torch.set_float32_matmul_precision('high')` for 19-bit internal matmul precision. [DONE]
- bfloat16 mixed precision: `torch.autocast` for faster, lower-precision training without float16 overflow issues. [DONE]
- Kernel fusion via `torch.compile`: fuses ops into a single CUDA kernel to reduce overhead. [DONE]
- FlashAttention: `torch.nn.functional.scaled_dot_product_attention` to avoid materializing the NxN attention matrix. [DONE]

### Algorithmic Optimizations

- AdamW optimizer with GPT-3 hyperparameters: beta1=0.9, beta2=0.95, epsilon=1e-8. [DONE]
- Weight decay 0.1 on 2D parameters only (exclude biases and LayerNorm scales). [DONE]
- Cosine LR schedule: linear warmup to max LR, then decay to 10% of max. [DONE]
- Global grad clip 1.0 to prevent loss spikes. [DONE]
- Fused AdamW (`fused=True`) for faster updates. [DONE]
- Gradient accumulation to reach large effective batch sizes without OOM. [DONE]

## TODO (Living List, not in order of implementation)

- Fine Tuning [DONE]
- RLHF

## Retraining Notes

If pretraining TinyGPT from scratch again, the defaults in `Hyperparameters` are already calibrated — no changes needed. Key values to know:

- **Learning rate:** `lr=6e-4` (peak). Validated in attempt 4 with `effective_batch_size=512`. If you change the batch size, scale LR proportionally (linear scaling rule: 2× batch → 2× LR). The 4000-step warmup handles fresh-start gradient instability.
- **Effective batch size:** `512` sequences (`effective_batch_size=512`, split across accumulation steps). Increasing this was the single biggest lever — attempt 4 showed 16× batch improvement in 60K steps outpaced attempt 3's final 300K steps.
- **Micro-batch size (`batch_size`):** `16` — a GPU memory limit, not a training hyperparameter. Calibrated for an RTX 4090 (24 GB VRAM) with a 163M param model at sequence length 1024. A GPU with more VRAM (e.g. RTX 5090) can use a larger micro-batch (`batch_size=32` or `64`), which reduces the accumulation steps needed to hit the same effective batch of 512 — same training curve, faster wall-clock time. Formula: `accumulation_steps = effective_batch_size / batch_size`.
- **Max steps:** `max_iters=50_000`, `warmup_iters=2_500` (5% warmup). At `effective_batch_size=512`, each step processes 512 × 1,024 = 524K tokens. 50K steps ≈ 26B tokens — ~2.9× what nanoGPT used (9B tokens → val loss 2.85). TinyGPT is 31% larger (163M vs 124M params) so proportionally needs more data; 26B is a solid target. The old 600K was designed for `effective_batch_size=32` (16× smaller); at 512 it would mean 300B tokens and 3 full passes through the dataset.
- **Dataset:** FineWeb-Edu `sample-100BT` streaming (`STREAMING_HF_DATASET` / `STREAMING_HF_SUBSET` globals). This is the right dataset for a like-for-like retrain. Do not downgrade to FineWeb-Edu `sample-10BT` — attempt 2 hit a hard ceiling at val loss 3.34 because that dataset was exhausted. For dataset changes (e.g. switching to base FineWeb), see `FUTURE_ENHANCEMENTS.md`.

## Dataset

**FemtoGPT** — Uses Shakespeare (Karpathy's dataset).

**TinyGPT** — Streams directly from [FineWeb-Edu sample-100BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (~100B tokens of high-quality educational web text) during training. No offline dataset.bin required. Earlier attempts used a smaller offline dataset (~2.5 GB from FineWeb-Edu sample-10BT).

### prepare_dataset.py

`prepare_dataset.py` builds an offline mixed dataset (WikiText-103 + OpenWebText sample + FineWeb-Edu sample), cleans it, and saves `dataset.txt` and `dataset.bin`. Kept as a fallback; set `G_USE_STREAMING = False` in `tinygpt.py` to use it.

Tokenizer: `tiktoken` with GPT-2 encoding.

## Results

### FemtoGPT (~10M params, 6 layers, 6 heads, embd=384, ctx=256, char-level tokenizer)

| Config                          | Train Loss | Val Loss |
|---------------------------------|------------|----------|
| batch=64, iters=2000, lr=3e-4   | 1.3349     | 1.6529   |

### TinyGPT (163.04M params, 12 layers, 12 heads, embd=768, ctx=1024)

| Attempt | Steps | Dataset | Eff. Batch | LR | Best Val Loss | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ~60K | Offline curated corpus | 16 | 1e-5 | ~3.64 | Resumed from prior run; very low LR; dataset exhausted |
| 2 | 100K | FineWeb-Edu 10BT (~2.5 GB), offline | 32 | 3e-4 | **3.34** (step 99.4K) | Still declining at cutoff; run stopped at 100K step limit |
| 3 | 379,400 | FineWeb-Edu 100BT, streaming | 32 | 3e-4 | **3.1878** (step 376K) | Slow crawl due to small effective batch; checkpoint saved for attempt 4 |
| 4 | 59,600 | FineWeb-Edu 100BT, streaming | 512 | 6e-4 | **2.8368** (step 436.5K) | 16× batch increase; matched nanoGPT GPT-2 small benchmark |

**Key lesson across attempts:** Dataset size and effective batch size are the two most important variables. Attempts 1 and 2 were limited by dataset size. Attempt 3 was limited by gradient noise from a small effective batch (32). Scaling the batch 16× in attempt 4 delivered more improvement in 60K steps than attempt 3 did in its final 300K steps.

**nanoGPT GPT-2 small reference** (124M params, OpenWebText): val loss ~2.85. TinyGPT at 163M params on FineWeb-Edu reached **2.8368** — effectively matching the benchmark.

### Instruction Fine-Tuning — Alpaca Cleaned (52K examples)

Full fine-tune of the pretrained TinyGPT weights on the [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset using a custom PyTorch training loop. Prompt template: `### Instruction / ### Input (optional) / ### Response`. Loss masked on padding tokens.

| Setting | Value |
| --- | --- |
| Base model | TinyGPT pretrained (val loss 2.8368) |
| Dataset | Alpaca Cleaned (52K examples) |
| Dropout | 0.1 |
| Learning rate | 1e-4 (warmup + cosine decay) |
| Effective batch size | 64 (4 micro-batch × 16 accumulation steps) |
| Best val loss | **1.8405** (step 3,600 of 5,000) |
| Final weights | `tinygpt_finetuned_checkpoint_alpaca.pt` |

**Key lesson:** Dropout = 0.1 is critical at fine-tuning time — without it, the train/val gap exceeded 0.80 within 2,000 steps. With it, the gap stayed at 0.40–0.50. Format acquisition is fast (first 100 steps); factual accuracy is limited by the 163M parameter capacity.

The notebook and raw fine-tuning outputs are in [`gpt2-training-artifacts/finetuning_outputs_alpaca_dataset.md`](gpt2-training-artifacts/finetuning_outputs_alpaca_dataset.md).

## Training Story & Learning Journey

- [Training Story](gpt2-training-artifacts/tinygpt-training-story.md) — detailed account of all four pretraining attempts, loss curve analysis, and final results.
- [Learning Journey](gpt2-training-artifacts/tinygpt-learning-journey.md) — how the codebase evolved commit-by-commit, what each phase taught the model, instruction fine-tuning on Alpaca Cleaned, and open questions about Transformer architecture.

## Learning Roadmap

Run → Understand → Control → Scale → Customize

## Credits

- Inspired by Andrej Karpathy and GPT architectures.
- For TinyGPT, I used ChatGPT for helping with the starting code.
- Both ChatGPT and Gemini answered a lot of questions along the way.
- Thanks to Kaggle and RunPod for GPU access.
