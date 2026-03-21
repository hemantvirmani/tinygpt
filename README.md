# 🧪 FemtoGPT & 🧠 TinyGPT

This repo is a learning-first GPT project. My goal is to understand how GPTs work by building a toy GPT model and progressively evolving it into a more serious implementation. I want to understand how GPT learns, not just how to run it.

We start with a minimal, readable model in `femtoGPT.py` (0.2M parameter model) which is based on Andrej Karpathy's minimal GPT, utilizing a character-level tokenizer and trained on the Shakespeare dataset. I used [this youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY) to develop this model. Its character based tokenizer is custom-made inside tis code only, as described in the video too. It has all features including Multihead attention, etc.

`tinygpt.py` will grow to a mature GPT-2/nano-gpt like transformer model (86.32M parameter as of 3/18). It uses tiktoken's GPT-2 tokenizer. This has loss of 4+ as of 3/18, which should come down as I learn better data pipelines, cleaner training loops, stronger modeling choices, and improved evaluation. 

## 💾 Checkpoints

Training saves a checkpoint every 500 steps in tinyGPT. Did not bother to save femtoGPT model yet.

## 🎛️ Reproducibility

We use a fixed random seed (`G_SEED`) so experiments are easier to compare. This seeds both CPU and CUDA (when available).

## 🎯 Current Focus

- Build a working toy GPT end-to-end.
- Keep the code simple enough to explain line-by-line.
- Iterate in small, understandable steps.

## Performance Optimizations

### Hardware Performance Optimizations :
- TensorFloat-32 (TF32) precision: set `torch.set_float32_matmul_precision('high')` for 19-bit internal matmul precision. (1000ms - 333ms per step) [DONE]
- bfloat16 mixed precision: use `torch.autocast` for faster, lower-precision training without float16 overflow issues. (drops to 300ms per step) [WIP]
- Kernel fusion via `torch.compile`: fuses ops into a single CUDA kernel to reduce overhead.  (drops to 130ms per step) [DONE]
- FlashAttention: use `torch.nn.functional.scaled_dot_product_attention` to avoid materializing the NxN attention matrix. (drops to 96ms per step) [WIP]
- "Nice" numbers (padding): increase vocab size from 50,257 to 50,304 (multiple of 128) for better GPU tile alignment. (drops to 93ms per step)

## Algorithmic Optimizations:
- AdamW optimizer with GPT-3 hyperparameters: beta1=0.9, beta2=0.95, epsilon=1e-8. [DONE - partially]
- Weight decay 0.1 on 2D parameters only (exclude biases and LayerNorm scales). [DONE]
- Cosine LR schedule: linear warmup to max LR (6e-4 for 124M), then decay to 10% of max. [DONE]
- Global grad clip 1.0 to prevent loss spikes. [DONE]
- Fused AdamW (`fused=True`) for faster updates.
- Gradient accumulation to reach large effective batch sizes without OOM. [DONE]


## 🧩 TODO (Living List, not in order of implementation)

- multi head attention: different heads learn different patterns
- Temperature sampling, Top-k / Top-p sampling
- Fine Tuning
- RLHF

## 📂 Dataset

FemtoGPT - Uses Shakespeare (Karpathy's dataset) as starting point.
TinyGPT - It uses my custom dataset (see next sub section on how its prepared) which is hosted at following two places:
* [https://huggingface.co/datasets/hemantvirmani/gpt-training-dataset](https://huggingface.co/datasets/hemantvirmani/gpt-training-dataset)
* [https://www.kaggle.com/datasets/hemantvirmani/gpt-training-dataset](https://www.kaggle.com/datasets/hemantvirmani/gpt-training-dataset)

### prepare_dataset.py

`prepare_dataset.py` builds a larger mixed dataset (WikiText-103 + OpenWebText sample), cleans it, and saves:

- `dataset.txt` (with `<|endoftext|>` separators)
- `tokens.npy` (token IDs as NumPy array)
- `dataset.bin` (binary token stream)

Tokenizer: `tiktoken` with GPT-2 encoding, and `<|endoftext|>` is allowed as a special token.

## 📈 Results - TinyGPT

### Target
Train Loss: \~1.2 -- 2.0\
Val Loss: \~1.5 -- 2.5

### 📝 Current
3/18: tinygpt.py - Loss: train 4.6248 | val 4.5250
   * G_BATCH_SIZE = 32, G_BLOCK_SIZE = 256, G_N_EMBD = 512
   * G_MAX_ITERS = 10000, G_LR = 3e-4

3/19: Femtogpt.py (Significant Scale up) - Training Loss = 1.3349 and Validation Loss: 1.6529
   * 10.8M parameters
   * G_BATCH_SIZE = 64, G_BLOCK_SIZE = 256, G_N_EMBD = 384
   * G_MAX_ITERS = 2000, G_LR = 3e-4, G_N_HEAD = 6, G_N_LAYER = 6
   * G_DROPOUT = 0.2

3/18: Femtogpt.py - Training Loss = 1.6655 and Validation Loss: 1.8924
   * 0.2M parameters
   * G_BATCH_SIZE = 16, G_BLOCK_SIZE = 32, G_N_EMBD = 64,
   * G_MAX_ITERS = 5000, G_LR = 1e-3, G_N_HEAD = 4,
   * G_N_LAYER = 4, G_DROPOUT = 0.0

#### # FemtoGPT's loss is already in the target range.

## 🚀 Learning Roadmap

Run → Understand → Control → Scale → Customize

## 🙌 Credits

* Inspired by Andrej Karpathy and GPT architectures. 
* For TinyGPT, I used ChatGPT (not codex) for helping with the starting code. 
* Both ChatGPT and Gemini have answered a bunch of stupid questions to it.
* Thanks to Kaggle for their generous 30 hrs per week free GPU access.
