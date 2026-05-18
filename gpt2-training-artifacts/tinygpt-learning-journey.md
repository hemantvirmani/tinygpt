# Building a GPT-2–Class Model from Scratch: A Learning Journey

Late March – early May 2026 | Two months, two models, one 163M-parameter language model that matches GPT-2 small

---

## The Destination

This is a from-scratch implementation of a 163M-parameter GPT-2–class language model, built over two months across two models, four training attempts, and ~43.6 billion tokens of training data. The final model — TinyGPT — reached a validation loss of **2.8368** (perplexity 17.1), matching Karpathy's nanoGPT reproduction of GPT-2 small: the canonical benchmark for this model class.

The repo also includes **FemtoGPT**, a ~10M parameter character-level model on Shakespeare that served as the fast-iteration testbed throughout development.

| Parameter | Value |
| --- | --- |
| Architecture | GPT-2 (decoder-only Transformer) |
| Parameters | **163.04M** (162.9M decayed + 125K non-decayed) |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Context length | 1,024 tokens |
| Vocabulary | GPT-2 tokenizer (50,257 tokens) |
| Dataset | HuggingFaceFW/fineweb-edu (sample-100BT) |
| Attention | Flash attention (scaled dot-product) |
| Precision | bfloat16 |

![TinyGPT Training Loss Curve](tinygpt_loss_curve.png)

---

## Key Technical Learnings

1. **Dataset size before architecture.** Attempting to scale a 163M parameter model on a small dataset is wasted compute. The ceiling is set by the data, not the model. Attempts 1 and 2 hit hard ceilings at ~3.6 and ~3.4 respectively simply because the training data ran out before the model did.

2. **Batch size is a first-class hyperparameter.** Phase 1 spent 379k steps with batch=32 and reached val loss 3.19. Phase 2 spent 60k steps with batch=512 and surpassed that by 0.35 nats. With a noisy gradient, more steps doesn't necessarily mean more learning — each Phase 2 step processed 16× more tokens and produced 16× more reliable gradient estimates.

3. **Given sufficient data and the right architecture, training loss keeps falling until you pull the plug.** At the point training was halted (step 438,700), new val loss lows were still being set. There was no plateau, no diminishing returns — the model stops improving when *you* stop, not before. A continuation to 600k steps would have improved the model further.

4. **Checkpointing is infrastructure, not an afterthought.** Saving the full training state — model weights, optimizer state, scheduler state, step count — is what makes long multi-week runs recoverable. A scheduler resume bug discovered mid-run was a real example of what happens when state is partially saved.

5. **The validation loss gap reveals overfitting.** The narrow train/val gap (~0.05–0.10) throughout the TinyGPT run confirmed the model was generalizing. When that gap widens, you have a problem.

6. **Hardware-aware optimization compounds.** Flash attention, bfloat16, gradient clipping, `torch.compile` — none of these changes the model's architecture. They only change how efficiently the same computation runs. Together they made the difference between a training run that's feasible and one that isn't.

7. **The learning curve is always front-loaded.** The model improves dramatically in the first 1% of training and incrementally for the remaining 99%. This is universal — it's the nature of gradient descent on language models. The early losses are "easy" (learning that `the` is common), while the late losses require learning subtle relationships that only appear in rare contexts.

---

## All Attempts at a Glance

| Attempt | Steps | Dataset | Eff. Batch | LR | Best Val Loss | What stopped it |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ~60K | Offline curated corpus | 16 | 1e-5 | ~3.64 | Dataset exhausted; LR too low |
| 2 | 100K | FineWeb-Edu 10BT (~2.5 GB), offline | 32 | 3e-4 | **3.34** (step 99.4K) | Slow decline · convergence stalled |
| 3 | 379,400 | FineWeb-Edu 100BT, streaming | 32 | 3e-4 | **3.1878** (step 376K) | Gradient noise from small batch |
| 4 | 59,600 | FineWeb-Edu 100BT, streaming | 512 | 6e-4 | **2.8368** (step 436.5K) | Matched benchmark; training stopped |

---

## What is a Language Model, Actually?

Before tracing the journey, here's the core idea — explained so a high school student can follow along.

A language model is a machine that has learned to predict: *given these words, what word comes next?* That's it. The entire capability of GPT-2, ChatGPT, or any large language model comes from doing this prediction task over and over, on billions of sentences, until the model builds an internal representation of how language works.

Think of it like autocomplete on your phone, but trained on far more text, with a far more powerful architecture that can hold more context in mind. When you type a question into ChatGPT, it's not "looking up" an answer — it's doing billions of tiny probability calculations that together produce the most likely next token, one at a time.

**Tokens** are the atoms the model works with. Rather than individual characters or words, modern models use *subword tokens* — pieces of words. The word "unbelievable" might become `un` + `believ` + `able`, each mapped to a number. The GPT-2 tokenizer has 50,257 unique tokens.

### The Architecture in One Paragraph

A Transformer language model has three key stages:

1. **Embedding:** Convert each token number into a vector of numbers (768 numbers in GPT-2 small). This is the model's "working representation" of a token.
2. **Transformer blocks (repeated N times):** Each block has two sub-components:
   - *Attention*: Every position in the sequence looks at every other position and asks: "which earlier tokens are relevant to understanding me right now?" This is how the model learns that "the bank" near "river" means something different than "the bank" near "money."
   - *Feed-forward network*: A small neural network that processes each position independently, adding more expressive capacity.
3. **Output head:** Project back to vocabulary size (50,257 numbers) and pick the most likely next token.

The magic is that none of this was hand-coded with rules. The model learns what to pay attention to, what matters, and how to respond — all from the training signal: *was my prediction right or wrong?*

---

## How This Started

This wasn't a project that was planned end-to-end. It started as a single file — `tinygpt.py` — with a comment that said *"adding basic model. lot more needs to be done."* That first commit was in late March 2026, and over the next eight weeks the codebase grew into a production-quality training run on a 100-billion-token dataset.

The commit history tells the story better than any retrospective could. Every bug fixed, every paper read, every training run that hit a wall and had to be rethought — it's all there in the log.

---

## Phase 1: First Code (late March 2026)

The first commit added a basic model. The structure was already recognizable as a Transformer — embedding layer, attention, feed-forward — but many things that a production training loop needs were missing. The commits that followed in rapid succession fill in the gaps:

- **Checkpointing:** The ability to save model weights to disk and resume from them. Without this, every training run starts from scratch. This seems obvious in retrospect, but it's the first taste of a recurring theme: *training is expensive, so protect your work.*

- **Train/val split:** Splitting data into training and validation sets, and measuring validation loss separately. This is how you distinguish "the model is learning" from "the model is memorizing." If training loss falls but validation loss stays high, the model is overfitting — learning to recite training data rather than generalizing.

- **Reproducibility seed:** Setting a fixed random seed ensures that runs can be reproduced exactly. Small discipline, but essential when debugging.

- **Code cleanup:** A round of commits that stripped out over-configuration and bloat. A sign that the code had gotten complex enough to need trimming.

What the model was learning to do at this stage: nothing — it wasn't training yet. These are the scaffolding commits.

---

## Phase 2: The Fork — FemtoGPT and TinyGPT Diverge (late March 2026)

This is the moment the codebase split into two files with different purposes.

**FemtoGPT** (`femtogpt.py`) became a small, educational model:

- Shakespeare text as the training dataset (~1MB of text)
- Character-level tokenizer: every unique character in Shakespeare is a "token" (~65 tokens total)
- ~10M parameters: 384-dimensional embeddings, 6 layers, 6 attention heads
- Purpose: learn the mechanics in a fast feedback loop

**TinyGPT** (`tinygpt.py`) remained the production target:

- Web-scale dataset via HuggingFace
- GPT-2's subword tokenizer (tiktoken), 50,257 tokens
- 163M parameters: 768-dimensional embeddings, 12 layers, 12 heads
- Purpose: actually match GPT-2 quality

This split was the right call. FemtoGPT trains in minutes on a laptop and gives immediate feedback when you change something. TinyGPT takes weeks on a GPU. Iterating on architecture and training ideas on FemtoGPT, then promoting the changes to TinyGPT, was the pattern that worked.

At the same time, TinyGPT got the GPT-2 subword tokenizer (replacing an HF tokenizer), basic loss plotting was added — the first time training progress became visible as a curve rather than just numbers — and the codebase started being tested on a dataset uploaded to HuggingFace Hub.

---

## Phase 3: Building the Real Training Loop (late March 2026)

Three important additions:

### Learning Rate Scheduling

The first version trained with a fixed learning rate. A fixed LR is fine for small experiments but wrong for serious training. Two improvements:

- **Warmup:** Start with a tiny learning rate and ramp it up over the first several hundred steps. Why? At the start of training, the model's weights are random and the gradients are chaotic. Taking large steps on chaotic gradients is counterproductive. Starting small lets the model "find its footing" before committing to bigger updates.

- **Cosine annealing:** After warmup, the learning rate follows a cosine curve down to a small minimum. The idea is that early in training you want to explore the loss landscape quickly (high LR), but as you converge, you want to fine-tune carefully (low LR). The cosine shape gives a smooth, natural decay.

Think of it like learning to ride a bike: you take it slowly at first to find your balance, then pedal harder once you've got it, then ease up as you approach a stop.

### Dropout

Dropout randomly switches off 20% of neurons during each training step. This forces the model not to rely on any single path — every neuron has to be useful even when some of its neighbors are temporarily absent. The result is a model that generalizes better rather than memorizing the training data.

### Refactoring into a Class

The training code was reorganized into a `TinyGPT` class with proper methods for batch loading, training setup, and the training loop. This is the code getting serious — a clean class structure makes it easier to save and resume state, add inference separately from training, and reason about what the code does.

---

## Phase 4: FemtoGPT Reaches a Stable State (late March 2026)

FemtoGPT hit a milestone: *"a good version of femto GPT — used Andrej's hyperparams and multi-head attention."* This matched Karpathy's nanoGPT tutorial hyperparameters exactly, producing a small model that could generate Shakespeare-like text after a short training run.

Shortly after came *"significant scale up femto model to get better performance"* — bumping the model size and training longer. FemtoGPT was now a reliable testbed.

What FemtoGPT could do at this point: generate plausible-looking Elizabethan text. Not great, not coherent over long passages, but character-by-character it sounded Shakespearean. The model had learned:

- Which characters follow which (spelling patterns)
- Common words and their frequency
- Basic sentence structure — linebreaks, punctuation patterns
- Speaker turns (in dramatic texts)

It had not learned: meaning, coherence across sentences, or any real understanding of the content.

---

## Phase 5: Bringing TinyGPT Up to GPT-2 Architecture (late March 2026)

The most technically dense period in the commit history. Several commits in rapid succession, each adding a significant piece of the GPT-2 architecture to TinyGPT.

### Multi-Head Attention

The first attention implementation was a single attention head. Multi-head attention runs several attention mechanisms in parallel, each learning to attend to different aspects of the input. One head might learn syntactic relationships (subject-verb agreement), another might track pronoun references, another might handle semantic similarity. The outputs are concatenated and projected.

### Gradient Accumulation + GPT-2 Hyperparameters

**Gradient accumulation** is a trick for training with a large effective batch size when you don't have enough GPU memory to hold the full batch at once. Instead of computing one large gradient update, you compute many small gradients and *accumulate* (sum) them before doing the optimizer step. The math is equivalent to training with a larger batch — the model sees the same information, just in pieces.

This also set the hyperparameters to match GPT-2 small: 768-dimensional embeddings, 12 layers, 12 heads. TinyGPT was now architecturally identical to GPT-2 small (plus 39M extra parameters from being a slightly larger variant).

### CausalSelfAttention with Fused QKV

The original multi-head attention used separate linear layers for queries, keys, and values. GPT-2's actual implementation uses a **single fused projection** that computes Q, K, and V in one matrix multiply, then splits the output. This is more efficient (one large operation vs. three smaller ones) and was Karpathy's preferred implementation.

"Causal" refers to the masking: a position can only attend to itself and positions *before* it, never after. This is what makes autoregressive generation possible — at each step, the model only sees what has come before.

### Bias=False

Following Karpathy's recommendation: removing biases from the linear layers. In a large Transformer, biases add parameters without meaningful capacity — the layer norm that follows handles any offset correction anyway. Removing them slightly reduces memory and parameter count.

---

## Phase 6: Hardware Optimizations (late March 2026)

A cluster of optimizations from Karpathy's video series on reproducing GPT-2:

- **Flash Attention:** Rather than computing the full `N×N` attention matrix (which grows quadratically with sequence length), Flash Attention uses a memory-efficient algorithm that processes attention in tiles. On modern GPUs, this is significantly faster and uses less memory. In PyTorch, this is one line: `F.scaled_dot_product_attention`.

- **bfloat16 mixed precision:** Neural networks don't need 32-bit floating point precision. Training in bfloat16 (16-bit, but with the same exponent range as float32) roughly halves memory usage and increases throughput on modern GPUs that have dedicated bfloat16 hardware.

- **Gradient clipping:** If a gradient update is unusually large (which can happen when the model encounters a confusing batch), it can destabilize training by pushing weights too far. Clipping the gradient norm to a maximum value (1.0) prevents these "gradient explosions."

- **Weight decay:** Adds a small penalty to the optimizer for large weights, encouraging the model to use many small weights rather than a few large ones. This regularizes the model and often improves generalization.

- **`torch.compile`:** PyTorch 2.0's compiler traces the computation graph and generates optimized CUDA code, reducing Python overhead and improving GPU utilization.

Each of these individually gives a few percent improvement. Together they add up to a 30–50% speedup and a more stable training run — which matters enormously when you're paying for GPU time.

---

## Phase 7: First Real Training Runs (late March – early April 2026)

A separate inference path was added — separating the generate function from the training loop. Training and inference have different needs (inference doesn't need gradients), and keeping them separate makes both cleaner.

Shortly after: three commits that improved robustness:

- Saving the learning rate scheduler state to checkpoints so training can resume with the correct LR rather than restarting from warmup.
- Increasing the number of batches used to estimate validation loss. More samples = more reliable estimate.
- Increasing the micro-batch size.

These are the small-but-important infrastructure commits that happen when you've started real long runs and discovered what breaks when you let things run for hours.

This period also covers Attempt 1 and Attempt 2 — the first real training runs, both of which hit dataset ceilings:

- **Attempt 1** (~60K steps, resumed from a prior run): val loss reached ~3.64 and stopped improving. The dataset was too small, and the learning rate (1e-5) was too conservative to push through the noise.
- **Attempt 2** (100K steps, FineWeb-Edu 10BT, ~2.5 GB, offline): a much better run. Val loss fell to 4.03 by step 10,000, then ground slowly — 3.70 by step 30,000, 3.51 by step 50,000 — to a minimum of **3.34** at step 99,400. The run hit the 100K step limit with val 3.4164 — still declining, not truly stalled. The ~2.5 GB dataset was the ceiling.

Both runs taught the same lesson: the ceiling isn't the architecture or the optimizer. It's the data.

---

## Phase 8: The Switch to FineWeb-Edu 100BT (late April 2026)

This is where Attempt 3 began.

The fix was **FineWeb-Edu (sample-100BT)** — a 100-billion-token dataset of filtered educational content from the web, assembled by HuggingFace. Streaming it meant the model never actually runs out of data; it's always seeing new text. The code change was enabling HuggingFace streaming and pointing to the new dataset.

The dataset lesson is perhaps the most important practical insight from this project: **no amount of architectural cleverness or hyperparameter tuning can compensate for an undersized dataset.** The model's capacity to learn is gated by how much diverse text it has to learn from.

---

## Phase 9: The Crawl and the Fix (late April – early May 2026)

Attempt 3 ran for **379,400 steps** on FineWeb-Edu 100BT with conservative settings:

| Setting | Value |
| --- | --- |
| Learning rate | 3e-4 |
| Effective batch size | 32 sequences (2 accumulation steps × 16 micro-batch) |
| Tokens per step | 32 × 1,024 = **32,768** |
| LR schedule | Linear warmup (4,000 steps) → Cosine decay to 3e-5 |

### The Fast Descent (Steps 1–10,000)

The model learned the basics of language almost immediately. Starting from random weights, it fell to **~3.92** within the first 10,000 steps — learning word co-occurrence, basic grammar, and common phrases.

| Step | Val loss | Perplexity |
| --- | --- | --- |
| 1,000 | 6.10 | ~445 |
| 2,000 | 5.36 | ~213 |
| 5,000 | ~4.15 | ~63 |
| 10,000 | ~3.92 | ~50 |

### The Long Crawl (Steps 10,000–379,000)

After the initial drop, progress slowed to a crawl. The model spent the next **369,000 steps** — over 97% of the run — inching from val loss ~3.92 down to ~3.30. Each additional 4,000 steps bought only ~0.2 nats of improvement, with high variance masking real progress.

Average val loss by phase (5k-step windows):

| Step range | Avg val loss | Change |
| --- | --- | --- |
| 5k–30k | ~3.65 | −0.50 |
| 30k–100k | ~3.56 | −0.09 |
| 100k–200k | ~3.47 | −0.09 |
| 200k–300k | ~3.40 | −0.07 |
| 300k–379k | **~3.32** | −0.08 |

**Why so slow?** The culprit was the small effective batch size. With only 32 sequences per step (32,768 tokens), each gradient estimate was noisy. The model was learning, but each update step was less reliable — two steps forward, one step back. The val loss curve had high variance, and the model spent enormous compute essentially spinning in place.

The best val loss reached was **3.1878** at step 376,000. This checkpoint was saved and became the starting point for Attempt 4.

### Scaling Up: Attempt 4

Resuming from step 379,000 with two key changes:

| Setting | Attempt 3 | Attempt 4 | Change |
| --- | --- | --- | --- |
| Learning rate | 3e-4 | **6e-4** | 2× |
| Effective batch size | 32 | **512** | 16× |
| Tokens per step | 32,768 | **524,288** | 16× |
| Accumulation steps | 2 | 32 | 16× |

The LR increase follows the linear scaling rule: doubling the batch size warrants doubling the LR. Staying at 2× LR (rather than the full 16×) was the conservative choice for a mid-training resume.

The effect was visible within the first few thousand steps. The val loss, which had been hovering around 3.10–3.30 for hundreds of thousands of steps, began falling cleanly:

| Step | Val loss | Perplexity |
| --- | --- | --- |
| 379,100 (start) | 3.09 | ~22.0 |
| 395,000 | 2.95 | ~19.1 |
| 410,000 | 2.95 | ~19.1 |
| 420,000 | 2.93 | ~18.8 |
| 430,000 | **2.88** | ~17.8 |
| 436,500 | **2.8368** | **17.1** |

In just **59,600 steps**, the model improved by ~0.26 nats — more than Attempt 3 achieved in its final *300,000 steps*. Larger batches give better gradient estimates. Better estimates mean more reliable steps forward. The model stops "wandering" and starts actually converging.

Training was stopped at step **438,700**, having crossed the nanoGPT GPT-2 small benchmark (val loss ~2.85):

| Metric | Value | Step |
| --- | --- | --- |
| Best validation loss | **2.8368** | 436,500 |
| Best training loss | **2.6744** | 436,700 |
| Final val loss (last point) | 2.9–3.07 (noisy) | 438,700 |
| Best perplexity | **~17.1** | 436,500 |

The train/val gap remained narrow (~0.05–0.10) throughout, confirming no overfitting.

---

## Where the Model Landed

Final best validation loss: 2.8368 at step 436,500 (~17.1 perplexity)

The canonical benchmark is Karpathy's nanoGPT reproduction of GPT-2 small (124M parameters), which reaches ~val loss 2.85 on OpenWebText. TinyGPT (163M parameters, trained on fineweb-edu) reached **2.8368** — effectively matching, and marginally beating, the reference.

Two important caveats on that comparison:
1. TinyGPT has **163M parameters vs. 124M** for GPT-2 small — a ~31% larger model, so a slight edge is expected.
2. The datasets differ: fineweb-edu is a filtered, higher-quality educational subset of Common Crawl, while OpenWebText mirrors Reddit-curated web text. Educational text is more structured and predictable, which typically yields lower perplexity, so the numbers aren't directly comparable.

Perplexity of 17.1 means: on average, the model is as "surprised" as if it had to choose between 17 equally likely options at each step. A random model over 50,000 tokens would have perplexity 50,000. A perfect model would have perplexity 1.

| Phase | Steps | Tokens/step | Total tokens |
| --- | --- | --- | --- |
| Attempt 3 | 379,400 | 32,768 | ~12.4B |
| Attempt 4 | 59,600 | 524,288 | ~31.2B |
| **Total** | **439,000** | — | **~43.6B** |

The model had seen ~43.6 billion tokens total — roughly 13× what the Chinchilla scaling laws suggest is optimal for a 163M parameter model (163M × 20 ≈ 3.3B tokens). This "over-training" is intentional: a smaller, over-trained model is cheaper to run at inference time than a larger model trained to the compute-optimal point, while achieving similar quality.

---

## What Each Training Phase Actually Taught the Model

Here's a high-level map of what the model was learning at different stages — in terms a high school student can follow:

| Steps | Val Loss | What the model learned |
| --- | --- | --- |
| 1,000 | ~6.1 | Basic statistics: some letters/words appear more than others |
| 5,000 | ~4.15 | Common words, basic spelling, which words appear near each other |
| 10,000 | ~3.92 | Rough sentence structure; punctuation patterns; common phrases |
| 50,000 | ~3.56 | Grammar patterns, paragraph structure, topic coherence within a sentence |
| 100,000 | ~3.47 | More consistent vocabulary per topic, basic logic within a sentence |
| 200,000 | ~3.40 | Better multi-sentence coherence, factual patterns from educational text |
| 379,000 | ~3.19 | Solid general language model; reads as coherent at the sentence level |
| 436,500 | **2.84** | Near GPT-2 small quality; paragraph-level coherence, factual text, structured reasoning |

The early steps (1–10k) are where the most dramatic learning happens. Everything after that is the model getting progressively *less wrong* about edge cases, uncommon words, long-range dependencies, and nuanced structure.

---

## What I Still Wonder

### How did the "Attention is All You Need" authors develop the original intuition?

It's easy to understand the Transformer *after* reading dozens of explainers. But the 2017 paper replaced recurrent networks (RNNs/LSTMs, which processed sequences step-by-step, like reading a sentence one word at a time) with something that seemed audacious at the time: every position attending to every other position, all at once. No sequential processing, just a big parallelizable attention matrix.

The authors were working on machine translation. The specific frustration that likely motivated it: RNNs had to compress the entire source sentence into a single fixed-size vector before decoding the translation — a massive bottleneck. Earlier "attention" mechanisms (Bahdanau, 2014) had already shown that letting the decoder peek at all encoder states was better. The Transformer took this idea to its logical conclusion: *what if everything was attention, all the way down?*

But knowing the motivation still doesn't fully explain the intuition. Why would a purely attention-based architecture — with no recurrence, no convolution — generalize so powerfully to tasks the authors weren't thinking about in 2017? Text generation, code, reasoning, image captioning, protein folding. They designed it for German-to-English translation.

The honest answer is probably: they didn't fully know. They had a strong hypothesis, ran the experiments, and it worked better than expected. Then the research community spent the next five years figuring out *why* it worked so well, and finding new things to do with it.

Science often works this way. The insight comes first, and the full explanation of why it generalizes follows later. The Transformer's self-attention mechanism turned out to have a beautiful property: it's a general-purpose *sequence-to-sequence function approximator* that can be parallelized efficiently on GPUs. Once you have that, plus enough data and compute, the architecture can learn almost any pattern in language.

### Why does a text-translation architecture work for everything else?

Underneath the translation task is a more general problem: *learn a representation of each input token that encodes its meaning in context*. That's what attention does — it lets each token's representation be informed by all other tokens. "Bank" next to "river" learns a different vector than "bank" next to "interest rate."

These contextual representations turn out to be useful for far more than translation. They're useful for any task that involves understanding language. Text generation is just decoding those representations one step at a time. The architecture didn't contain anything translation-specific; it contained something more fundamental.

The community discovered this empirically between 2018 and 2020, with BERT, GPT, and GPT-2 showing that the same architecture, slightly modified, worked for classification, generation, question answering, summarization — basically everything. At that point, the question shifted from "why does it work for translation" to "is there a task where it *doesn't* work?" And the answer, increasingly, has been: not many.

---

## Appendix: If Training Had Continued

At stop time (step 438,700), training was **73% through the cosine decay schedule** (target: 600,000 steps). The learning rate had decayed from its peak of 6e-4 to approximately **~1.0×10⁻⁴**, with the schedule still heading down to 6e-5 by step 600k.

The model was still actively learning — new val loss lows were being set in the final 10k steps before the run ended. Had training continued to completion:

- The cosine tail (steps 438k→600k) would have provided ~161k more steps of refined learning at slowly decreasing LR.
- Projected final val loss: approximately **2.78–2.83**.
- Estimated final perplexity: **~16.1–16.8**.

---

*Implementation: custom PyTorch. FemtoGPT trained on Shakespeare (char-level). TinyGPT trained on FineWeb-Edu via HuggingFace streaming. Hardware: RTX 4090 on RunPod.*
