"""
Convert TinyGPT weights to HuggingFace GPT-2 format.

Usage:
    pip install transformers
    python export_to_hf.py
"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

WEIGHTS_FILE = "tinygpt_weights.pt"
OUTPUT_DIR   = "tinygpt_hf"
N_LAYERS     = 12
N_EMBD       = 768

# Keys whose weights need .T when HF uses Conv1D (shape in×out vs nn.Linear out×in).
# Whether transposition is actually needed is auto-detected below.
PROJ_KEYS = {
    "attention.c_attn.weight",
    "attention.c_proj.weight",
    "feed_forward.0.weight",
    "feed_forward.2.weight",
}

def build_name_map(n_layers: int) -> dict:
    m = {
        "token_embedding_table.weight": "transformer.wte.weight",
        "position_embedding_table.weight": "transformer.wpe.weight",
        "ln_f.weight":  "transformer.ln_f.weight",
        "ln_f.bias":    "transformer.ln_f.bias",
        "head.weight":  "lm_head.weight",
    }
    for i in range(n_layers):
        p = f"blocks.{i}"
        h = f"transformer.h.{i}"
        m[f"{p}.ln1.weight"] = f"{h}.ln_1.weight"
        m[f"{p}.ln1.bias"]   = f"{h}.ln_1.bias"
        m[f"{p}.ln2.weight"] = f"{h}.ln_2.weight"
        m[f"{p}.ln2.bias"]   = f"{h}.ln_2.bias"
        m[f"{p}.attention.c_attn.weight"] = f"{h}.attn.c_attn.weight"
        m[f"{p}.attention.c_attn.bias"]   = f"{h}.attn.c_attn.bias"
        m[f"{p}.attention.c_proj.weight"] = f"{h}.attn.c_proj.weight"
        m[f"{p}.attention.c_proj.bias"]   = f"{h}.attn.c_proj.bias"
        m[f"{p}.feed_forward.0.weight"]   = f"{h}.mlp.c_fc.weight"
        m[f"{p}.feed_forward.2.weight"]   = f"{h}.mlp.c_proj.weight"
    return m

def convert():
    print(f"Loading weights from {WEIGHTS_FILE} ...")
    tiny_sd = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=True)

    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=N_EMBD,
        n_layer=N_LAYERS,
        n_head=12,
        activation_function="gelu",  # TinyGPT uses nn.GELU() exact, not the tanh approx
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,   # TinyGPT trained wte and lm_head as separate matrices
    )
    hf_model = GPT2LMHeadModel(config)

    # Auto-detect whether HF uses Conv1D (weight shape: in×out) or nn.Linear (out×in).
    # Conv1D(3*n_embd, n_embd) → weight (n_embd, 3*n_embd) = (768, 2304)
    # nn.Linear(n_embd, 3*n_embd) → weight (3*n_embd, n_embd) = (2304, 768)
    hf_c_attn_shape = tuple(hf_model.transformer.h[0].attn.c_attn.weight.shape)
    needs_transpose = hf_c_attn_shape == (N_EMBD, 3 * N_EMBD)
    print(f"HF c_attn.weight shape: {hf_c_attn_shape}")
    print(f"Transposing projection weights: {needs_transpose}  "
          f"({'Conv1D detected' if needs_transpose else 'nn.Linear detected'})")

    name_map = build_name_map(N_LAYERS)
    new_sd  = {}
    skipped = []

    for tiny_key, tensor in tiny_sd.items():
        hf_key = name_map.get(tiny_key)
        if hf_key is None:
            skipped.append(tiny_key)
            continue
        if needs_transpose and any(tiny_key.endswith(k) for k in PROJ_KEYS):
            tensor = tensor.T
        new_sd[hf_key] = tensor

    # MLP Conv1D biases are zero-initialised; TinyGPT trained without them.
    for i in range(N_LAYERS):
        new_sd[f"transformer.h.{i}.mlp.c_fc.bias"]  = torch.zeros(4 * N_EMBD)
        new_sd[f"transformer.h.{i}.mlp.c_proj.bias"] = torch.zeros(N_EMBD)

    missing, unexpected = hf_model.load_state_dict(new_sd, strict=False)

    # TinyGPT's head is nn.Linear(bias=True); HF's default lm_head is bias=False.
    # Replace lm_head with a biased Linear so the trained bias is preserved.
    hf_model.lm_head = torch.nn.Linear(N_EMBD, config.vocab_size, bias=True)
    hf_model.lm_head.weight = torch.nn.Parameter(tiny_sd["head.weight"].clone())
    hf_model.lm_head.bias   = torch.nn.Parameter(tiny_sd["head.bias"].clone())
    hf_model.transformer.wte.weight.data.copy_(tiny_sd["token_embedding_table.weight"])
    hf_model.config.tie_word_embeddings = False

    # Sanity-check: wte and lm_head must hold different values.
    wte_eq_lmhead = torch.allclose(hf_model.transformer.wte.weight,
                                   hf_model.lm_head.weight)
    print(f"wte == lm_head (should be False): {wte_eq_lmhead}")
    if wte_eq_lmhead:
        raise RuntimeError("wte and lm_head are still tied — lm_head will be wrong.")

    hf_model.save_pretrained(OUTPUT_DIR)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(OUTPUT_DIR)

    size_mb = sum(
        p.numel() * p.element_size()
        for p in hf_model.parameters()
    ) / 1024 / 1024
    print(f"\nSaved to {OUTPUT_DIR}/  (~{size_mb:.0f} MB in memory)")
    if skipped:
        print(f"Skipped (buffers / not mapped): {skipped}")
    real_missing = [k for k in missing if "attn.bias" not in k and "masked_bias" not in k]
    if real_missing:
        print(f"WARNING — missing parameter keys (kept random init): {real_missing}")
    else:
        print("All parameter keys loaded successfully.")

if __name__ == "__main__":
    convert()
