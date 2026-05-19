"""
Layer-by-layer diagnostic to pinpoint where TinyGPT → HF conversion diverges.
Usage: python verify_conversion.py
"""
import torch
import tiktoken
import os
import tinygpt
from transformers import GPT2LMHeadModel
from safetensors.torch import load_file

PROMPT  = "Hello, I'm a language model,"
HF_DIR  = "tinygpt_hf"

def load_hf(hf_dir):
    model = GPT2LMHeadModel.from_pretrained(hf_dir)
    sf_sd = load_file(os.path.join(hf_dir, "model.safetensors"), device="cpu")
    # Restore lm_head with bias if the exported model has one
    if "lm_head.bias" in sf_sd:
        n_embd = model.config.n_embd
        new_head = torch.nn.Linear(n_embd, model.config.vocab_size, bias=True)
        new_head.weight = torch.nn.Parameter(sf_sd["lm_head.weight"])
        new_head.bias   = torch.nn.Parameter(sf_sd["lm_head.bias"])
        model.lm_head = new_head
    else:
        model.lm_head.weight = torch.nn.Parameter(sf_sd["lm_head.weight"])
    model.transformer.wte.weight.data.copy_(sf_sd["transformer.wte.weight"])
    return model.cpu().eval(), sf_sd

def main():
    enc    = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(PROMPT)
    x      = torch.tensor(tokens).unsqueeze(0)          # (1, T)

    print("Loading TinyGPT ...")
    orig = tinygpt.load_model_for_inference()
    orig.cpu().eval()

    print("Loading HF model ...")
    hf, sf_sd = load_hf(HF_DIR)

    tiny_sd = torch.load("tinygpt_weights.pt", map_location="cpu", weights_only=True)

    # ── 1. Weight-value checks (directly compare saved tensors) ──────────────
    print("\n-- Weight value checks --")
    wte_ok = torch.allclose(sf_sd["transformer.wte.weight"],
                            tiny_sd["token_embedding_table.weight"])
    lmh_ok = torch.allclose(sf_sd["lm_head.weight"],
                            tiny_sd["head.weight"])
    print(f"wte    matches token_embedding_table : {wte_ok}")
    print(f"lm_head matches head.weight          : {lmh_ok}")

    # c_attn: after transposing, hf(768,2304) should equal tiny(2304,768).T
    orig_c_attn = tiny_sd["blocks.0.attention.c_attn.weight"]   # (2304, 768)
    hf_c_attn   = sf_sd["transformer.h.0.attn.c_attn.weight"]   # (768, 2304)
    cattn_ok = torch.allclose(orig_c_attn, hf_c_attn.T)
    print(f"c_attn  correctly transposed         : {cattn_ok}  "
          f"(max diff {(orig_c_attn - hf_c_attn.T).abs().max():.4e})")

    # c_proj in attention
    orig_c_proj = tiny_sd["blocks.0.attention.c_proj.weight"]   # (768, 768)
    hf_c_proj   = sf_sd["transformer.h.0.attn.c_proj.weight"]   # (768, 768)
    cproj_ok = torch.allclose(orig_c_proj, hf_c_proj.T)
    print(f"c_proj  correctly transposed         : {cproj_ok}  "
          f"(max diff {(orig_c_proj - hf_c_proj.T).abs().max():.4e})")

    # MLP c_fc
    orig_c_fc = tiny_sd["blocks.0.feed_forward.0.weight"]        # (3072, 768)
    hf_c_fc   = sf_sd["transformer.h.0.mlp.c_fc.weight"]         # (768, 3072)
    cfc_ok = torch.allclose(orig_c_fc, hf_c_fc.T)
    print(f"mlp.c_fc correctly transposed        : {cfc_ok}  "
          f"(max diff {(orig_c_fc - hf_c_fc.T).abs().max():.4e})")

    # MLP c_proj
    orig_mlp_proj = tiny_sd["blocks.0.feed_forward.2.weight"]    # (768, 3072)
    hf_mlp_proj   = sf_sd["transformer.h.0.mlp.c_proj.weight"]   # (3072, 768)
    mlpproj_ok = torch.allclose(orig_mlp_proj, hf_mlp_proj.T)
    print(f"mlp.c_proj correctly transposed      : {mlpproj_ok}  "
          f"(max diff {(orig_mlp_proj - hf_mlp_proj.T).abs().max():.4e})")

    # ── 1b. Block 11 weight checks (blocks 0-10 are identical; block 11 is special) ──
    print("\n-- Block 11 weight checks --")
    last = tinygpt.G_N_LAYERS - 1
    orig_c_attn_L = tiny_sd[f"blocks.{last}.attention.c_attn.weight"]
    hf_c_attn_L   = sf_sd[f"transformer.h.{last}.attn.c_attn.weight"]
    ok = torch.allclose(orig_c_attn_L, hf_c_attn_L.T)
    print(f"Block {last} c_attn  transposed ok: {ok}  (max diff {(orig_c_attn_L - hf_c_attn_L.T).abs().max():.4e})")

    orig_fc_L = tiny_sd[f"blocks.{last}.feed_forward.0.weight"]
    hf_fc_L   = sf_sd[f"transformer.h.{last}.mlp.c_fc.weight"]
    ok = torch.allclose(orig_fc_L, hf_fc_L.T)
    print(f"Block {last} mlp.c_fc transposed ok: {ok}  (max diff {(orig_fc_L - hf_fc_L.T).abs().max():.4e})")

    orig_mp_L = tiny_sd[f"blocks.{last}.feed_forward.2.weight"]
    hf_mp_L   = sf_sd[f"transformer.h.{last}.mlp.c_proj.weight"]
    ok = torch.allclose(orig_mp_L, hf_mp_L.T)
    print(f"Block {last} mlp.c_proj transposed ok: {ok}  (max diff {(orig_mp_L - hf_mp_L.T).abs().max():.4e})")

    # ── 2. Layer-by-layer forward pass comparison ─────────────────────────────
    print("\n-- Layer-by-layer forward pass --")
    with torch.no_grad():
        # Embeddings (manual)
        tok_orig = orig.token_embedding_table(x)
        pos_orig = orig.position_embedding_table(torch.arange(x.shape[1]))
        emb_orig = tok_orig + pos_orig

        # HF full forward with all hidden states; use_cache=False avoids stale state
        hf_out = hf(x, output_hidden_states=True, use_cache=False)
        # hidden_states[0] = after embed+drop (before block 0)
        # hidden_states[i+1] = after block i
        hf_hs = hf_out.hidden_states

        print(f"len(hf_hs) = {len(hf_hs)}  (expected {tinygpt.G_N_LAYERS + 1})")
        diff = (emb_orig - hf_hs[0]).abs()
        print(f"After embedding  - max diff: {diff.max():.6f}  mean: {diff.mean():.6f}")

        # Block by block (TinyGPT manual, HF from hidden_states)
        h_orig = emb_orig
        for i in range(tinygpt.G_N_LAYERS):
            h_orig = orig.blocks[i](h_orig)
            diff   = (h_orig - hf_hs[i + 1]).abs()
            print(f"After block {i:2d}   - max diff: {diff.max():.6f}  mean: {diff.mean():.6f}")
            if diff.max() > 1e-3:
                # Also check: is the divergence explained by ln_f being included in hf_hs?
                h_with_lnf = orig.ln_f(h_orig)
                diff_lnf = (h_with_lnf - hf_hs[i + 1]).abs()
                print(f"  ^^^ divergence starts here  (with TinyGPT ln_f applied: max {diff_lnf.max():.6f})")

        # Final logits — full forward passes, ln_f included
        orig_logits = orig(x)[:, -1, :]
        hf_logits   = hf_out.logits[:, -1, :]

    corr = torch.corrcoef(torch.stack([orig_logits[0], hf_logits[0]]))[0, 1]
    print(f"\nFinal logit correlation: {corr:.6f}")

    print("\nTinyGPT top-5:")
    for v, i in zip(*orig_logits[0].topk(5)):
        print(f"  {enc.decode([i.item()])!r:20s} {v.item():.3f}")
    print("HF model top-5:")
    for v, i in zip(*hf_logits[0].topk(5)):
        print(f"  {enc.decode([i.item()])!r:20s} {v.item():.3f}")

if __name__ == "__main__":
    main()
