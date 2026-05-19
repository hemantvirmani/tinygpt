"""
Inference using the HuggingFace-format TinyGPT model.

Usage:
    python infer_hf.py --prompt "The key differences between mitosis and meiosis are"
    python infer_hf.py --prompt "Once upon a time" --max_tokens 200 --temperature 0.8
"""
import argparse
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "tinygpt_hf"

BENCHMARK_PROMPTS = [
    # Karpathy's prompt
    "Hello, I'm a language model,",

    # Science / Factual
    "The human brain contains approximately",
    "Photosynthesis is the process by which plants",
    "The theory of relativity states that ",

    # History
    "The Roman Empire fell due to several factors including",
    "During the Industrial Revolution, workers ",

    # Educational / Instructional (FineWeb-Edu sweet spot)
    "To solve a quadratic equation, you must first",
    "The key differences between mitosis and meiosis are ",

    # Creative
    "Once upon a time in ancient India, there lived a king who ",
]

def load_model(model_dir: str = MODEL_DIR):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_dir} on {device} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    # TinyGPT's head has a trained bias; restore it from safetensors.
    # If lm_head.bias is present we replace lm_head with a bias-enabled Linear.
    sf_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(sf_path):
        from safetensors.torch import load_file
        sd = load_file(sf_path, device="cpu")
        if "lm_head.bias" in sd:
            n_embd = model.config.n_embd
            new_head = torch.nn.Linear(n_embd, model.config.vocab_size, bias=True)
            new_head.weight = torch.nn.Parameter(sd["lm_head.weight"])
            new_head.bias   = torch.nn.Parameter(sd["lm_head.bias"])
            model.lm_head = new_head
        elif "lm_head.weight" in sd:
            model.lm_head.weight = torch.nn.Parameter(sd["lm_head.weight"])

    model = model.to(device)
    model.eval()
    print("Ready.")
    return model, tokenizer, device

def generate(model, tokenizer, device, prompt: str, max_new_tokens: int = 500,
             temperature: float = 0.7, repetition_penalty: float = 1.3) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,       # full vocab, matches original tinygpt.generate_text()
            top_p=1.0,     # disabled
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",      type=str, default=None,
                        help="Single prompt. Omit to run all benchmark prompts.")
    parser.add_argument("--max_tokens",  type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--model_dir",   type=str, default=MODEL_DIR)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_dir)

    prompts = [args.prompt] if args.prompt else BENCHMARK_PROMPTS
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        text = generate(model, tokenizer, device, prompt,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty)
        print(text)

if __name__ == "__main__":
    main()
