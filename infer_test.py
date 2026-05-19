#Load the Model - PyTorch native format - and run inference


import tinygpt
model = tinygpt.load_model_for_inference()

prompts = [
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

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    print(model.generate_text(start_text=prompt, max_tokens=500, temperature=0.7))

