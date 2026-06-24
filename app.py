# =============================================================================
# SECTION 1: IMPORTS
# Standard library and Modal SDK imports needed to define and run the app.
# =============================================================================

import modal

# =============================================================================
# SECTION 2: MODAL IMAGE SETUP
#
# A Modal "Image" defines the container environment your function runs in —
# think of it as a Dockerfile, but expressed in Python.
#
# Here we:
#   - Start from a slim Debian base image
#   - Install Python 3.11
#   - pip-install the libraries the model needs at *build time*, so they are
#     baked into the image and don't have to be re-installed on every run.
#
# "transformers"  — Hugging Face library that downloads and runs the model.
# "torch"         — PyTorch backend required by most HF models.
# "accelerate"    — Optional but recommended; speeds up model loading.
# "sentencepiece" — Tokeniser used by many LLaMA-family models.
# =============================================================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "accelerate>=0.25.0",
        "sentencepiece>=0.1.99",
    )
)

# =============================================================================
# SECTION 3: MODAL APP DEFINITION
#
# modal.App is the top-level container that groups all your Modal functions
# and scheduled jobs together.  The name appears in the Modal dashboard.
# =============================================================================

app = modal.App(name="tinyllama-qa", image=image)

# =============================================================================
# SECTION 4: MODEL CONSTANTS
#
# TinyLlama-1.1B-Chat is a 1.1-billion-parameter instruction-tuned model that
# fits comfortably in CPU RAM, making it ideal for quick demos with no GPU
# required.  Swap MODEL_ID for any other HF model ID to use a different model.
# =============================================================================

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# =============================================================================
# SECTION 5: THE MODAL FUNCTION
#
# @app.function() turns a regular Python function into a Modal "remote function"
# that executes inside the container image defined above.
#
# Key parameters used here:
#   cpu=2        — request 2 vCPUs (model loads faster with more cores)
#   memory=4096  — request 4 GB RAM (TinyLlama needs ~2–3 GB)
#   timeout=300  — allow up to 5 minutes for cold-start + inference
#
# Inside the function we:
#   1. Import HF libraries (imports inside the function run in the container).
#   2. Load the tokeniser and model.
#   3. Build a chat-style prompt using the model's expected template.
#   4. Tokenise, generate, decode, and return the assistant reply.
# =============================================================================

@app.function(cpu=2, memory=4096, timeout=300)
def answer_question(question: str) -> str:
    """
    Load TinyLlama and generate an answer to `question`.
    This function runs remotely inside the Modal container.
    """

    # -------------------------------------------------------------------------
    # 5a. Hugging Face imports (inside the function = inside the container)
    # -------------------------------------------------------------------------
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"[Modal] Loading model: {MODEL_ID}")

    # -------------------------------------------------------------------------
    # 5b. Load tokeniser
    #
    # The tokeniser converts raw text into token IDs the model understands, and
    # converts the model's output IDs back into human-readable text.
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # -------------------------------------------------------------------------
    # 5c. Load model in float32 (safe default for CPU inference)
    #
    # device_map="auto" lets Hugging Face decide where to place model layers
    # (CPU in our case since no GPU is requested).
    # -------------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    print("[Modal] Model loaded. Generating response …")

    # -------------------------------------------------------------------------
    # 5d. Build the chat prompt
    #
    # TinyLlama-Chat expects messages in the ChatML format.
    # apply_chat_template() handles the special tokens automatically.
    # -------------------------------------------------------------------------
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, concise assistant. "
                "Answer the user's question clearly and accurately."
            ),
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # appends the "<|assistant|>" turn opener
    )

    # -------------------------------------------------------------------------
    # 5e. Tokenise the prompt
    # -------------------------------------------------------------------------
    inputs = tokenizer(prompt, return_tensors="pt")

    # -------------------------------------------------------------------------
    # 5f. Generate tokens
    #
    # max_new_tokens  — cap output length to avoid runaway generation
    # do_sample=True  — enable sampling for more natural-sounding replies
    # temperature     — controls randomness (lower = more deterministic)
    # top_p           — nucleus sampling: only consider top-p probability mass
    # -------------------------------------------------------------------------
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # -------------------------------------------------------------------------
    # 5g. Decode only the *new* tokens (skip the prompt tokens)
    # -------------------------------------------------------------------------
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()

# =============================================================================
# SECTION 6: LOCAL ENTRYPOINT
#
# @app.local_entrypoint() marks the function that Modal calls when you run:
#
#     modal run app.py
#
# It executes *locally* (on your machine) and can call remote Modal functions
# via .remote().  This is the bridge between your terminal and the cloud.
#
# IMPORTANT: Modal parses its own CLI, so sys.argv does NOT work here.
# Instead, declare typed parameters directly in the function signature —
# Modal automatically exposes them as --flag style CLI options.
#
# Usage:
#   modal run app.py                              ← prompts interactively
#   modal run app.py --question "What is AI?"     ← passes question directly
# =============================================================================

@app.local_entrypoint()
def main(question: str = ""):
    # -------------------------------------------------------------------------
    # 6a. Read the question — either from --question flag or interactive prompt.
    #
    # Modal maps the `question` parameter to --question on the CLI.
    # If omitted, we fall back to input() for interactive use.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  TinyLlama Q&A — powered by Modal + Hugging Face")
    print("=" * 60)

    if not question:
        question = input("\nEnter your question: ").strip()

    if not question:
        print("No question provided. Exiting.")
        return

    print(f"\n[Local]  Sending question to Modal: {question!r}")

    # -------------------------------------------------------------------------
    # 6b. Call the remote function — Modal spins up the container,
    #     runs inference, and returns the result to your terminal.
    # -------------------------------------------------------------------------
    response = answer_question.remote(question)

    # -------------------------------------------------------------------------
    # 6c. Display the answer
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")