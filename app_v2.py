# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

import modal

# =============================================================================
# SECTION 2: MODAL IMAGE SETUP
#
# Same as before — defines the container environment with all dependencies
# baked in at build time.
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
# =============================================================================

app = modal.App(name="tinyllama-qa", image=image)

# =============================================================================
# SECTION 4: MODEL CONSTANT
# =============================================================================

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# =============================================================================
# SECTION 5: THE MODEL CLASS  ← THE KEY CHANGE
#
# Previously the model was loaded INSIDE the answer_question() function.
# That meant every single request paid the full cost of downloading and
# loading the model weights — typically 30–90 seconds for TinyLlama.
#
# The fix: use a Modal *Class* instead of a plain function.
#
# HOW MODAL CONTAINERS WORK (beginner explanation)
# ─────────────────────────────────────────────────
# Think of a Modal container like a rented computer in the cloud.
#
#   1. You send a request  →  Modal boots a fresh container (a "cold start")
#   2. The container runs your code and returns the result
#   3. Instead of immediately shutting down, Modal keeps the container ALIVE
#      for a short idle period (controlled by `container_idle_timeout`)
#   4. If another request arrives while the container is still alive,
#      it reuses the same container — this is called a "warm" hit
#   5. Eventually, if no requests arrive, the container shuts down
#
# WHAT IS A COLD START?
# ─────────────────────
# A cold start happens when no warm container is available and Modal must
# boot a brand-new one from scratch.  Steps that happen during a cold start:
#
#   a. Pull the container image (cached after the first run)
#   b. Start the Python interpreter
#   c. Run __init__ — which for us means downloading + loading the model
#
# With the OLD code (function-based), EVERY request triggered step (c) because
# the model was loaded inside the function body, not in any setup step.
# Even on a warm container, each call would reload the model from disk.
#
# WHAT IS A WARM CONTAINER?
# ──────────────────────────
# A warm container is one that is already running and has already executed
# __init__.  When a request arrives and a warm container is available:
#
#   ✓  No image pull
#   ✓  No Python startup
#   ✓  No model download or loading  ← this is the big win
#   ✓  The tokeniser and model are already in RAM, ready to go
#
# With the NEW code (class-based), __init__ loads the model ONCE when the
# container first starts.  Every subsequent request on that container jumps
# straight to inference — which is just a few seconds instead of 30–90.
#
# WHY THIS IMPROVES LATENCY
# ──────────────────────────
# Cold start  (unavoidable on first request):  ~60–90 s  (same as before)
# Warm request (every request after that):     ~2–5 s    (vs 60–90 s before)
#
# For a real app fielding many questions, almost every request hits a warm
# container, so the average response time drops dramatically.
#
# WHY THIS IMPROVES COST EFFICIENCY
# ───────────────────────────────────
# Modal bills for compute time.  With the old code, 60 seconds of model
# loading was billed on EVERY request.  With the new code, that 60-second
# cost is paid once per container lifetime, then amortised across all the
# requests that container handles.
#
# Example — 100 questions, 60 s load time, 5 s inference time per question:
#
#   OLD  (load every request):  100 × (60 + 5) =  6,500 seconds billed
#   NEW  (load once per container, 20 q per container): 5 × 60 + 100 × 5 =
#        300 + 500 = 800 seconds billed   →  ~88 % cost reduction
#
# HOW @modal.cls ENABLES THIS
# ────────────────────────────
# @modal.cls turns a Python class into a Modal "class function".
# Modal calls __init__ exactly once per container lifecycle — at cold-start
# time.  All @modal.method functions on the class share the instance, so
# self.model and self.tokenizer are already in memory for every call.
#
# Key decorator parameters used:
#   cpu=2                    — 2 vCPUs for faster model loading
#   memory=4096              — 4 GB RAM (TinyLlama needs ~2–3 GB)
#   timeout=300              — max 5 min for cold-start + inference
#   container_idle_timeout=120  — keep the container warm for 2 minutes
#                                 after the last request, then shut down
# =============================================================================

@app.cls(
    cpu=2,
    memory=4096,
    timeout=300,
    scaledown_window=120,
)
class TinyLlamaModel:

    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"[Cold start] Loading model: {MODEL_ID}")
        print("[Cold start] This happens ONCE per container — not per request.")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto",
        )

        print("[Cold start] Model ready. Container is now warm.")

    @modal.method()
    def answer(self, question: str) -> str:
        import torch

        print(f"[Warm request] Answering: {question!r}")

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

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

# =============================================================================
# SECTION 6: LOCAL ENTRYPOINT
#
# Almost identical to before.  The only change: we now instantiate the class
# and call .answer.remote() instead of the old answer_question.remote().
#
# Modal handles routing the call to a warm container when one is available,
# or spinning up a new container (cold start) when none exists.
#
# Usage:
#   modal run app.py                              ← interactive prompt
#   modal run app.py --question "What is AI?"     ← pass question as flag
# =============================================================================

@app.local_entrypoint()
def main(question: str = ""):
    print("=" * 60)
    print("  TinyLlama Q&A — powered by Modal + Hugging Face")
    print("=" * 60)

    if not question:
        question = input("\nEnter your question: ").strip()

    if not question:
        print("No question provided. Exiting.")
        return

    print(f"\n[Local] Sending to Modal: {question!r}")

    # Instantiate the class — Modal routes this to the running container
    model = TinyLlamaModel()

    # .answer.remote() sends the request to the container.
    # If a warm container exists → near-instant response.
    # If no warm container   → cold start, then response.
    response = model.answer.remote(question)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response)
    print("=" * 60 + "\n")