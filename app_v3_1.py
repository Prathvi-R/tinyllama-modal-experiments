# =============================================================================
# app_v3_1.py
#
# Goal:
# - Keep v3's warm-container class design
# - Send multiple requests in a single modal run
# - Show per-request metrics so warm reuse is visible
# =============================================================================

import time
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "accelerate>=0.25.0",
        "sentencepiece>=0.1.99",
    )
)

app = modal.App(name="tinyllama-qa-v3-1", image=image)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


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

        start = time.perf_counter()
        print(f"[Cold start] Loading model: {MODEL_ID}")
        print("[Cold start] This should run once per container.")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            device_map="auto",
        )

        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None

        self.load_time_s = time.perf_counter() - start
        self.request_count = 0

        print(f"[Cold start] Model ready in {self.load_time_s:.2f}s")

    @modal.method()
    def answer(self, question: str) -> dict:
        import time
        import torch

        self.request_count += 1
        infer_start = time.perf_counter()

        print(f"[Request #{self.request_count}] Question: {question!r}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, concise assistant. "
                    "Answer clearly and accurately in 3-5 sentences."
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
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        infer_time_s = time.perf_counter() - infer_start

        return {
            "question": question,
            "response": response,
            "metrics": {
                "request_count_on_container": self.request_count,
                "model_load_time_s": round(self.load_time_s, 2),
                "inference_time_s": round(infer_time_s, 2),
                "max_new_tokens": 96,
                "model_id": MODEL_ID,
            },
        }


@app.local_entrypoint()
def main(
    question1: str = "What is the difference between a compiler and an interpreter?",
    question2: str = "What is a cold start in serverless infrastructure?",
    question3: str = "What is CoreWeave?",
):
    print("=" * 60)
    print("  TinyLlama Q&A v3.1 — Multi-request warm-container test")
    print("=" * 60)

    questions = [question1.strip(), question2.strip(), question3.strip()]
    questions = [q for q in questions if q]

    if not questions:
        print("No questions provided. Exiting.")
        return

    model = TinyLlamaModel()

    for i, question in enumerate(questions, start=1):
        print(f"\n[Local] Sending question {i}/{len(questions)}: {question!r}")
        result = model.answer.remote(question)

        print("\n" + "-" * 60)
        print(f"QUESTION {i}:")
        print(question)
        print("-" * 60)
        print("ANSWER:")
        print(result["response"])
        print("-" * 60)
        print("METRICS:")
        for k, v in result["metrics"].items():
            print(f"- {k}: {v}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("Run complete.")
    print("=" * 60)