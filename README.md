# TinyLlama on Modal: Iterative Infra Experiment

This repository tracks my experiments deploying a small open-source LLM (TinyLlama) on Modal under a strict $30/month budget.

## Goal

Learn how serverless AI infrastructure behaves in practice:
- cold starts
- warm container reuse
- model loading overhead
- inference latency
- cost-aware iteration

## Versions

### v1 — `app.py`
Basic function-based TinyLlama Q&A on Modal.

**What it does**
- Runs TinyLlama remotely on Modal CPU.
- Takes a question and returns an answer.

**Main limitation**
- Loads the tokenizer and model inside the function on every request.

---

### v2 — `app_v2.py`
Class-based refactor using Modal lifecycle hooks.

**What changed**
- Switched to `@app.cls(...)`
- Model loads once at container startup using `@modal.enter()`
- Inference moved into `@modal.method()`

**Why it matters**
- Enables warm-container reuse.
- Avoids reloading the model on every request.

---

### v3 — `app_v3.py`
Instrumentation and cleanup.

**What changed**
- Added timing metrics for model load and inference.
- Reduced generation length.
- Cleaned up parts of the config.

**What I learned**
- Separate `modal run` calls do not prove warm reuse by themselves.

---

### v3.1 — `app_v3_1.py`
Multi-request warm-container test.

**What changed**
- Sends multiple questions in one `modal run`.
- Uses the same class-backed Modal instance across requests.

**What it proved**
- One cold start.
- Multiple warm requests on the same container.
- `request_count_on_container` increased from 1 → 2 → 3.

## Sample result

In `app_v3_1.py`, I observed:
- one model load at startup,
- repeated requests without reloading,
- lower per-request overhead after initialization.

## How to run

### v1
```bash
modal run app.py --question "What is a compiler?"
```

### v2
```bash
modal run app_v2.py --question "What is a compiler?"
```

### v3
```bash
modal run app_v3.py --question "What is a compiler?"
```

### v3.1
```bash
modal run app_v3_1.py --question1 "What is a compiler?" --question2 "What is a tokenizer?" --question3 "What is CoreWeave?"
```

## Notes

- TinyLlama on CPU is useful for infrastructure experiments, but answer quality is limited.
- The main value of this project is understanding container lifecycle, warm reuse, and cost/latency tradeoffs on Modal.

## Next step

Build a v4 that keeps the same infrastructure pattern but improves answer quality using grounded context instead of relying only on the base model.
