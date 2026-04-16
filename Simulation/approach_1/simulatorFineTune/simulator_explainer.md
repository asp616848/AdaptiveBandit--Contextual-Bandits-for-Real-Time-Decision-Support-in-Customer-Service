# Customer Simulator: What & Why

## The Big Picture

We're building a **customer-support RL agent** that learns to pick the best action at each step of a conversation. To train it, the agent needs a partner to talk to — a **customer simulator** — so we don't need a real human in the loop for every training episode.

This notebook fine-tunes a small language model (SLM) to *be* that simulator.

---

## The Dataset — ABCD

ABCD contains ~10,000 real customer-service conversations between agents and customers, across flows like returns, billing, and shipping.

Each conversation is a list of turns tagged as `agent`, `customer`, or `action` (a system event like "Account pulled up").

---

## What the Model Learns

For every customer turn in the dataset we create one training example:

```
INPUT:
[HISTORY]
Agent: Hi! How can I help?
Customer: I need to return something.
Agent: Sure, can I get your name?

[AGENT ACTION]
Agent: What's your order ID?

OUTPUT  →  "It's 3348917502."
```

The model learns: *given what has been said so far + what the agent just did, what would a customer say next?*

---

## The Model — Gemma 4B, QLoRA

- **Gemma 4B** — Google's compact open model, served locally via Ollama (`gemma4:e4b`)
- **QLoRA** — loads the model in **4-bit** (saves ~75 % VRAM), then trains only a tiny set of lightweight adapter weights (~1 % of params). Full model weights stay frozen.
- **RTX 4050 (6 GB)** — the whole thing fits in 6 GB VRAM with `MAX_SEQ_LEN = 512` and `batch = 2`.

---

## Notebook Flow

| Section | What it does |
|---------|-------------|
| 0. Install | Pip installs all deps |
| 1. Explore | Loads ABCD, prints structure + sample turns |
| 2. Build Dataset | Converts each customer turn → (history + agent action, customer reply) pair |
| 3. Estimate VRAM | Prints safe sample ceiling for your GPU; you pick `SUBSET_SIZE` |
| 4. Tokenise | Wraps samples in prompt template, masks input tokens so loss is computed only on the customer reply |
| 5. Fine-tune | Standard HuggingFace `Trainer` + PEFT, prints loss every 10 steps |
| 6. Save | Saves LoRA adapter to `./gemma_simulator_lora/` |
| 7. Inference | Two options: load the LoRA adapter (HF) or quick-test via Ollama |
| 8. Export | Optional: merge adapter into a full model for Ollama Modelfile |

---

## How the RL Agent Will Use This

```
RL Agent picks action → Simulator generates customer reply → reward computed → Agent updates policy
```

The fine-tuned simulator replaces the human, making millions of training episodes feasible offline.
