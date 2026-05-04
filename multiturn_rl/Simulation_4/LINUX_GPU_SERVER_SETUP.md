# Linux GPU Server Setup and Runbook (Simulation_4)

## 1. What to move to the Linux machine

Use this as the minimum safe transfer set.

### Required code folders/files
- `Simulation_4/env/`
- `Simulation_4/rag/`
- `Simulation_4/training/`
- `Simulation_4/validation/`
- `Simulation_4/scripts/phase10_full_pipeline.py`

### Required artifacts for simulator dynamics and reward model
- `Simulation_4/artifacts/phase 3/`
- `Simulation_4/artifacts/phase 4/`
- `Simulation_4/artifacts/phase 5/`
- `Simulation_4/artifacts/phase 6/`

### Recommended artifacts to include
- `Simulation_4/artifacts/phase 1/` (subflow stats; fallback exists but this is preferred)
- `Simulation_4/artifacts/phase 2/`
- `Simulation_4/artifacts/phase9/`

### If you want training continuation from existing checkpoints
Also copy one of these (or both):
- `Simulation_4/artifacts/phase10_final/models/`
- `Simulation_4/artifacts/phase10_v3/models/`

### Easiest option (recommended)
Copy the full `Simulation_4/` directory.

---

## 2. Target structure on Linux server

Create this structure under a project root, for example `/srv/rl/AdaptiveBandit`:

```text
/srv/rl/AdaptiveBandit/
  Simulation_4/
    env/
    rag/
      documents/
      index/
      scenario_templates.json
      subflow_mapping.json
    training/
    validation/
    scripts/
      phase10_full_pipeline.py
    artifacts/
      phase 1/
      phase 2/
      phase 3/
      phase 4/
      phase 5/
      phase 6/
      phase9/
      phase10_final/   (optional, for continuation)
      phase10_v3/      (optional, for continuation)
```

---

## 3. Transfer commands

From your local machine (run where SSH access is available):

```bash
# Copy full folder (recommended)
scp -r Simulation_4 user@YOUR_SERVER:/srv/rl/AdaptiveBandit/

# Or use rsync for resumable transfer
rsync -avh --progress Simulation_4/ user@YOUR_SERVER:/srv/rl/AdaptiveBandit/Simulation_4/
```

---

## 4. Linux environment setup

Run on server:

```bash
cd /srv/rl/AdaptiveBandit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch with CUDA wheels (adjust if your server uses a different CUDA runtime):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:

```bash
pip install \
  stable-baselines3 \
  gymnasium \
  numpy \
  pandas \
  scikit-learn \
  sentence-transformers \
  openai \
  requests
```

Optional retrieval acceleration:

```bash
# Optional only. If this fails, code still works (retriever has non-faiss fallback).
pip install faiss-cpu
```

Quick GPU check:

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
```

---

## 5. Commands after installing Ollama

Install/start Ollama on Linux, then run:

```bash
# Start Ollama server (keep running in a tmux/screen session or service)
ollama serve
```

In another shell:

```bash
# Pull model requested by this repo's NLGLayer default
ollama pull llama3

# Verify model exists
ollama list

# Basic health check
curl -s http://localhost:11434/api/tags | head
```

If memory is tight, you can use a smaller model (for example `qwen3:4b`) and update model selection in `Simulation_4/env/nlg_layer.py`.

---

## 6. Run commands

From project root (`/srv/rl/AdaptiveBandit`), with virtual env active.

### A) Quick simulator smoke test (no NLG)

```bash
python - <<'PY'
import numpy as np
from Simulation_4.env.support_env import SupportEnv

rng = np.random.default_rng(123)
env = SupportEnv(artifacts_root='Simulation_4/artifacts', nlg_enabled=False)
print('Simulator:', env.__class__.__name__, 'actions=', env.action_space.n)

for ep in range(3):
    obs, info = env.reset(seed=ep + 1)
    done = False
    total = 0.0
    steps = 0
    last_info = {}
    while not done and steps < 25:
        action = int(rng.integers(0, env.action_space.n))
        obs, reward, done, truncated, last_info = env.step(action)
        total += float(reward)
        steps += 1
        if truncated:
            break
    terminal = (last_info.get('last_transition_outcome', {}) or {}).get('terminal_type', 'timeout')
    print(f'episode={ep+1} steps={steps} reward={total:.3f} terminal={terminal}')

env.close()
PY
```

### B) Full phase10 pipeline run

```bash
python Simulation_4/scripts/phase10_full_pipeline.py \
  --artifacts-root Simulation_4/artifacts \
  --output-subdir phase10_prod \
  --timesteps 120000 \
  --continue-from auto \
  --eval-episodes 1000 \
  --validation-level1-episodes 1000 \
  --validation-level2-episodes 800 \
  --validation-level3-episodes 1000 \
  --demo-episodes 10
```

---

## 7. Output files to expect

After a successful run, check:
- `Simulation_4/artifacts/phase10_prod/full_pipeline_report.json`
- `Simulation_4/artifacts/phase10_prod/validation_report_pretrain.json`
- `Simulation_4/artifacts/phase10_prod/training_log.json`
- `Simulation_4/artifacts/phase10_prod/training_summary.json`
- `Simulation_4/artifacts/phase10_prod/demo_rollouts.json`
- `Simulation_4/artifacts/phase10_prod/models/best_model.zip`

---

## 8. Notes specific to this codebase

- Core simulator class is `Simulation_4/env/support_env.py` (`SupportEnv`).
- RAG files are loaded from `Simulation_4/rag/`.
- This code can run without `abcd_v1.1.json`; it has fallback behavior.
- If Ollama is unavailable, run with `nlg_enabled=False` (already used in phase10 pipeline).
