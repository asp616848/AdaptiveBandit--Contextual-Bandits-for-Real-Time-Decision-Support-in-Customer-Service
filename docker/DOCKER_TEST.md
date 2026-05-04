# Docker Testing Guide

This guide demonstrates how to verify the entire pipeline runs on a fresh Ubuntu 22.04 Docker image.

## Quick Start (No GPU Docker Test)

This will test on CPU only (faster for validation):

```bash
# From the repo root directory
docker build -f Dockerfile.test -t adaptive-bandit-test:latest .
```

The build process will:
1. Start with a fresh Ubuntu 22.04 image
2. Install all system dependencies
3. Create a Python venv
4. Install all Python dependencies
5. Download the HuggingFace fine-tuned Qwen model
6. Run numerical training (2 timesteps)
7. Run NLG training with HuggingFace backend (2 timesteps)
8. Run contextual bandit training (2 timesteps)
9. Output all generated artifacts

## Expected Output Structure

After successful execution, you will see:

```
output/
├── numerical-multi-turn/
│   ├── training_summary.json
│   ├── training_log.json
│   ├── plots/
│   └── models/
├── nlp-multi-turn/
│   ├── training_summary.json
│   ├── training_log.json
│   ├── demo_rollouts.json
│   ├── plots/
│   └── models/
└── contextual-bandit/
    ├── training_summary.json
    ├── summary.csv
    ├── aggregate.csv
    ├── results.json
    ├── logs/
    ├── models/
    └── plots/

best_model/
├── numerical/
│   ├── best_model.zip
│   └── final_model.zip
└── nlp/
    ├── best_model.zip
    └── final_model.zip
```

## Testing with GPU (Optional)

If you have NVIDIA GPU and nvidia-docker:

```bash
# Replace docker with nvidia-docker
nvidia-docker build -f Dockerfile.test -t adaptive-bandit-test:latest .
```

## Manual Step-by-Step Test

If you prefer to test manually:

```bash
# Clone fresh repo
git clone <repo-url> adaptive-bandit-test
cd adaptive-bandit-test

# Run with minimal timesteps
NUM_TIMESTEPS_NUMERICAL=2 NUM_TIMESTEPS_NLG=2 bash run.sh
```

## Customization

To test with different parameters, modify the docker build command:

```bash
# Example: 50 timesteps each
docker build \
  --build-arg NUM_TIMESTEPS_NUMERICAL=50 \
  --build-arg NUM_TIMESTEPS_NLG=50 \
  -f Dockerfile.test \
  -t adaptive-bandit-test:latest \
  .
```

## What Gets Tested

✓ Fresh Ubuntu 22.04 system setup  
✓ Automatic system dependency installation  
✓ Python venv creation  
✓ All Python package installations  
✓ HuggingFace model download (12.3GB Qwen model)  
✓ Numerical RL training with masked observation  
✓ NLG RL training with fine-tuned Qwen backend  
✓ Contextual bandit algorithms (LinUCB, Thompson Sampling, Epsilon-Greedy)  
✓ Output artifact generation (CSVs, JSONs, models)  

## Troubleshooting

**If NLG training hangs:**
- This is normal on systems without GPU (model offloading to disk)
- On CPU, 2 timesteps typically takes 5-15 minutes
- Use `docker exec` to check progress

**If you hit memory limits:**
- Reduce timesteps further: `NUM_TIMESTEPS_NLG=1`
- Or use CPU-only mode (already set in Dockerfile.test)

## Validation Checklist

After the build completes, verify:

- [ ] `output/numerical-multi-turn/training_summary.json` exists
- [ ] `output/nlp-multi-turn/training_summary.json` exists  
- [ ] `output/contextual-bandit/summary.csv` exists
- [ ] `best_model/numerical/best_model.zip` exists
- [ ] `best_model/nlp/best_model.zip` exists
- [ ] All JSON files are valid (can be parsed)
- [ ] CSV files have > 0 rows
