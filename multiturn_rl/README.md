# Multi-Turn RL

Main code lives in `Simulation_4/`.

Entry point for automated evaluation is the repository-root script:

```bash
bash ../run.sh
```

Key paths:
- `Simulation_4/scripts/phase10_full_pipeline.py`: end-to-end runner (train + evaluate + demo rollouts)
- `Simulation_4/artifacts/`: calibration artifacts + generated training runs
- `../output/`: exported reports/plots/models created by `run.sh`

Notes:
- The included `Qwen2.5-7B-Instruct-merged/` snapshot is not required for the automated Docker evaluation.
- The “NLP” pipeline in `run.sh` uses a text-only observation wrapper with an offline-safe hashing embedder.
