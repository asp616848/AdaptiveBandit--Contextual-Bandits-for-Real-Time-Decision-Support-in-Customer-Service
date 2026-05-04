# Qwen Fine-Tuning Server Reference

> **Note**: This file is provided for **reference only**. The resulting fine-tuned model is already hosted on the Hugging Face Hub at `abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b` and is loaded automatically by the RL training scripts. **You do not need to run these commands to test the multi-turn RL pipeline.**

## Fine-Tuning Workflow

This describes the exact workflow used to produce the `abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b` model on a Linux GPU Server.

### 1. Download Base Model
We used `huggingface-cli` to securely download the base model (`Qwen/Qwen2.5-7B-Instruct`) to the server.

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
```

### 2. Environment Setup
The training required specific libraries:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-server.txt
```

### 3. Launch LoRA Fine-Tuning
We ran the `train_behavior_server.py` script, which loads the base Qwen model and performs LoRA (Low-Rank Adaptation) fine-tuning based on the generated dataset containing specific customer behavior tags (`<behavior>{...}</behavior>`).

```bash
mkdir -p logs
nohup python train_behavior_server.py \
  --model_name /path/to/Qwen2.5-7B-Instruct \
  > logs/train_behavior.log 2>&1 &
```

The script automatically handled:
1. Tokenization and padding setup.
2. Peft/LoRA configuration (`r=16`, `lora_alpha=32`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`).
3. Loading data and passing it to Hugging Face's `SFTTrainer`.
4. Merging the LoRA weights back into the base model upon completion to create a standalone artifact.

### 4. Merging and Uploading
After training finished, the `train_behavior_server.py` script automatically merged the adapter weights into the base model, creating the merged output. This output was then uploaded to the Hugging Face Hub as `abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b` to be easily accessible by the RL pipeline without requiring manual weight transfer.
