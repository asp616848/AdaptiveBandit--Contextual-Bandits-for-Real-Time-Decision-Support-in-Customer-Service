# Offline Server Run (No Internet on Server)

Use this when server DNS cannot reach huggingface.co.

## Required project files in server folder

Keep these in /data/interns/studentiotlab/finetune/FINE_TUNE

- abcd_v1.1.json
- requirements-server.txt
- train_behavior_server.py
- RUN_SERVER.md

Add this local model folder (copied from internet machine):

- /data/interns/studentiotlab/finetune/FINE_TUNE/Qwen2.5-7B-Instruct

## 1) Download model on a machine with internet

Run on your local machine (not server):

```bash
python -m pip install --upgrade huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
```

Optional (higher limits):

```bash
export HF_TOKEN=YOUR_HF_TOKEN
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
```

## 2) Transfer model folder to server

From internet machine:

```bash
scp -r Qwen2.5-7B-Instruct studentiotlab@10.1.40.46:/data/interns/studentiotlab/finetune/FINE_TUNE/
```

If transfer is unstable, use tar:

```bash
tar -czf Qwen2.5-7B-Instruct.tgz Qwen2.5-7B-Instruct
scp Qwen2.5-7B-Instruct.tgz studentiotlab@10.1.40.46:/data/interns/studentiotlab/finetune/FINE_TUNE/
```

Then on server:

```bash
cd /data/interns/studentiotlab/finetune/FINE_TUNE
tar -xzf Qwen2.5-7B-Instruct.tgz
```

## 3) Verify model files exist on server

```bash
ls -lah /data/interns/studentiotlab/finetune/FINE_TUNE/Qwen2.5-7B-Instruct
```

You should see files like:

- config.json
- generation_config.json
- tokenizer.json or tokenizer.model
- tokenizer_config.json
- special_tokens_map.json
- model.safetensors files (and index json)

## 4) Activate venv and install deps on server

```bash
cd /data/interns/studentiotlab/finetune/FINE_TUNE
source /data/interns/studentiotlab/venv/bin/activate
PYTHON_BIN=/data/interns/studentiotlab/venv/bin/python
$PYTHON_BIN --version
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install torch --index-url https://download.pytorch.org/whl/cu121
$PYTHON_BIN -m pip install -r requirements-server.txt
```

## 5) Force offline mode on server

```bash
export HF_HOME=/data/interns/studentiotlab/finetune/FINE_TUNE/.hf_home
export HUGGINGFACE_HUB_CACHE=/data/interns/studentiotlab/finetune/FINE_TUNE/.hf_home/hub
export TRANSFORMERS_CACHE=/data/interns/studentiotlab/finetune/FINE_TUNE/.hf_home/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
mkdir -p /data/interns/studentiotlab/finetune/FINE_TUNE/.hf_home
```

## 6) Start offline training

```bash
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
  /data/interns/studentiotlab/venv/bin/python train_behavior_server.py \
  --data_path /data/interns/studentiotlab/finetune/FINE_TUNE/abcd_v1.1.json \
  --split train \
  --model_name /data/interns/studentiotlab/finetune/FINE_TUNE/Qwen2.5-7B-Instruct \
  --local_files_only \
  --output_dir /data/interns/studentiotlab/finetune/FINE_TUNE/model_out \
  --use_4bit \
  --max_length 1024 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --logging_steps 10 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 5 \
  > /data/interns/studentiotlab/finetune/FINE_TUNE/train.log 2>&1 &
```

## 7) Monitor and resume

```bash
tail -f /data/interns/studentiotlab/finetune/FINE_TUNE/train.log
ls -lah /data/interns/studentiotlab/finetune/FINE_TUNE/model_out
ls -lah /data/interns/studentiotlab/finetune/FINE_TUNE/model_out | grep checkpoint
```

To resume after stop/crash, run the same command from step 6 again.

## 8) Final model location

Final adapter is saved at:

- /data/interns/studentiotlab/finetune/FINE_TUNE/model_out/final

Main file:

- adapter_model.safetensors

Quick check:

```bash
ls -lah /data/interns/studentiotlab/finetune/FINE_TUNE/model_out/final
```
