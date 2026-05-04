import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Regex patterns - compiled once at import time
# ---------------------------------------------------------------------------
INFO_PATTERN = re.compile(
    r"(<account_id>|<order_id>|<email>|<phone>|<username>|<pin_number>"
    r"|account id|order id|email|phone|username|member level|membership"
    r"|name:|address|zip|pin|code)",
    re.IGNORECASE,
)
ASK_PATTERN = re.compile(
    r"(can i get|may i have|need your|provide|what is your"
    r"|order id|account id|email|phone|username|membership|name)",
    re.IGNORECASE,
)
FAILURE_PATTERN = re.compile(
    r"(sorry|unable|cannot|can't|not possible|do not qualify|out of stock|issue)",
    re.IGNORECASE,
)
ESCALATE_PATTERN = re.compile(
    r"(manager|supervisor|escalat|complaint|call me back|talk to someone higher)",
    re.IGNORECASE,
)
FRUSTRATION_PATTERN = re.compile(
    r"(frustrat|angry|ridiculous|still not working|not helping|this is useless|upset)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BehaviorSample:
    prompt: str
    completion: str
    agent_action: str
    info_provided: int
    frustration_signal: int
    wants_escalation: int


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("behavior-train")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dialogs(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)

        dialogs = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            dialogs.append(json.loads(line))
    return dialogs


def _unwrap_root(raw: Any, split: str = "train") -> List[Any]:
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], dict):
            d0 = raw[0]
            if split in d0 and isinstance(d0.get(split), list):
                return d0[split]
            if all(k in d0 for k in ["train", "dev", "test"]):
                return d0.get("train", [])
        return raw
    if isinstance(raw, dict):
        if split in raw and isinstance(raw.get(split), list):
            return raw[split]
        if all(k in raw for k in ["train", "dev", "test"]):
            return raw.get("train", [])
        for key in ["data", "dialogs", "dialogues", "conversations", "records", "items", "train"]:
            val = raw.get(key)
            if isinstance(val, list):
                return val
        return [raw]
    return []


def _normalize_turn(turn: Any) -> Optional[dict]:
    if isinstance(turn, dict):
        speaker = str(turn.get("speaker", "")).strip().lower()
        text = str(turn.get("text", "")).strip()
        if speaker and text:
            return {"speaker": speaker, "text": text, "targets": turn.get("targets", [])}
        return None
    if isinstance(turn, (list, tuple)) and len(turn) >= 2:
        speaker = str(turn[0]).strip().lower()
        text = str(turn[1]).strip()
        if speaker and text:
            return {"speaker": speaker, "text": text, "targets": []}
    return None


def extract_turns(convo: Any) -> List[dict]:
    if not isinstance(convo, dict):
        return []
    for key in ["delexed", "dialog", "dialogue", "turns", "conversation", "messages", "original"]:
        val = convo.get(key)
        if not isinstance(val, list):
            continue
        normalized = [_normalize_turn(t) for t in val]
        normalized = [t for t in normalized if t is not None]
        if normalized:
            return normalized
    return []


def infer_agent_action(turn: dict) -> str:
    text = str(turn.get("text", "")).lower()
    targets = turn.get("targets") or []
    act_type = targets[1] if len(targets) > 1 else None
    act_name = targets[2] if len(targets) > 2 else None

    if act_type == "take_action":
        return f"TakeAction:{act_name}" if act_name else "TakeAction"
    if ASK_PATTERN.search(text):
        return "AskInfo"
    if "anything else" in text:
        return "CheckResolved"
    if act_type == "retrieve_utterance":
        return "Respond"
    return "Other"


def has_info(text: str) -> int:
    if INFO_PATTERN.search(text):
        return 1
    if ":" in text and len(text) < 120:
        return 1
    return 0


def build_samples(
    dialogs: Any,
    split: str = "train",
    max_dialogs: Optional[int] = None,
) -> Tuple[List[BehaviorSample], dict]:
    dialogs = _unwrap_root(dialogs, split=split)
    if max_dialogs is not None:
        dialogs = dialogs[:max_dialogs]

    samples: List[BehaviorSample] = []
    convo_lengths: List[int] = []

    for convo in dialogs:
        delexed = extract_turns(convo)
        if not delexed:
            continue

        scenario = convo.get("scenario") or {}
        subflow = scenario.get("subflow", "unknown")
        convo_lengths.append(len(delexed))

        last_agent_action = "START"
        failed_attempts = 0
        customer_turns = 0
        info_collected = 0

        for idx, turn in enumerate(delexed):
            speaker = str(turn.get("speaker", "")).lower()
            text = str(turn.get("text", "")).strip()
            text_l = text.lower()

            if speaker == "agent":
                last_agent_action = infer_agent_action(turn)
                if FAILURE_PATTERN.search(text_l):
                    failed_attempts += 1
                continue

            if speaker != "customer":
                continue

            customer_turns += 1
            info_provided = has_info(text_l)
            info_collected += info_provided
            wants_escalation = 1 if ESCALATE_PATTERN.search(text_l) else 0
            frustration_signal = 1 if FRUSTRATION_PATTERN.search(text_l) else 0

            state = {
                "subflow": subflow,
                "turn_index": idx + 1,
                "failed_attempts": failed_attempts,
                "info_collected_ratio": round(info_collected / max(customer_turns, 1), 3),
            }
            behavior = {
                "info_provided": int(info_provided),
                "frustration_signal": int(frustration_signal),
                "wants_escalation": int(wants_escalation),
            }

            prompt = (
                "You are an ABCD customer simulator. Your top priority is behavioral fidelity.\n"
                "Given the state and agent action, return a realistic customer reply and behavior tag.\n"
                "Return strictly:\n"
                "<response>...</response>\n"
                "<behavior>{\"info_provided\":0,\"frustration_signal\":0,\"wants_escalation\":0}</behavior>\n"
                f"state={json.dumps(state, ensure_ascii=True)}\n"
                f"agent_action={last_agent_action}\n"
                "assistant:\n"
            )
            completion = (
                f"<response>{text}</response>\n"
                f"<behavior>{json.dumps(behavior, ensure_ascii=True, separators=(',', ':'))}</behavior>"
            )

            samples.append(
                BehaviorSample(
                    prompt=prompt,
                    completion=completion,
                    agent_action=last_agent_action,
                    info_provided=int(info_provided),
                    frustration_signal=int(frustration_signal),
                    wants_escalation=int(wants_escalation),
                )
            )

    ask_rows = [s for s in samples if s.agent_action == "AskInfo"]
    summary = {
        "num_conversations": len(convo_lengths),
        "avg_conversation_length": round(mean(convo_lengths), 3) if convo_lengths else 0.0,
        "num_samples": len(samples),
        "frustration_signal_rate": round(mean([s.frustration_signal for s in samples]), 4) if samples else 0.0,
        "wants_escalation_rate": round(mean([s.wants_escalation for s in samples]), 4) if samples else 0.0,
        "info_given_after_ask_rate": round(mean([s.info_provided for s in ask_rows]), 4) if ask_rows else 0.0,
    }
    return samples, summary


def split_samples(
    samples: List[BehaviorSample],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[BehaviorSample], List[BehaviorSample]]:
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(samples), generator=g).tolist()
    samples = [samples[i] for i in perm]

    eval_size = max(1, int(len(samples) * eval_ratio))
    eval_samples = samples[:eval_size]
    train_samples = samples[eval_size:]

    if not train_samples:
        raise ValueError(
            "Training split is empty after shuffle. "
            "Lower --eval_ratio or provide more data."
        )
    return train_samples, eval_samples


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def encode_samples(
    samples: List[BehaviorSample],
    tokenizer: Any,
    max_length: int,
) -> Tuple[List[dict], int]:
    rows: List[dict] = []
    skipped = 0
    eos_token = tokenizer.eos_token or ""

    for s in samples:
        prompt_ids = tokenizer(s.prompt, add_special_tokens=False)["input_ids"]
        full_text = s.prompt + s.completion + eos_token
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Skip samples whose prompt alone is too long - nothing to supervise
        if len(prompt_ids) >= max_length - 8:
            skipped += 1
            continue

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # Skip if there are no supervised tokens
        if not any(t != -100 for t in labels):
            skipped += 1
            continue

        rows.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )

    return rows, skipped


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------
try:
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def __init__(self, logger: logging.Logger) -> None:
            self.logger = logger

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            step = state.global_step
            loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            lr = logs.get("learning_rate")
            if loss is not None:
                self.logger.info(f"step={step} train_loss={loss:.6f} lr={lr}")
            if eval_loss is not None:
                ppl = math.exp(min(eval_loss, 20))  # cap to avoid overflow
                self.logger.info(f"step={step} eval_loss={eval_loss:.6f} ppl={ppl:.3f}")

except ImportError:
    ProgressCallback = None  # type: ignore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavior-focused LoRA fine-tuning for Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument("--data_path", type=Path, default=Path("abcd_v1.1.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/qwen2_5_behavior"))
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Enable 4-bit quantisation via BitsAndBytes (requires CUDA).",
    )
    parser.add_argument("--max_dialogs", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model/tokenizer only from local files (offline servers).",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force CPU training (slow, for debugging only).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Output dir + logger
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(args.output_dir / "train.log")
    logger.info("Starting behavior-focused training job")
    logger.info(f"args={vars(args)}")

    # ------------------------------------------------------------------
    # CUDA availability checks
    # ------------------------------------------------------------------
    cuda_available = torch.cuda.is_available() and not args.no_cuda
    if not cuda_available:
        logger.warning(
            "CUDA is NOT available (or --no_cuda was set). "
            "Training on CPU - this will be very slow for 7B models."
        )

    if args.use_4bit and not cuda_available:
        logger.warning("--use_4bit requires CUDA. Disabling 4-bit quantisation.")
        args.use_4bit = False

    # ------------------------------------------------------------------
    # Model path sanity check (offline mode)
    # ------------------------------------------------------------------
    if os.path.sep in args.model_name and not Path(args.model_name).exists():
        raise FileNotFoundError(
            f"Model path does not exist: {args.model_name}. "
            "For offline training pass a valid local directory to --model_name."
        )

    # ------------------------------------------------------------------
    # Lazy imports - only after validation to give fast-fail errors
    # ------------------------------------------------------------------
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------
    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    logger.info(f"Loading dialogs from {args.data_path}")
    dialogs = load_dialogs(args.data_path)
    samples, summary = build_samples(dialogs, split=args.split, max_dialogs=args.max_dialogs)
    logger.info(f"dataset_summary={json.dumps(summary, ensure_ascii=True)}")

    if not samples:
        raise RuntimeError(
            "No samples were extracted from the dataset. "
            "Check --data_path and --split arguments."
        )

    train_samples, eval_samples = split_samples(samples, eval_ratio=args.eval_ratio, seed=args.seed)
    logger.info(f"train_samples={len(train_samples)}  eval_samples={len(eval_samples)}")

    # ------------------------------------------------------------------
    # Tokenizer - load before encoding
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    # Must be set before any tokenisation call
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    train_rows, train_skipped = encode_samples(train_samples, tokenizer, args.max_length)
    eval_rows, eval_skipped = encode_samples(eval_samples, tokenizer, args.max_length)

    logger.info(
        f"tokenization: train_rows={len(train_rows)} (skipped={train_skipped}) "
        f"eval_rows={len(eval_rows)} (skipped={eval_skipped})"
    )

    if not train_rows:
        raise RuntimeError(
            "No train rows remain after tokenisation. "
            "Try increasing --max_length or inspect your data."
        )
    if not eval_rows:
        raise RuntimeError(
            "No eval rows remain after tokenisation. "
            "Try increasing --max_length or reducing --eval_ratio."
        )

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    # ------------------------------------------------------------------
    # Quantisation config
    # ------------------------------------------------------------------
    quant_config = None
    if args.use_4bit:
        compute_dtype = (
            torch.bfloat16
            if cuda_available and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        logger.info(f"4-bit quantisation enabled (compute_dtype={compute_dtype})")

    # ------------------------------------------------------------------
    # Model dtype for non-quantised loads
    # ------------------------------------------------------------------
    if quant_config is None:
        if cuda_available and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
        elif cuda_available:
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32  # CPU needs fp32
    else:
        model_dtype = None  # BitsAndBytes manages dtype internally

    # ------------------------------------------------------------------
    # Model load
    # ------------------------------------------------------------------
    logger.info(f"Loading model {args.model_name} ...")
    model_kwargs: dict = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    elif cuda_available:
        model_kwargs["torch_dtype"] = model_dtype
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32
        # No device_map on CPU - let PyTorch place it

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Precision flags for TrainingArguments
    # ------------------------------------------------------------------
    use_bf16 = cuda_available and torch.cuda.is_bf16_supported() and not args.use_4bit
    use_fp16 = cuda_available and not use_bf16 and not args.use_4bit

    # ------------------------------------------------------------------
    # dataloader_num_workers: 0 is safest inside containers/slurm
    # Raise it only when you know forked workers won't deadlock.
    # ------------------------------------------------------------------
    num_workers = min(4, os.cpu_count() or 1) if cuda_available else 0

    # ------------------------------------------------------------------
    # eval_strategy / evaluation_strategy compatibility shim
    # Works with transformers >= 4.31 (eval_strategy) and older versions
    # ------------------------------------------------------------------
    import inspect as _inspect

    _ta_params = set(_inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_strategy_key = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"

    train_arg_dict: dict = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": False,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        eval_strategy_key: "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": ["tensorboard"],
        "logging_dir": str(args.output_dir / "tb_logs"),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "dataloader_num_workers": num_workers,
        "seed": args.seed,
        "no_cuda": args.no_cuda,
    }

    train_args = TrainingArguments(**train_arg_dict)

    # ------------------------------------------------------------------
    # Data collator (pad_to_multiple_of=8 for tensor-core efficiency)
    # ------------------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    callbacks = [ProgressCallback(logger)] if ProgressCallback is not None else []

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # Checkpoint resume logic
    # ------------------------------------------------------------------
    resume: Optional[str] = args.resume_from_checkpoint
    if resume is None:
        try:
            last_ckpt = get_last_checkpoint(str(args.output_dir))
            if last_ckpt is not None:
                resume = last_ckpt
                logger.info(f"Auto-resuming from checkpoint: {resume}")
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(f"Could not detect last checkpoint: {exc}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting training ...")
    trainer.train(resume_from_checkpoint=resume)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Training complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
