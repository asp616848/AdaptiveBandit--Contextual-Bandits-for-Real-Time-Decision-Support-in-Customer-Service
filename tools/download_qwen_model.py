from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the fine-tuned Qwen model snapshot from Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b",
        help="Hugging Face repo id",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=str(Path("Multi-Turn RL") / "Qwen2.5-7B-Instruct-merged"),
        help="Destination directory (relative path recommended)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install it (pip install huggingface_hub) and retry. "
            f"Original error: {exc}"
        )

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(dst),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Downloaded model snapshot to: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
