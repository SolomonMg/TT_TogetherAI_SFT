
#!/usr/bin/env python3
"""
run_sft.py — Upload (optional) and launch a Together LoRA fine-tune, with file ID reuse.

Examples:

# Reuse existing uploaded files (NO upload):
python run_sft.py \
  --train-id file-fdd870d6-7244-4c4e-96d4-aa9dcc5d9a44 \
  --model openai/gpt-oss-120b \
  --n-epochs 2 --batch-size 16 --learning-rate 1e-4 --n-evals 10 \
  --suffix tiktok-sft-gptoss120b 

# Upload local files (fresh upload):
python run_sft.py \
  --train data/train_BAL.jsonl \
  --val   data/val_BAL.jsonl \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference \
  --n-epochs 2 --batch-size 8 --learning-rate 1e-4 --n-evals 10 \
  --suffix tiktok-sft-v1 --watch

  """
import os
import time
import argparse
from typing import Optional, Dict, Any

try:
    from together import Together
except ImportError as e:
    raise SystemExit("Please `pip install together` first.") from e


def upload_file(client: Together, path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"Uploading: {path}")
    up = client.files.upload(file=path)
    file_id = up.id
    print(f"  → File ID: {file_id}")
    return file_id


def create_ft_job(
    client: Together,
    training_file: str,
    validation_file: Optional[str],
    model: str,
    lora: bool,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    n_evals: int,
    suffix: str,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    kwargs = dict(
        training_file=training_file,
        model=model,
        lora=lora,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_evals=n_evals,
        suffix=suffix,
    )
    if validation_file:
        kwargs["validation_file"] = validation_file
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    print("\nCreating fine-tune job with params:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    job = client.fine_tuning.create(**kwargs)  # Pydantic model
    ft_id = job.id
    result_model = getattr(job, "result_model", None)
    print(f"\n✅ Launched FT job: {ft_id}")
    if result_model:
        print(f"Result model (when finished): {result_model}")
    return ft_id


def print_status(client: Together, ft_id: str):
    job = client.fine_tuning.retrieve(ft_id)
    status = getattr(job, "status", None)
    result_model = getattr(job, "result_model", None)
    print(f"\nStatus: {status}")
    if result_model:
        print(f"Result model: {result_model}")
    return status, result_model


def tail_events(client: Together, ft_id: str, poll_sec: int = 10):
    """Poll and print new events as they appear. Stops when job reaches a terminal state."""
    print("\nTailing events (Ctrl+C to stop)…")
    seen = set()
    terminal = {"succeeded", "failed", "cancelled", "canceled", "completed"}
    while True:
        try:
            ev = client.fine_tuning.list_events(ft_id)
            events = getattr(ev, "data", None) or []
            for e in events:
                eid = getattr(e, "id", None)
                if eid and eid in seen:
                    continue
                if eid:
                    seen.add(eid)
                ts = getattr(e, "created_at", None)
                et = getattr(e, "type", None)
                msg = getattr(e, "message", None)
                print(f"[{ts}] {et}: {msg}")

            status, _ = print_status(client, ft_id)
            if status and str(status).lower() in terminal:
                print("\n⏹ Job reached terminal state.")
                break
            time.sleep(poll_sec)
        except KeyboardInterrupt:
            print("\n⏹ Stopped tailing (job continues on Together).")
            break


def coerce_extra(kvs) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise SystemExit(f"--extra expects key=value, got: {kv}")
        k, v = kv.split("=", 1)
        if v.lower() in {"true", "false"}:
            v = (v.lower() == "true")
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        extra[k] = v
    return extra


def main():
    ap = argparse.ArgumentParser(description="Launch a Together LoRA SFT job, with optional reuse of file IDs.")
    # Either provide IDs...
    ap.add_argument("--train-id", help="Existing Together file ID for training JSONL (file-xxx)")
    ap.add_argument("--val-id", help="Existing Together file ID for validation JSONL (file-xxx)")
    # ...or local paths to upload
    ap.add_argument("--train", help="Path to training JSONL (uploads if --train-id not provided)")
    ap.add_argument("--val", help="Path to validation JSONL (uploads if --val-id not provided)")
    # Model & hyperparams
    ap.add_argument("--model", required=True, help="Base model ID (e.g., gpt-oss-120b)")
    ap.add_argument("--n-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--n-evals", type=int, default=10, help="Eval loops (ignored if no val file provided)")
    ap.add_argument("--suffix", default="tiktok-sft-v1")
    ap.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tune)")
    ap.add_argument("--watch", action="store_true", help="Poll and print events until the job completes")
    ap.add_argument("--poll-sec", type=int, default=10)
    ap.add_argument("--extra", nargs="*", default=[], help="Extra FT params as key=value (e.g. lora_rank=16)")
    args = ap.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise SystemExit("Please export TOGETHER_API_KEY before running.")

    client = Together(api_key=api_key)

    # Resolve training file id
    train_id = args.train_id
    val_id = args.val_id

    if train_id and args.train:
        print("Note: --train-id provided; skipping upload of --train path and reusing existing file ID.")
    if val_id and args.val:
        print("Note: --val-id provided; skipping upload of --val path and reusing existing file ID.")

    # Upload if IDs not given
    if not train_id:
        if not args.train:
            raise SystemExit("Provide either --train-id (preferred) or --train (path to upload).")
        train_id = upload_file(client, args.train)
    if args.val or val_id:
        if not val_id and args.val:
            val_id = upload_file(client, args.val)
    # else: no validation file

    print(f"\nTRAIN_ID={train_id}")
    if val_id:
        print(f"VAL_ID={val_id}")
    else:
        print("VAL_ID=None (no validation file attached)")

    extra_kwargs = coerce_extra(args.extra)

    # Create job
    ft_id = create_ft_job(
        client=client,
        training_file=train_id,
        validation_file=val_id,
        model=args.model,
        lora=not args.no_lora,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_evals=args.n_evals if val_id else 0,
        suffix=args.suffix,
        extra_kwargs=extra_kwargs or None,
    )

    # Show immediate status
    print_status(client, ft_id)

    if args.watch:
        tail_events(client, ft_id, poll_sec=args.poll_sec)


if __name__ == "__main__":
    main()
