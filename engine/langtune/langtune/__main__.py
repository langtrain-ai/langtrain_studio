"""`python -m langtune train --config job.json`"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="langtune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train a LoRA adapter")
    t.add_argument("--config", help="job config JSON (as written by langtrain serve)")
    t.add_argument("--base", help="base model .gguf (instead of --config)")
    t.add_argument("--data", help="training JSONL (instead of --config)")
    t.add_argument("--out", default=".", help="output directory")
    t.add_argument("--name", default="adapter")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--train-layers", type=int)
    t.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    from .trainer import main_train

    if args.config:
        return main_train(args.config)
    if not (args.base and args.data):
        ap.error("pass --config, or --base and --data")
    cfg = {
        "base_gguf": args.base,
        "train_file": args.data,
        "out_dir": args.out,
        "adapter_name": args.name,
        "epochs": args.epochs,
        "train_layers": args.train_layers,
        "device": args.device,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
    return main_train(f.name)


if __name__ == "__main__":
    sys.exit(main())
