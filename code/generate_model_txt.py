"""Generate model.txt from a YOLOv7 QuantModel .pt checkpoint."""
from __future__ import annotations

import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Generate model.txt from .pt checkpoint")
    parser.add_argument("--weights",    required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output",     required=True, help="Path to write model.txt")
    parser.add_argument("--yolov7-dir", default="",    help="YOLOv7 repo root to add to sys.path")
    args = parser.parse_args()

    # torch.load unpickles model classes; they import from yolov7's `models/` package
    if args.yolov7_dir:
        sys.path.insert(0, args.yolov7_dir)

    print(f"  Loading checkpoint: {args.weights}")
    ckpt = torch.load(args.weights, map_location="cpu")

    # EMA weights are the best inference weights; fall back to raw model
    model = ckpt.get("ema") or ckpt.get("model")
    if model is None:
        print(f"ERROR: checkpoint keys = {list(ckpt.keys())}", file=sys.stderr)
        print("ERROR: expected 'ema' or 'model' key in checkpoint", file=sys.stderr)
        sys.exit(1)

    # float() ensures repr shows full precision FL values, not half
    if hasattr(model, "float"):
        model = model.float()

    txt = str(model)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(txt)

    lines = txt.count("\n")
    print(f"  Saved model.txt -> {args.output}  ({lines} lines)")


if __name__ == "__main__":
    main()
