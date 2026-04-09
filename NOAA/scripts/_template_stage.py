#!/usr/bin/env python3
"""Pipeline stage template for the ship trajectory benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_v1.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[TODO] implement {Path(__file__).name}")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"config={args.config}")


if __name__ == "__main__":
    main()
