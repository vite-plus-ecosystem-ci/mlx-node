#!/usr/bin/env python3
"""Concatenate curated/*.jsonl into train.jsonl + eval.jsonl with a stratified
90/10 split (seed=42). Stratifies by source-prefix so eval mirrors train.

Source-prefix groups (file basename prefix → group key):
  nemotron_NNN          -> "nemotron"
  nemotron_secrets_NNN  -> "nemotron"
  gretel_NNN            -> "gretel"
  gretel_secrets_NNN    -> "gretel"
  swe_chat_NNN          -> "swe_chat"
  swe_chat_secrets_*    -> "swe_chat"
  synthetic_cli_NNN     -> "synthetic_cli"
  synthetic_transcript_NNN -> "synthetic_transcript"

Skipped: any file containing "stage" or "audit" in its name.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
EVAL_FRACTION = 0.10
CURATED = Path("/Volumes/P4510/.cache/datasets/curated")
TRAIN_OUT = CURATED / "train.jsonl"
EVAL_OUT = CURATED / "eval.jsonl"


def group_for(path: Path) -> str:
    name = path.name
    if name.startswith("nemotron"):
        return "nemotron"
    if name.startswith("gretel"):
        return "gretel"
    if name.startswith("swe_chat"):
        return "swe_chat"
    if name.startswith("synthetic_cli"):
        return "synthetic_cli"
    if name.startswith("synthetic_transcript"):
        return "synthetic_transcript"
    return "other"


def main() -> None:
    files = sorted(p for p in CURATED.glob("*.jsonl") if "stage" not in p.name and "audit" not in p.name)
    by_group: dict[str, list[dict]] = defaultdict(list)
    seen_hashes: set[str] = set()
    dups = 0

    for f in files:
        group = group_for(f)
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            if not isinstance(text, str) or not text:
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                dups += 1
                continue
            seen_hashes.add(h)
            by_group[group].append(obj)

    rng = random.Random(SEED)
    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    print(f"Loaded {sum(len(v) for v in by_group.values())} unique rows from {len(files)} files; {dups} dups dropped")
    print()
    print(f"{'group':<22} {'total':>6} {'train':>6} {'eval':>5}")
    for group, rows in sorted(by_group.items()):
        rng.shuffle(rows)
        n_eval = max(1, round(len(rows) * EVAL_FRACTION))
        eval_part = rows[:n_eval]
        train_part = rows[n_eval:]
        eval_rows.extend(eval_part)
        train_rows.extend(train_part)
        print(f"{group:<22} {len(rows):>6} {len(train_part):>6} {len(eval_part):>5}")

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    TRAIN_OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n")
    EVAL_OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n")

    print()
    print(f"wrote {len(train_rows)} -> {TRAIN_OUT}")
    print(f"wrote {len(eval_rows)} -> {EVAL_OUT}")


if __name__ == "__main__":
    main()
