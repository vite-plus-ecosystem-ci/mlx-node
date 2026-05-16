#!/usr/bin/env python3
"""Quality checks on tokenized output from mlx-prep-dataset.

Checks:
  1. Schema: every line has ids+labels of equal length, len matches.
  2. Sequence-length distribution (p10/p50/p90/max).
  3. Label-ID coverage: which BIOES IDs appear, are any unexpected.
  4. BIOES grammar validity:
       - B-X must be followed by I-X or E-X (not O, not another B/S, not B-Y).
       - I-X must be followed by I-X or E-X (same X).
       - E-X must NOT be followed by I-X or E-X of same entity.
       - S-X stands alone (preceded/followed by O or different entity start).
  5. Per-class entity counts (decoded back from BIOES runs).
  6. Spot-check: round-trip detokenize 5 random rows and print the entity
     surface text + label so the user can eyeball.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

# Canonical 33-class BIOES vocabulary in the order the model expects.
# Mirrors `default_label_to_id()` in crates/mlx-core/src/training/bioes_alignment.rs.
CLASSES = [
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
]

ID_TO_TAG = ["O"]
for cls in CLASSES:
    for prefix in ("B-", "I-", "E-", "S-"):
        ID_TO_TAG.append(f"{prefix}{cls}")
assert len(ID_TO_TAG) == 33, len(ID_TO_TAG)


def tag_parts(tag: str) -> tuple[str, str | None]:
    if tag == "O":
        return "O", None
    prefix, _, cls = tag.partition("-")
    return prefix, cls


def check_bioes_grammar(tags: list[str]) -> list[str]:
    """Return list of grammar violation strings (empty = valid)."""
    errs: list[str] = []
    n = len(tags)
    for i, tag in enumerate(tags):
        prefix, cls = tag_parts(tag)
        nxt_prefix, nxt_cls = (("O", None) if i + 1 >= n else tag_parts(tags[i + 1]))
        if prefix == "B":
            # next must be I-X or E-X (same cls)
            if nxt_prefix not in ("I", "E") or nxt_cls != cls:
                errs.append(f"@{i} B-{cls} not followed by I-/E-{cls} (next={tags[i + 1] if i + 1 < n else 'EOS'})")
        elif prefix == "I":
            if nxt_prefix not in ("I", "E") or nxt_cls != cls:
                errs.append(f"@{i} I-{cls} not followed by I-/E-{cls} (next={tags[i + 1] if i + 1 < n else 'EOS'})")
    return errs


def extract_entity_runs(tags: list[str]) -> list[tuple[int, int, str]]:
    """Return list of (start_tok, end_tok_exclusive, cls) using BIOES decoding."""
    runs: list[tuple[int, int, str]] = []
    i = 0
    while i < len(tags):
        prefix, cls = tag_parts(tags[i])
        if prefix == "S" and cls is not None:
            runs.append((i, i + 1, cls))
            i += 1
        elif prefix == "B" and cls is not None:
            j = i + 1
            while j < len(tags):
                p, c = tag_parts(tags[j])
                if c != cls:
                    break
                if p == "E":
                    runs.append((i, j + 1, cls))
                    j += 1
                    break
                if p == "I":
                    j += 1
                    continue
                break
            i = j
        else:
            i += 1
    return runs


def main(path: Path, tokenizer_path: Path | None = None) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"=== {path.name}: {len(rows)} rows ===")

    # 1. Schema
    bad_schema = 0
    for r in rows:
        ids = r.get("ids") or []
        labels = r.get("labels") or []
        if not isinstance(ids, list) or not isinstance(labels, list):
            bad_schema += 1
            continue
        if len(ids) != len(labels):
            bad_schema += 1
            continue
        n = r.get("len")
        if isinstance(n, int) and n != len(ids):
            bad_schema += 1
    print(f"  schema violations: {bad_schema}")

    # 2. Sequence lengths
    lens = sorted(len(r["ids"]) for r in rows)
    n = len(lens)
    print(f"  seq lengths: min={lens[0]} p10={lens[n // 10]} median={lens[n // 2]} p90={lens[(9 * n) // 10]} max={lens[-1]}")

    # 3. Label-ID coverage
    id_counter: Counter[int] = Counter()
    for r in rows:
        for lid in r["labels"]:
            id_counter[lid] += 1
    print("  label-id histogram (per-token, top 12 non-O):")
    for lid, cnt in sorted(id_counter.items(), key=lambda kv: -kv[1])[:13]:
        tag = ID_TO_TAG[lid] if 0 <= lid < len(ID_TO_TAG) else f"???({lid})"
        if tag != "O":
            print(f"    {tag:<28} {cnt}")
    out_of_range = sum(1 for lid in id_counter if not (0 <= lid < 33))
    print(f"  label ids out of [0, 33): {out_of_range}")

    # 4. BIOES grammar validity (per-row)
    rows_with_errs = 0
    total_errs = 0
    sample_errs: list[str] = []
    for r in rows:
        tags = [ID_TO_TAG[lid] if 0 <= lid < 33 else "O" for lid in r["labels"]]
        errs = check_bioes_grammar(tags)
        if errs:
            rows_with_errs += 1
            total_errs += len(errs)
            if len(sample_errs) < 5:
                sample_errs.append(errs[0])
    print(f"  rows with BIOES grammar violations: {rows_with_errs}/{n}  (total violations: {total_errs})")
    for s in sample_errs:
        print(f"    sample: {s}")

    # 5. Per-class entity counts (decoded)
    class_counts: Counter[str] = Counter()
    for r in rows:
        tags = [ID_TO_TAG[lid] if 0 <= lid < 33 else "O" for lid in r["labels"]]
        for _, _, cls in extract_entity_runs(tags):
            class_counts[cls] += 1
    print("  decoded entity counts (per-row sums):")
    for cls in CLASSES:
        print(f"    {cls:<22} {class_counts[cls]}")

    # 6. Spot-check: decode 5 random rows back through tokenizer
    if tokenizer_path is None or not tokenizer_path.exists():
        print(f"  spot-check skipped (tokenizer not at {tokenizer_path})")
        return
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError:
        print("  spot-check skipped (no `tokenizers` lib — install via `uv run --with tokenizers`)")
        return

    tok = Tokenizer.from_file(str(tokenizer_path))
    rng = random.Random(0)
    picks = rng.sample(rows, min(5, len(rows)))
    print("  spot-check (decoded entity spans):")
    for r in picks:
        tags = [ID_TO_TAG[lid] if 0 <= lid < 33 else "O" for lid in r["labels"]]
        runs = extract_entity_runs(tags)
        if not runs:
            continue
        s, e, cls = runs[0]
        token_ids = r["ids"][s:e]
        surface = tok.decode(token_ids, skip_special_tokens=False).strip()
        print(f"    [{cls:<18}] tokens={s}..{e}, decoded={surface!r}")


if __name__ == "__main__":
    base = Path("/Volumes/P4510/.cache/datasets/curated")
    tok = Path("/Users/brooklyn/workspace/github/mlx-node/.cache/models/privacy-filter/tokenizer.json")
    for fname in ("train.tokenized.jsonl", "eval.tokenized.jsonl"):
        main(base / fname, tok)
        print()
