#!/usr/bin/env -S uv run --with pyarrow --with pandas python
"""Curate nvidia/Nemotron-PII into JSONL training shards for the privacy-filter LoRA.

Reads train + test parquet, remaps the 50+ Nemotron labels to our 8-class
target vocabulary, converts CHARACTER offsets to UTF-8 BYTE offsets, validates
each row, then writes 30 shards of 100 examples each to
``/Volumes/P4510/.cache/datasets/curated/nemotron_NNN.jsonl``.

Deterministic given seed=42.

Run:
    uv run scripts/sample_nemotron.py
or directly (shebang uses uv):
    ./scripts/sample_nemotron.py
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

# ---- Config ----------------------------------------------------------------

SEED = 42
TRAIN_PARQUET = "/Volumes/P4510/.cache/datasets/Nemotron-PII/data/train-00000-of-00001.parquet"
TEST_PARQUET = "/Volumes/P4510/.cache/datasets/Nemotron-PII/data/test-00000-of-00001.parquet"
OUT_DIR = Path("/Volumes/P4510/.cache/datasets/curated")
N_FILES = 30
PER_FILE = 100
TOTAL_TARGET = N_FILES * PER_FILE  # 3000
MIN_SECRET_QUOTA = 200
MAX_TEXT_BYTES = 8000

TARGET_LABELS = {
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
}

# Direct label → target mapping. Anything not in this table AND not matching
# the suffix rules below is dropped (entity-level; the row is kept).
DIRECT_MAP: dict[str, str] = {
    # secret
    "password": "secret",
    "pin": "secret",
    "api_key": "secret",
    "key": "secret",
    "token": "secret",
    "secret": "secret",
    "passphrase": "secret",
    "private_key": "secret",
    "auth_token": "secret",
    "access_token": "secret",
    "client_secret": "secret",
    "aws_secret": "secret",
    "http_cookie": "secret",  # cookies are session secrets
    "cvv": "secret",
    # email
    "email": "private_email",
    "email_address": "private_email",
    # phone
    "phone_number": "private_phone",
    "phone": "private_phone",
    "telephone": "private_phone",
    "mobile": "private_phone",
    "fax_number": "private_phone",
    # address
    "street_address": "private_address",
    "address": "private_address",
    "building_number": "private_address",
    "city": "private_address",
    "state": "private_address",
    "zip": "private_address",
    "zip_code": "private_address",
    "postal_code": "private_address",
    "postcode": "private_address",
    "country": "private_address",
    "region": "private_address",
    "district": "private_address",
    "county": "private_address",
    "coordinate": "private_address",
    # person
    "first_name": "private_person",
    "last_name": "private_person",
    "full_name": "private_person",
    "name": "private_person",
    "person_name": "private_person",
    "middle_name": "private_person",
    "user_name": "private_person",
    "username": "private_person",
    # url
    "url": "private_url",
    "website": "private_url",
    "link": "private_url",
    "web_address": "private_url",
    # date
    "date": "private_date",
    "dob": "private_date",
    "date_of_birth": "private_date",
    "birth_date": "private_date",
    "birthday": "private_date",
    "date_time": "private_date",
    # account_number
    "account_number": "account_number",
    "bank_account": "account_number",
    "iban": "account_number",
    "credit_card_number": "account_number",
    "credit_debit_card": "account_number",
    "ssn": "account_number",
    "social_security_number": "account_number",
    "tax_id": "account_number",
    "driver_license": "account_number",
    "passport_number": "account_number",
    "national_id": "account_number",
    "customer_id": "account_number",
    "employee_id": "account_number",
    "medical_record_number": "account_number",
    "health_plan_beneficiary_number": "account_number",
    "bank_routing_number": "account_number",
    "swift_bic": "account_number",
    "certificate_license_number": "account_number",
    "license_plate": "account_number",
    "vehicle_identifier": "account_number",
    "unique_id": "account_number",
    "biometric_identifier": "account_number",
    "device_identifier": "account_number",
    "mac_address": "account_number",
    "ipv4": "account_number",
    "ipv6": "account_number",
}

SUFFIX_RULES = ("_key", "_token", "_password", "_secret")

# Labels we deliberately drop (not PII in our 8-class taxonomy):
# company_name, occupation, time (without date), employment_status, education_level,
# language, gender, age, religious_belief, race_ethnicity, blood_type,
# political_view, sexuality.


def map_label(src: str) -> str | None:
    if src in DIRECT_MAP:
        return DIRECT_MAP[src]
    if src.endswith(SUFFIX_RULES):
        return "secret"
    return None


# ---- Parsing helpers ------------------------------------------------------


def parse_spans(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    # The parquet column is a Python-literal string like "[{...}, {...}]".
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []


def char_to_byte_offsets(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    return (
        len(text[:char_start].encode("utf-8")),
        len(text[:char_end].encode("utf-8")),
    )


# ---- Row validation -------------------------------------------------------


def build_example(
    text: str, spans: list[dict], drop_stats: Counter
) -> tuple[dict | None, str | None]:
    """Return (example, None) on success or (None, reason) on drop."""
    if not text or not text.strip():
        return None, "empty_text"

    encoded = text.encode("utf-8")
    n_bytes = len(encoded)
    if n_bytes > MAX_TEXT_BYTES:
        return None, "text_too_long"

    entities: list[dict] = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        label_src = s.get("label")
        if not isinstance(label_src, str):
            continue
        tgt = map_label(label_src)
        if tgt is None:
            drop_stats[f"unmapped_label::{label_src}"] += 1
            continue
        try:
            cs = int(s["start"])
            ce = int(s["end"])
        except (KeyError, TypeError, ValueError):
            drop_stats["bad_span_offsets"] += 1
            continue
        if not (0 <= cs < ce <= len(text)):
            drop_stats["char_offset_oob"] += 1
            continue
        bs, be = char_to_byte_offsets(text, cs, ce)
        if not (0 <= bs < be <= n_bytes):
            return None, "byte_offset_oob"
        # Surface-text round-trip.
        expected_surface = s.get("text")
        actual_surface = encoded[bs:be].decode("utf-8", errors="strict")
        if expected_surface is not None and actual_surface != expected_surface:
            return None, "surface_mismatch"
        if actual_surface != text[cs:ce]:
            return None, "char_byte_mismatch"
        entities.append({"start": bs, "end": be, "label": tgt})

    if not entities:
        # Keep as a negative example (per spec).
        return {"text": text, "entities": []}, None

    entities.sort(key=lambda e: (e["start"], e["end"]))
    for prev, curr in zip(entities, entities[1:]):
        if curr["start"] < prev["end"]:
            return None, "overlap"

    return {"text": text, "entities": entities}, None


# ---- Main pipeline --------------------------------------------------------


def load_rows() -> list[dict]:
    """Combine train + test, keep only needed columns."""
    cols = ["text", "spans"]
    pieces = []
    for path in (TRAIN_PARQUET, TEST_PARQUET):
        table = pq.read_table(path, columns=cols)
        pieces.extend(table.to_pylist())
    return pieces


SECRET_PATTERN = re.compile(
    r"^(?:password|pin|api_key|key|token|secret|passphrase|private_key|auth_token|"
    r"access_token|client_secret|aws_secret|http_cookie|cvv|.*_key|.*_token|"
    r".*_password|.*_secret)$"
)


def row_has_secret(spans_raw) -> bool:
    for s in parse_spans(spans_raw):
        if not isinstance(s, dict):
            continue
        lbl = s.get("label")
        if isinstance(lbl, str) and map_label(lbl) == "secret":
            return True
    return False


def main() -> None:
    rng = random.Random(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading parquet from train+test...")
    rows = load_rows()
    print(f"loaded {len(rows)} rows total (train+test)")

    rng.shuffle(rows)

    # Partition into secret-bearing pool and general pool, preserving the
    # shuffled order so both pools are random.
    secret_pool: list[dict] = []
    general_pool: list[dict] = []
    for r in rows:
        if row_has_secret(r["spans"]):
            secret_pool.append(r)
        else:
            general_pool.append(r)
    print(
        f"pre-validation pools: secret_candidates={len(secret_pool)} "
        f"general_candidates={len(general_pool)}"
    )

    drop_stats: Counter = Counter()
    label_hist: Counter = Counter()

    accepted: list[dict] = []
    secret_accepted = 0

    # Pass 1: take up to MIN_SECRET_QUOTA secret-bearing rows.
    def consume(pool: list[dict], limit: int | None, want_secret_only: bool) -> int:
        nonlocal secret_accepted
        taken = 0
        while pool and (limit is None or taken < limit):
            r = pool.pop()
            text = r["text"]
            spans = parse_spans(r["spans"])
            example, reason = build_example(text, spans, drop_stats)
            if example is None:
                drop_stats[reason] += 1
                continue
            has_secret = any(e["label"] == "secret" for e in example["entities"])
            if want_secret_only and not has_secret:
                # Secret got mapped/dropped out; recycle to general for pass 2.
                general_pool.append(r)
                continue
            accepted.append(example)
            for e in example["entities"]:
                label_hist[e["label"]] += 1
            if has_secret:
                secret_accepted += 1
            taken += 1
            if len(accepted) >= TOTAL_TARGET:
                break
        return taken

    consume(secret_pool, MIN_SECRET_QUOTA, want_secret_only=True)
    print(f"after secret pass: accepted={len(accepted)} secret_rows={secret_accepted}")

    # Pass 2: fill the rest from general pool (and any recycled secret rows).
    if len(accepted) < TOTAL_TARGET:
        consume(general_pool, None, want_secret_only=False)

    # Pass 3: if still short (e.g. lots of drops), drain remaining secret pool.
    if len(accepted) < TOTAL_TARGET and secret_pool:
        consume(secret_pool, None, want_secret_only=False)

    if len(accepted) < TOTAL_TARGET:
        raise RuntimeError(
            f"only accepted {len(accepted)} examples, need {TOTAL_TARGET}; "
            f"top drop reasons: {drop_stats.most_common(5)}"
        )

    examples = accepted[:TOTAL_TARGET]

    # Write shards.
    total_written = 0
    for shard_idx in range(N_FILES):
        chunk = examples[shard_idx * PER_FILE : (shard_idx + 1) * PER_FILE]
        out_path = OUT_DIR / f"nemotron_{shard_idx:03d}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for ex in chunk:
                fh.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")
        total_written += len(chunk)
        dropped = sum(drop_stats.values())
        print(
            f"{out_path.name}: written {len(chunk)}, total {total_written}, "
            f"dropped_so_far {dropped}"
        )

    # ---- Verification --------------------------------------------------
    print("\n--- verification ---")
    final_label_hist: Counter = Counter()
    secret_rows_final = 0
    all_lines: list[str] = []
    for shard_idx in range(N_FILES):
        p = OUT_DIR / f"nemotron_{shard_idx:03d}.jsonl"
        with p.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) == PER_FILE, f"{p.name} has {len(lines)} lines"
        all_lines.extend(lines)
        for line in lines:
            obj = json.loads(line)
            has_secret = False
            for e in obj["entities"]:
                final_label_hist[e["label"]] += 1
                if e["label"] == "secret":
                    has_secret = True
            if has_secret:
                secret_rows_final += 1
    assert len(all_lines) == TOTAL_TARGET, f"total={len(all_lines)}"
    print(f"file count check OK: 30 files x 100 lines = {len(all_lines)}")

    # Round-trip 5 random examples.
    sample_rng = random.Random(SEED + 1)
    sample_idxs = sample_rng.sample(range(TOTAL_TARGET), 5)
    print("round-trip checks on 5 random examples:")
    for idx in sample_idxs:
        obj = json.loads(all_lines[idx])
        encoded = obj["text"].encode("utf-8")
        surfaces = []
        for e in obj["entities"]:
            surface = encoded[e["start"]:e["end"]].decode("utf-8")
            assert e["start"] < e["end"] <= len(encoded)
            surfaces.append(f'{e["label"]}={surface!r}')
        print(f"  idx={idx} text_len={len(obj['text'])} entities={len(obj['entities'])} {surfaces[:3]}")

    print("\nper-label histogram (entity counts across 3000 rows):")
    for lbl in sorted(TARGET_LABELS):
        print(f"  {lbl}: {final_label_hist.get(lbl, 0)}")
    print(f"rows containing at least one secret entity: {secret_rows_final}")

    # Drop reason summary.
    print("\ntop drop reasons:")
    for reason, n in drop_stats.most_common(10):
        print(f"  {reason}: {n}")
    total_dropped = sum(drop_stats.values())
    print(f"total dropped (entity- and row-level): {total_dropped}")

    # One literal example for the report.
    print("\nLITERAL example line (for report):")
    print(all_lines[sample_idxs[0]].rstrip())


if __name__ == "__main__":
    main()
