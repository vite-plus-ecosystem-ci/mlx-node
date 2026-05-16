#!/usr/bin/env -S uv run --with pyarrow --with pandas python
"""Mine additional secret-bearing rows from Nemotron-PII and Gretel-PII parquets.

Targets:
  - 800 Nemotron rows (8 shards x 100), each with >=1 `secret` entity after remap.
  - 400 Gretel rows (4 shards x 100), each with >=1 `secret` entity after remap.

Hard rules (identical to scripts/sample_nemotron.py & scripts/sample_gretel.py):
  - Byte-offset half-open intervals
  - No overlapping spans
  - Surface round-trip (byte-slice -> decode == source surface)
  - text <= 8000 bytes, non-empty
  - Label remap tables imported below (same as originals)

Dedup: drop any row whose SHA256(text) already exists in any
`/Volumes/P4510/.cache/datasets/curated/*.jsonl` file.

Seed=43 (intentionally different from previous batches).

Output:
  /Volumes/P4510/.cache/datasets/curated/nemotron_secrets_NNN.jsonl
  /Volumes/P4510/.cache/datasets/curated/gretel_secrets_NNN.jsonl
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SEED = 43
OUT_DIR = Path("/Volumes/P4510/.cache/datasets/curated")
ROWS_PER_SHARD = 100
MAX_TEXT_BYTES = 8000

NEMOTRON_TRAIN = "/Volumes/P4510/.cache/datasets/Nemotron-PII/data/train-00000-of-00001.parquet"
NEMOTRON_TEST = "/Volumes/P4510/.cache/datasets/Nemotron-PII/data/test-00000-of-00001.parquet"
NEMOTRON_TARGET = 800
NEMOTRON_MAX_SHARDS = 8

GRETEL_DIR = Path("/Volumes/P4510/.cache/datasets/gretel-pii-masking-en-v1/data")
GRETEL_TARGET = 400
GRETEL_MAX_SHARDS = 4

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

# --------------------------------------------------------------------------- #
# Nemotron label remap (copied verbatim from scripts/sample_nemotron.py)
# --------------------------------------------------------------------------- #
NEMOTRON_DIRECT_MAP: dict[str, str] = {
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
    "http_cookie": "secret",
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

NEMOTRON_SUFFIX_RULES = ("_key", "_token", "_password", "_secret")


def nemotron_map_label(src: str) -> str | None:
    if src in NEMOTRON_DIRECT_MAP:
        return NEMOTRON_DIRECT_MAP[src]
    if src.endswith(NEMOTRON_SUFFIX_RULES):
        return "secret"
    return None


# --------------------------------------------------------------------------- #
# Gretel label remap (copied verbatim from scripts/sample_gretel.py)
# --------------------------------------------------------------------------- #
GRETEL_SECRET_KEYS = {
    "password", "api_key", "ssh_key", "private_key", "secret_key", "access_token",
    "bearer_token", "oauth_token", "client_secret", "aws_access_key", "aws_secret_key",
    "gcp_key", "azure_key", "jwt", "pgp_key", "certificate", "tls_certificate", "pin",
    "cvv", "passphrase",
}

GRETEL_LABEL_MAP: dict[str, str] = {}
for _k in GRETEL_SECRET_KEYS:
    GRETEL_LABEL_MAP[_k] = "secret"
GRETEL_LABEL_MAP.update({
    "email": "private_email",
    "email_address": "private_email",
    "phone_number": "private_phone",
    "phone": "private_phone",
    "mobile_phone": "private_phone",
    "fax": "private_phone",
    "fax_number": "private_phone",
    "address": "private_address",
    "street_address": "private_address",
    "city": "private_address",
    "state": "private_address",
    "country": "private_address",
    "zipcode": "private_address",
    "postal_code": "private_address",
    "postcode": "private_address",
    "region": "private_address",
    "province": "private_address",
    "first_name": "private_person",
    "last_name": "private_person",
    "name": "private_person",
    "full_name": "private_person",
    "username": "private_person",
    "user_name": "private_person",
    "customer_name": "private_person",
    "company_name": "private_person",
    "url": "private_url",
    "website": "private_url",
    "domain_name": "private_url",
    "date": "private_date",
    "date_of_birth": "private_date",
    "dob": "private_date",
    "birthdate": "private_date",
    "date_time": "private_date",
    "time": "private_date",
    "account_number": "account_number",
    "bank_account": "account_number",
    "bank_routing_number": "account_number",
    "iban": "account_number",
    "swift_bic": "account_number",
    "credit_card": "account_number",
    "credit_card_number": "account_number",
    "ssn": "account_number",
    "tax_id": "account_number",
    "passport_number": "account_number",
    "national_id": "account_number",
    "driver_license": "account_number",
    "customer_id": "account_number",
    "employee_id": "account_number",
    "unique_identifier": "account_number",
    "id": "account_number",
    "mac_address": "account_number",
    "ip_address": "account_number",
    "ipv4": "account_number",
    "ipv6": "account_number",
    "mac": "account_number",
    "medical_record_number": "account_number",
    "health_plan_beneficiary_number": "account_number",
    "certificate_license_number": "account_number",
    "license_plate": "account_number",
    "vehicle_identifier": "account_number",
    "device_identifier": "account_number",
    "biometric_identifier": "account_number",
    "coordinate": "account_number",
})


def gretel_map_label(src: str) -> str | None:
    if not src:
        return None
    low = src.lower()
    if low in GRETEL_LABEL_MAP:
        return GRETEL_LABEL_MAP[low]
    for suf in ("_key", "_token", "_secret", "_password"):
        if low.endswith(suf):
            return "secret"
    return None


# --------------------------------------------------------------------------- #
# Common helpers
# --------------------------------------------------------------------------- #


def parse_spans(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
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


def find_all_occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    out = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Build per-dataset example
# --------------------------------------------------------------------------- #


def build_nemotron_example(text: str, spans: list[dict]) -> tuple[dict | None, str | None]:
    """Same logic as sample_nemotron.build_example, but row is dropped if no `secret`."""
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
        tgt = nemotron_map_label(label_src)
        if tgt is None:
            continue
        try:
            cs = int(s["start"])
            ce = int(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= cs < ce <= len(text)):
            continue
        bs, be = char_to_byte_offsets(text, cs, ce)
        if not (0 <= bs < be <= n_bytes):
            return None, "byte_offset_oob"
        expected_surface = s.get("text")
        actual_surface = encoded[bs:be].decode("utf-8", errors="strict")
        if expected_surface is not None and actual_surface != expected_surface:
            return None, "surface_mismatch"
        if actual_surface != text[cs:ce]:
            return None, "char_byte_mismatch"
        entities.append({"start": bs, "end": be, "label": tgt})

    if not entities:
        return None, "no_entities"

    entities.sort(key=lambda e: (e["start"], e["end"]))
    for prev, curr in zip(entities, entities[1:]):
        if curr["start"] < prev["end"]:
            return None, "overlap"

    if not any(e["label"] == "secret" for e in entities):
        return None, "no_secret"

    return {"text": text, "entities": entities}, None


def build_gretel_example(text: str, entities_raw: list[dict]) -> tuple[dict | None, str | None]:
    """Same logic as sample_gretel.build_record, but row is dropped if no `secret`."""
    if not text:
        return None, "empty_text"
    text_bytes = text.encode("utf-8")
    if len(text_bytes) > MAX_TEXT_BYTES:
        return None, "text_too_long"

    spans: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    for ent in entities_raw:
        if not isinstance(ent, dict):
            continue
        surface = ent.get("entity")
        types = ent.get("types") or []
        if not surface or not types:
            continue

        label = None
        for t in types:
            mapped = gretel_map_label(t)
            if mapped is not None:
                label = mapped
                break
        if label is None or label not in TARGET_LABELS:
            continue

        surface_clean = surface.strip()
        if not surface_clean:
            continue

        char_positions = find_all_occurrences(text, surface)
        used_surface = surface
        if not char_positions and surface_clean != surface:
            char_positions = find_all_occurrences(text, surface_clean)
            used_surface = surface_clean
        if not char_positions:
            continue

        for cp in char_positions:
            b_start = char_to_byte(text, cp)
            b_end = char_to_byte(text, cp + len(used_surface))
            if not (0 <= b_start < b_end <= len(text_bytes)):
                continue
            if text_bytes[b_start:b_end].decode("utf-8", errors="strict") != used_surface:
                continue
            key = (b_start, b_end)
            if key in seen:
                continue
            seen.add(key)
            spans.append((b_start, b_end, label))

    spans.sort(key=lambda s: (s[0], s[1]))
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            return None, "overlap"

    if not spans:
        return None, "no_entities"

    entities = [{"start": s, "end": e, "label": l} for (s, e, l) in spans]
    if not any(e["label"] == "secret" for e in entities):
        return None, "no_secret"
    return {"text": text, "entities": entities}, None


def char_to_byte(text: str, char_off: int) -> int:
    return len(text[:char_off].encode("utf-8"))


# --------------------------------------------------------------------------- #
# Dedup index
# --------------------------------------------------------------------------- #


def load_existing_text_hashes() -> set[str]:
    """SHA256 of every `text` field already present in curated/*.jsonl."""
    seen: set[str] = set()
    files = sorted(OUT_DIR.glob("*.jsonl"))
    print(f"loading existing text hashes from {len(files)} curated files...")
    for p in files:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("text")
                if isinstance(t, str):
                    seen.add(sha256_text(t))
    print(f"existing text-hash index size: {len(seen)}")
    return seen


# --------------------------------------------------------------------------- #
# Cheap pre-filters for "row plausibly contains a secret entity"
# --------------------------------------------------------------------------- #


def nemotron_row_has_secret_quick(spans_raw) -> bool:
    for s in parse_spans(spans_raw):
        if not isinstance(s, dict):
            continue
        lbl = s.get("label")
        if isinstance(lbl, str) and nemotron_map_label(lbl) == "secret":
            return True
    return False


def gretel_row_has_secret_quick(entities_raw) -> bool:
    """Look at the `types` lists of each entity dict."""
    if isinstance(entities_raw, str):
        try:
            ents = ast.literal_eval(entities_raw)
        except (ValueError, SyntaxError):
            return False
    else:
        ents = entities_raw
    if not isinstance(ents, list):
        return False
    for ent in ents:
        if not isinstance(ent, dict):
            continue
        for t in (ent.get("types") or []):
            if isinstance(t, str) and gretel_map_label(t) == "secret":
                return True
    return False


# --------------------------------------------------------------------------- #
# Pipeline: Nemotron
# --------------------------------------------------------------------------- #


def process_nemotron(text_hash_idx: set[str]) -> tuple[list[dict], dict]:
    print("\n=== NEMOTRON ===")
    rng = random.Random(SEED)
    cols = ["text", "spans"]
    pieces = []
    for path in (NEMOTRON_TRAIN, NEMOTRON_TEST):
        table = pq.read_table(path, columns=cols)
        pieces.extend(table.to_pylist())
    print(f"loaded {len(pieces)} nemotron rows")
    rng.shuffle(pieces)

    accepted: list[dict] = []
    scanned = 0
    dups = 0
    no_secret = 0
    other_drops = 0
    drop_reasons: collections.Counter[str] = collections.Counter()

    for r in pieces:
        if len(accepted) >= NEMOTRON_TARGET:
            break
        scanned += 1
        text = r.get("text")
        if not isinstance(text, str):
            other_drops += 1
            continue
        if not nemotron_row_has_secret_quick(r.get("spans")):
            no_secret += 1
            continue
        h = sha256_text(text)
        if h in text_hash_idx:
            dups += 1
            continue
        spans = parse_spans(r.get("spans"))
        example, reason = build_nemotron_example(text, spans)
        if example is None:
            drop_reasons[reason or "unknown"] += 1
            if reason == "no_secret":
                no_secret += 1
            else:
                other_drops += 1
            continue
        accepted.append(example)
        text_hash_idx.add(h)

    stats = {
        "scanned": scanned,
        "dups_skipped": dups,
        "no_secret": no_secret,
        "other_drops": other_drops,
        "drop_reasons": dict(drop_reasons),
        "pool_size": len(pieces),
    }
    print(
        f"nemotron scanned={scanned} accepted={len(accepted)} dups={dups} "
        f"no_secret={no_secret} other_drops={other_drops}"
    )
    return accepted, stats


# --------------------------------------------------------------------------- #
# Pipeline: Gretel
# --------------------------------------------------------------------------- #


def process_gretel(text_hash_idx: set[str]) -> tuple[list[dict], dict]:
    print("\n=== GRETEL ===")
    rng = random.Random(SEED)
    parquet_files = sorted(GRETEL_DIR.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files in {GRETEL_DIR}")
    df = pd.concat(
        (pd.read_parquet(p) for p in parquet_files), ignore_index=True
    )
    print(f"loaded {len(df)} gretel rows")
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    accepted: list[dict] = []
    scanned = 0
    dups = 0
    no_secret = 0
    other_drops = 0
    drop_reasons: collections.Counter[str] = collections.Counter()

    for _, raw in df.iterrows():
        if len(accepted) >= GRETEL_TARGET:
            break
        scanned += 1
        text = raw.get("text") if hasattr(raw, "get") else raw["text"]
        if not isinstance(text, str):
            other_drops += 1
            continue
        ents_raw = raw["entities"]
        if not gretel_row_has_secret_quick(ents_raw):
            no_secret += 1
            continue
        h = sha256_text(text)
        if h in text_hash_idx:
            dups += 1
            continue
        try:
            ents = ast.literal_eval(ents_raw) if isinstance(ents_raw, str) else ents_raw
        except Exception:
            drop_reasons["entities_parse_error"] += 1
            other_drops += 1
            continue
        if not isinstance(ents, list):
            drop_reasons["entities_not_list"] += 1
            other_drops += 1
            continue
        example, reason = build_gretel_example(text, ents)
        if example is None:
            drop_reasons[reason or "unknown"] += 1
            if reason == "no_secret":
                no_secret += 1
            else:
                other_drops += 1
            continue
        accepted.append(example)
        text_hash_idx.add(h)

    stats = {
        "scanned": scanned,
        "dups_skipped": dups,
        "no_secret": no_secret,
        "other_drops": other_drops,
        "drop_reasons": dict(drop_reasons),
        "pool_size": len(df),
    }
    print(
        f"gretel scanned={scanned} accepted={len(accepted)} dups={dups} "
        f"no_secret={no_secret} other_drops={other_drops}"
    )
    return accepted, stats


# --------------------------------------------------------------------------- #
# Writing shards
# --------------------------------------------------------------------------- #


def write_shards(prefix: str, max_shards: int, examples: list[dict]) -> list[Path]:
    paths: list[Path] = []
    written_total = 0
    no_secret_count = 0
    dups_count = 0  # already-deduped upstream; included only for the log line shape
    n_full_shards = min(max_shards, len(examples) // ROWS_PER_SHARD)

    for i in range(n_full_shards):
        chunk = examples[i * ROWS_PER_SHARD : (i + 1) * ROWS_PER_SHARD]
        path = OUT_DIR / f"{prefix}_{i:03d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ex in chunk:
                fh.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")
        paths.append(path)
        written_total += len(chunk)
        print(
            f"{path.name}: written {len(chunk)}, total_secret_rows {written_total}, "
            f"dups_skipped {dups_count}, no_secret {no_secret_count}"
        )

    # Partial shard for the remainder, if any AND we haven't hit max_shards yet.
    consumed = n_full_shards * ROWS_PER_SHARD
    remainder = examples[consumed:]
    if remainder and n_full_shards < max_shards:
        idx = n_full_shards
        path = OUT_DIR / f"{prefix}_{idx:03d}_partial.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ex in remainder:
                fh.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")
        paths.append(path)
        written_total += len(remainder)
        print(
            f"{path.name}: written {len(remainder)}, total_secret_rows {written_total}, "
            f"dups_skipped {dups_count}, no_secret {no_secret_count}"
        )

    return paths


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def verify(paths: list[Path], label: str) -> dict:
    print(f"\n--- verify {label} ---")
    all_lines: list[tuple[Path, str]] = []
    hist: collections.Counter[str] = collections.Counter()
    rows_with_secret = 0
    for p in paths:
        with p.open("r", encoding="utf-8") as fh:
            lines = [l for l in fh if l.strip()]
        if p.name.endswith("_partial.jsonl"):
            assert len(lines) < ROWS_PER_SHARD and len(lines) > 0, f"{p.name} partial size {len(lines)}"
        else:
            assert len(lines) == ROWS_PER_SHARD, f"{p.name} has {len(lines)} lines (expected {ROWS_PER_SHARD})"
        for ln in lines:
            obj = json.loads(ln)
            ents = obj["entities"]
            assert isinstance(ents, list) and len(ents) > 0, f"empty entities in {p.name}"
            has_secret = False
            for e in ents:
                hist[e["label"]] += 1
                if e["label"] == "secret":
                    has_secret = True
            assert has_secret, f"row in {p.name} has NO secret entity"
            if has_secret:
                rows_with_secret += 1
            all_lines.append((p, ln))

    total = len(all_lines)
    print(f"{label}: files={len(paths)} rows={total} rows_with_secret={rows_with_secret}")

    sample_rng = random.Random(SEED + 1)
    sample = sample_rng.sample(all_lines, min(5, len(all_lines)))
    print(f"round-trip 5 random spans ({label}):")
    for src_path, ln in sample:
        obj = json.loads(ln)
        encoded = obj["text"].encode("utf-8")
        # pick 1 entity at random
        ent = random.Random(SEED + 2).choice(obj["entities"])  # deterministic-ish, fine for log
        surface = encoded[ent["start"]:ent["end"]].decode("utf-8", errors="strict")
        print(f"  {src_path.name} label={ent['label']} surface={surface!r}")

    print(f"per-label histogram ({label}):")
    for lbl in sorted(TARGET_LABELS):
        print(f"  {lbl}: {hist.get(lbl, 0)}")

    return {
        "files": len(paths),
        "rows": total,
        "rows_with_secret": rows_with_secret,
        "label_hist": dict(hist),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text_hash_idx = load_existing_text_hashes()
    starting_idx_size = len(text_hash_idx)

    nem_examples, nem_stats = process_nemotron(text_hash_idx)
    gre_examples, gre_stats = process_gretel(text_hash_idx)

    nem_paths = write_shards("nemotron_secrets", NEMOTRON_MAX_SHARDS, nem_examples)
    gre_paths = write_shards("gretel_secrets", GRETEL_MAX_SHARDS, gre_examples)

    nem_v = verify(nem_paths, "nemotron")
    gre_v = verify(gre_paths, "gretel")

    print("\n=== FINAL REPORT ===")
    print(f"starting curated hash-index size: {starting_idx_size}")
    print(f"nemotron rows written: {nem_v['rows']} (target {NEMOTRON_TARGET})")
    print(f"  scanned to hit: {nem_stats['scanned']} (pool size {nem_stats['pool_size']})")
    print(f"  dups skipped: {nem_stats['dups_skipped']}")
    print(f"  no_secret skipped: {nem_stats['no_secret']}")
    print(f"  other drops: {nem_stats['other_drops']} | reasons: {nem_stats['drop_reasons']}")
    print(f"gretel rows written:   {gre_v['rows']} (target {GRETEL_TARGET})")
    print(f"  scanned to hit: {gre_stats['scanned']} (pool size {gre_stats['pool_size']})")
    print(f"  dups skipped: {gre_stats['dups_skipped']}")
    print(f"  no_secret skipped: {gre_stats['no_secret']}")
    print(f"  other drops: {gre_stats['other_drops']} | reasons: {gre_stats['drop_reasons']}")

    combined_hist: collections.Counter[str] = collections.Counter()
    for lbl, n in nem_v["label_hist"].items():
        combined_hist[lbl] += n
    for lbl, n in gre_v["label_hist"].items():
        combined_hist[lbl] += n
    print("\ncombined per-label histogram across new batch:")
    for lbl in sorted(TARGET_LABELS):
        print(f"  {lbl}: {combined_hist.get(lbl, 0)}")

    print("\nliteral example (nemotron):")
    with nem_paths[0].open("r", encoding="utf-8") as fh:
        print(fh.readline().rstrip())

    print("\nliteral example (gretel):")
    with gre_paths[0].open("r", encoding="utf-8") as fh:
        print(fh.readline().rstrip())


if __name__ == "__main__":
    main()
