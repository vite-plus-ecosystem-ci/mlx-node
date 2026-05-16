#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Curate gretelai/gretel-pii-masking-en-v1 into the privacy-filter training format.

Output: 15 JSONL shards × 100 rows each at /Volumes/P4510/.cache/datasets/curated/gretel_NNN.jsonl

Schema (identical to nemotron_*):
    {"text": str, "entities": [{"start": byte_off, "end": byte_off, "label": str}]}
"""
from __future__ import annotations

import ast
import collections
import json
import random
import re
from pathlib import Path

import pandas as pd

# ---- paths ----
DATA_DIR = Path("/Volumes/P4510/.cache/datasets/gretel-pii-masking-en-v1/data")
OUT_DIR = Path("/Volumes/P4510/.cache/datasets/curated")
SHARD_COUNT = 15
ROWS_PER_SHARD = 100
TOTAL = SHARD_COUNT * ROWS_PER_SHARD  # 1500
SEED = 42
MAX_BYTES = 8000
MIN_SECRET_ROWS = 300
MAX_ALL_O_FRACTION = 0.10  # at most 10% all-O

# ---- domain prefilter ----
TARGET_DOMAIN_TOKENS = [
    "cybersecurity",
    "code-review",
    "cryptography",
    "digital-certificates",
    "information-technology",
    "technology-software",
    "software-development",
    "devops",
    "security",
    "networking",
    "network",
    "authentication",
    "cloud-services",
    "cloud-computing",
    "system-administration",
    "defense-security",
    "internet-services",
]

KEYWORD_RE = re.compile(
    r"\b(curl|wget|http|api[_-]?key|token|password|aws|ssh|tls|ssl|cert|crypto|sudo|kubectl|docker|git)\b",
    re.IGNORECASE,
)
# Looser fallback keyword set if the strict filter doesn't yield enough rows.
KEYWORD_RE_LOOSE = re.compile(
    r"\b(curl|wget|http|api[_-]?key|token|password|aws|ssh|tls|ssl|cert|crypto|sudo|"
    r"kubectl|docker|git|firewall|vpn|encryption|hash|cipher|server|client|ipv4|"
    r"ipv6|endpoint|gateway|router|proxy|auth|oauth|jwt|saml|ldap|kerberos)\b",
    re.IGNORECASE,
)

# ---- label remap ----
SECRET_KEYS = {
    "password", "api_key", "ssh_key", "private_key", "secret_key", "access_token",
    "bearer_token", "oauth_token", "client_secret", "aws_access_key", "aws_secret_key",
    "gcp_key", "azure_key", "jwt", "pgp_key", "certificate", "tls_certificate", "pin",
    "cvv", "passphrase",
}

LABEL_MAP: dict[str, str] = {}
for k in SECRET_KEYS:
    LABEL_MAP[k] = "secret"
LABEL_MAP.update({
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
    "company_name": "private_person",  # treat company brand mentions as person-like surface

    "url": "private_url",
    "website": "private_url",
    "domain_name": "private_url",

    "date": "private_date",
    "date_of_birth": "private_date",
    "dob": "private_date",
    "birthdate": "private_date",
    "date_time": "private_date",
    "time": "private_date",

    # account-number bucket (identifiers)
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


def map_label(src: str) -> str | None:
    """Map a Gretel entity type to the 8-class taxonomy or None to drop."""
    if not src:
        return None
    low = src.lower()
    if low in LABEL_MAP:
        return LABEL_MAP[low]
    # heuristic suffix routing -> secret
    for suf in ("_key", "_token", "_secret", "_password"):
        if low.endswith(suf):
            return "secret"
    return None


VALID_LABELS = {
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
}


def find_all_occurrences(text: str, needle: str) -> list[int]:
    """Return all char-offsets where needle occurs in text."""
    if not needle:
        return []
    out = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1  # allow overlapping-search; we dedup by exact span anyway
    return out


def char_to_byte(text: str, char_off: int) -> int:
    return len(text[:char_off].encode("utf-8"))


def build_record(text: str, entities_raw: list[dict], drop_counters: collections.Counter,
                 unmapped: collections.Counter) -> dict | None:
    """Convert a raw Gretel row to the curated schema, or return None to drop the row."""
    if not text:
        drop_counters["empty_text"] += 1
        return None
    text_bytes = text.encode("utf-8")
    if len(text_bytes) > MAX_BYTES:
        drop_counters["text_too_long"] += 1
        return None

    spans: list[tuple[int, int, str]] = []  # (byte_start, byte_end, label)
    seen: set[tuple[int, int]] = set()

    for ent in entities_raw:
        surface = ent.get("entity")
        types = ent.get("types") or []
        if not surface or not types:
            drop_counters["entity_missing_field"] += 1
            continue

        # Pick the first mappable type (Gretel rarely has >1).
        label = None
        src_type = None
        for t in types:
            mapped = map_label(t)
            if mapped is not None:
                label = mapped
                src_type = t
                break
        if label is None:
            for t in types:
                unmapped[t] += 1
            drop_counters["entity_unmapped"] += 1
            continue
        if label not in VALID_LABELS:
            drop_counters["entity_invalid_label"] += 1
            continue

        # Strip the surface for matching — Gretel sometimes includes leading whitespace.
        surface_clean = surface.strip()
        if not surface_clean:
            drop_counters["entity_blank_surface"] += 1
            continue

        # Try the literal surface first, then the stripped surface.
        char_positions = find_all_occurrences(text, surface)
        used_surface = surface
        if not char_positions and surface_clean != surface:
            char_positions = find_all_occurrences(text, surface_clean)
            used_surface = surface_clean
        if not char_positions:
            drop_counters["entity_not_found"] += 1
            continue

        for cp in char_positions:
            b_start = char_to_byte(text, cp)
            b_end = char_to_byte(text, cp + len(used_surface))
            if not (0 <= b_start < b_end <= len(text_bytes)):
                drop_counters["entity_bad_offsets"] += 1
                continue
            # Round-trip check
            if text_bytes[b_start:b_end].decode("utf-8", errors="strict") != used_surface:
                drop_counters["entity_roundtrip_fail"] += 1
                continue
            key = (b_start, b_end)
            if key in seen:
                continue
            seen.add(key)
            spans.append((b_start, b_end, label))

    # Sort and overlap-check
    spans.sort(key=lambda s: (s[0], s[1]))
    for i in range(1, len(spans)):
        prev_s, prev_e, _ = spans[i - 1]
        cur_s, cur_e, _ = spans[i]
        if cur_s < prev_e:
            drop_counters["overlap"] += 1
            return None

    entities = [{"start": s, "end": e, "label": l} for (s, e, l) in spans]
    return {"text": text, "entities": entities}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load all parquet files ----
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files in {DATA_DIR}")
    print(f"Loading {len(parquet_files)} parquet file(s): {[p.name for p in parquet_files]}")
    df = pd.concat(
        (pd.read_parquet(p) for p in parquet_files),
        ignore_index=True,
    )
    print(f"Total rows in source: {len(df)}")

    # ---- 2. Shuffle ----
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # ---- 3. Domain prefilter ----
    domain_lower = df["domain"].fillna("").str.lower()
    domain_mask = pd.Series(False, index=df.index)
    for tok in TARGET_DOMAIN_TOKENS:
        domain_mask |= domain_lower.str.contains(tok, regex=False, na=False)
    domain_filtered = df[domain_mask].reset_index(drop=True)
    print(f"After domain filter: {len(domain_filtered)} rows "
          f"(domains: {sorted(domain_filtered['domain'].unique().tolist())})")

    # Sanity: if too few, also include keyword-hits from the full corpus.
    pool = domain_filtered
    if len(pool) < TOTAL * 3:  # want a comfortable headroom
        kw_mask = df["text"].fillna("").str.contains(KEYWORD_RE, na=False)
        extras = df[kw_mask & ~domain_mask].reset_index(drop=True)
        print(f"Domain pool small; adding {len(extras)} keyword-hit rows "
              f"(strict regex).")
        pool = pd.concat([pool, extras], ignore_index=True)

    if len(pool) < TOTAL * 2:
        kw_loose_mask = df["text"].fillna("").str.contains(KEYWORD_RE_LOOSE, na=False)
        extras = df[kw_loose_mask & ~domain_mask & ~df.index.isin(pool.index)].reset_index(drop=True)
        print(f"Pool still small; relaxing keyword regex; adding {len(extras)} more rows.")
        pool = pd.concat([pool, extras], ignore_index=True)

    # Reshuffle the combined pool deterministically.
    pool = pool.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Final pool size: {len(pool)}")

    # ---- 4. Process every pool row, separate secret-bearing vs others ----
    secret_rows: list[dict] = []
    other_rows: list[dict] = []
    all_o_rows: list[dict] = []
    drop_counters: collections.Counter[str] = collections.Counter()
    unmapped: collections.Counter[str] = collections.Counter()
    parsed_ok = 0

    for _, raw in pool.iterrows():
        if len(secret_rows) + len(other_rows) + len(all_o_rows) >= TOTAL * 4:
            # we have an excess; stop processing more
            break
        try:
            ents = ast.literal_eval(raw["entities"]) if isinstance(raw["entities"], str) else raw["entities"]
        except Exception:
            drop_counters["entities_parse_error"] += 1
            continue
        if not isinstance(ents, list):
            drop_counters["entities_not_list"] += 1
            continue
        rec = build_record(raw["text"], ents, drop_counters, unmapped)
        if rec is None:
            continue
        parsed_ok += 1
        labels = {e["label"] for e in rec["entities"]}
        if "secret" in labels:
            secret_rows.append(rec)
        elif labels:
            other_rows.append(rec)
        else:
            all_o_rows.append(rec)

    print(
        f"Parsed OK: {parsed_ok} | secret-bearing: {len(secret_rows)} | "
        f"other-labeled: {len(other_rows)} | all-O: {len(all_o_rows)}"
    )

    # ---- 5. Stratify: pull >=300 secret rows first, then fill with non-secret ----
    rng = random.Random(SEED)
    rng.shuffle(secret_rows)
    rng.shuffle(other_rows)
    rng.shuffle(all_o_rows)

    secrets_needed = min(MIN_SECRET_ROWS, len(secret_rows))
    selected_secrets = secret_rows[:max(secrets_needed, min(len(secret_rows), MIN_SECRET_ROWS))]

    remaining_slots = TOTAL - len(selected_secrets)

    # Cap all-O at 10% of total
    max_all_o = int(TOTAL * MAX_ALL_O_FRACTION)
    selected_all_o = all_o_rows[:min(max_all_o, len(all_o_rows), remaining_slots)]
    remaining_slots -= len(selected_all_o)

    selected_others = other_rows[:remaining_slots]
    remaining_slots -= len(selected_others)

    # If we still don't have enough, top off with leftover secrets (already used) — won't happen unless data is tiny.
    if remaining_slots > 0:
        # Use any remaining secret rows beyond what we picked.
        leftover_secrets = secret_rows[len(selected_secrets):]
        topup = leftover_secrets[:remaining_slots]
        selected_secrets.extend(topup)
        remaining_slots -= len(topup)

    selected = selected_secrets + selected_others + selected_all_o
    if len(selected) < TOTAL:
        raise SystemExit(
            f"FATAL: only {len(selected)} valid rows (need {TOTAL}). "
            f"secrets={len(selected_secrets)} others={len(selected_others)} all_o={len(selected_all_o)}"
        )

    # Final shuffle so secret rows aren't all clustered at the start.
    rng.shuffle(selected)
    selected = selected[:TOTAL]

    # ---- 6. Write shards ----
    written = 0
    for shard in range(SHARD_COUNT):
        path = OUT_DIR / f"gretel_{shard:03d}.jsonl"
        rows = selected[shard * ROWS_PER_SHARD : (shard + 1) * ROWS_PER_SHARD]
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written += len(rows)
        print(
            f"gretel_{shard:03d}.jsonl: written {len(rows)}, total {written}, "
            f"dropped_so_far {sum(drop_counters.values())}"
        )

    # ---- 7. Verification ----
    print("\n=== Verification ===")
    total_lines = 0
    label_hist: collections.Counter[str] = collections.Counter()
    rows_with_secret = 0
    rows_all_o = 0
    all_loaded: list[dict] = []
    for shard in range(SHARD_COUNT):
        path = OUT_DIR / f"gretel_{shard:03d}.jsonl"
        with path.open("r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == ROWS_PER_SHARD, f"{path} has {len(lines)} lines"
        total_lines += len(lines)
        for rec in lines:
            all_loaded.append(rec)
            labels = [e["label"] for e in rec["entities"]]
            for l in labels:
                label_hist[l] += 1
            if "secret" in labels:
                rows_with_secret += 1
            if not labels:
                rows_all_o += 1
    print(f"Total lines across 15 shards: {total_lines} (expected {TOTAL})")

    # Round-trip 5 random examples
    sample_rng = random.Random(SEED + 1)
    samples = sample_rng.sample(all_loaded, 5)
    for i, rec in enumerate(samples):
        tb = rec["text"].encode("utf-8")
        for ent in rec["entities"]:
            slc = tb[ent["start"]:ent["end"]].decode("utf-8")
            print(f"  rt#{i} [{ent['label']}] -> {slc!r}")
        print()

    print("Per-label histogram:")
    for k, v in label_hist.most_common():
        print(f"  {k}: {v}")
    print(f"\nRows with >=1 secret: {rows_with_secret} (target >={MIN_SECRET_ROWS})")
    print(f"Rows all-O (no entities): {rows_all_o} (cap {int(TOTAL * MAX_ALL_O_FRACTION)})")

    print("\nDrop counters:")
    for k, v in drop_counters.most_common():
        print(f"  {k}: {v}")

    print("\nTop 10 unmapped source labels:")
    for k, v in unmapped.most_common(10):
        print(f"  {k}: {v}")

    print("\nSample literal output line (first row of gretel_000.jsonl):")
    with (OUT_DIR / "gretel_000.jsonl").open("r", encoding="utf-8") as f:
        print(f.readline().rstrip())


if __name__ == "__main__":
    main()
