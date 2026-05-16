#!/usr/bin/env -S uv run --with orjson python
"""Mine SALT-NLP/SWE-chat transcripts for high-confidence secret regex matches.

The SWE-chat dataset was pre-redacted with TruffleHog before publication, but
TruffleHog is not perfect. We rescan every transcript with a narrow set of
high-precision regex patterns (distinctive vendor prefixes — ghp_, sk-ant-,
AKIA, etc.) and emit any matching turns as positive training rows labeled with
`secret` entities. This is the only batch giving us positive examples in the
CLI/transcript register the LoRA needs to learn.

Reuses the file-iteration + content-extraction logic from sample_swe_chat.py
(four formats: Claude-Code JSONL, Codex JSONL, OpenCode JSON, Gemini JSON).

Hard caps:
    * <= 500 rows total
    * 100 rows per shard, last shard may be partial -> swe_chat_secrets_partial.jsonl
    * 100 <= len(utf8) <= 4000 bytes
    * Skip text-hash dups vs prior curated files + within-batch
    * Dedupe overlapping regex matches (keep longest)

Side outputs:
    * swe_chat_secrets_audit.jsonl — {file_id, pattern_type, surface_first_8,
      surface_last_4} for each kept span (no full secrets).

Deterministic with seed=44.
"""

from __future__ import annotations

import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path

import orjson

# ---- Config ----------------------------------------------------------------

SEED = 44
SRC_DIR = Path("/Volumes/P4510/.cache/datasets/SWE-chat/transcripts")
OUT_DIR = Path("/Volumes/P4510/.cache/datasets/curated")
SHARD_PREFIX = "swe_chat_secrets"
PER_SHARD = 100
MAX_TOTAL = 500
AUDIT_PATH = OUT_DIR / "swe_chat_secrets_audit.jsonl"

MIN_BYTES = 100
MAX_BYTES = 4000

# ---- Regex set -------------------------------------------------------------

PATTERNS: dict[str, re.Pattern[str]] = {
    "github_classic": re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    "github_oauth": re.compile(r"\bgho_[A-Za-z0-9]{36}\b"),
    "github_server": re.compile(r"\bghs_[A-Za-z0-9]{36}\b"),
    "github_refresh": re.compile(r"\bghr_[A-Za-z0-9]{36}\b"),
    "github_user_pat": re.compile(r"\bghu_[A-Za-z0-9]{36}\b"),
    "github_fine": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82,}"),
    "openai_classic": re.compile(r"\bsk-[A-Za-z0-9]{40,}\b"),
    "openai_project": re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{40,}\b"),
    "anthropic": re.compile(r"\bsk-ant-(?:api|admin)\d+-[A-Za-z0-9_\-]{50,}\b"),
    "aws_akid": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "aws_temp": re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    "slack": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,72}\b"),
    "huggingface": re.compile(r"\bhf_[A-Za-z0-9]{34,}\b"),
    "cloudflare_token": re.compile(r"\bcfut_[A-Za-z0-9]{40,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    "google_api": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "stripe_live": re.compile(r"\bsk_live_[A-Za-z0-9]{20,}\b"),
    "stripe_test": re.compile(r"\bsk_test_[A-Za-z0-9]{20,}\b"),
    "private_key_hdr": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED |PGP )?PRIVATE KEY-----"
    ),
}


# ---- Transcript walker (cloned from sample_swe_chat.py) -------------------


def _flatten_parts(parts) -> str:
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, str):
            chunks.append(p)
            continue
        if not isinstance(p, dict):
            continue
        pt = p.get("type")
        if pt in ("text", "input_text", "output_text"):
            txt = p.get("text")
            if isinstance(txt, str):
                chunks.append(txt)
        elif pt == "thinking":
            txt = p.get("thinking")
            if isinstance(txt, str):
                chunks.append(txt)
        elif pt == "reasoning":
            txt = p.get("text")
            if isinstance(txt, str):
                chunks.append(txt)
    return "\n".join(chunks)


def _tool_use_text(part: dict) -> str | None:
    name = part.get("name", "")
    inp = part.get("input") or {}
    if not isinstance(inp, dict):
        return None
    cmd = inp.get("command")
    if isinstance(cmd, str) and cmd.strip():
        desc = inp.get("description")
        prefix = f"# {desc}\n" if isinstance(desc, str) and desc.strip() else ""
        return f"{prefix}$ {cmd}"
    fp = inp.get("file_path") or inp.get("notebook_path") or inp.get("path")
    if name == "Read" and isinstance(fp, str):
        return f"Read {fp}"
    if name in ("Write", "Edit", "MultiEdit") and isinstance(fp, str):
        return f"{name} {fp}"
    if name == "Grep":
        pat = inp.get("pattern")
        path = inp.get("path", "")
        if isinstance(pat, str):
            return f"Grep -E {pat!r}" + (f" {path}" if path else "")
    return None


def _tool_result_text(part: dict) -> str | None:
    c = part.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return _flatten_parts(c) or None
    return None


def _claude_msg_text(msg: dict, *, include_tools: bool) -> str | None:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    chunks: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            continue
        pt = p.get("type")
        if pt in ("text", "input_text", "output_text"):
            txt = p.get("text")
            if isinstance(txt, str):
                chunks.append(txt)
        elif pt == "thinking":
            txt = p.get("thinking")
            if isinstance(txt, str):
                chunks.append(txt)
        elif include_tools and pt == "tool_use":
            t = _tool_use_text(p)
            if t:
                chunks.append(t)
        elif include_tools and pt == "tool_result":
            t = _tool_result_text(p)
            if t:
                chunks.append(t)
    return "\n".join(chunks) or None


def _opencode_part_text(p: dict) -> str | None:
    pt = p.get("type")
    if pt in ("text", "reasoning"):
        return p.get("text") if isinstance(p.get("text"), str) else None
    if pt == "tool":
        tool = p.get("tool")
        state = p.get("state") or {}
        if not isinstance(state, dict):
            return None
        inp = state.get("input") or {}
        out = state.get("output")
        if tool == "bash" and isinstance(inp, dict):
            cmd = inp.get("command")
            if isinstance(cmd, str):
                if isinstance(out, str) and out.strip():
                    return f"$ {cmd}\n{out}"
                return f"$ {cmd}"
        if tool in ("read", "write", "edit") and isinstance(inp, dict):
            fp = inp.get("filePath") or inp.get("file_path") or inp.get("path")
            if isinstance(fp, str):
                if isinstance(out, str) and out.strip() and len(out) < 8000:
                    return f"{tool} {fp}\n{out}"
                return f"{tool} {fp}"
    return None


def _iter_records(file_path: Path):
    try:
        with file_path.open("rb") as f:
            chunk = f.read(2048)
    except OSError:
        return
    head = chunk.lstrip()
    is_single_object = head.startswith(b"{") and b"\n  " in head[:200]

    if is_single_object:
        try:
            with file_path.open("rb") as f:
                data = orjson.loads(f.read())
        except (OSError, orjson.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        msgs = data.get("messages")
        if not isinstance(msgs, list):
            return
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if "parts" in m and isinstance(m["parts"], list):
                yield {"_kind": "opencode", "info": m.get("info") or {}, "parts": m["parts"]}
            elif "content" in m and "type" in m:
                yield {"_kind": "gemini", **m}
        return

    try:
        with file_path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    d = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
                if not isinstance(d, dict):
                    continue
                t = d.get("type")
                if t in ("user", "assistant"):
                    yield {"_kind": "claude", **d}
                elif t == "response_item":
                    yield {"_kind": "codex", **d}
    except OSError:
        return


def extract_texts(rec: dict) -> list[str]:
    kind = rec.get("_kind")
    out: list[str] = []
    if kind == "claude":
        msg = rec.get("message")
        if not isinstance(msg, dict):
            return out
        nl = _claude_msg_text(msg, include_tools=False)
        if nl:
            out.append(nl)
        tl = _claude_msg_text(msg, include_tools=True)
        if tl and tl != nl:
            out.append(tl)
        return out
    if kind == "codex":
        pl = rec.get("payload") or {}
        if pl.get("type") == "message":
            content = pl.get("content")
            if isinstance(content, list):
                t = _flatten_parts(content)
                if t:
                    out.append(t)
            elif isinstance(content, str):
                out.append(content)
        return out
    if kind == "opencode":
        parts = rec.get("parts") or []
        chunks: list[str] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            t = _opencode_part_text(p)
            if t:
                chunks.append(t)
        if chunks:
            out.append("\n".join(chunks))
        return out
    if kind == "gemini":
        c = rec.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            chunks2: list[str] = []
            for p in c:
                if isinstance(p, dict):
                    txt = p.get("text")
                    if isinstance(txt, str):
                        chunks2.append(txt)
                elif isinstance(p, str):
                    chunks2.append(p)
            if chunks2:
                out.append("\n".join(chunks2))
        return out
    return out


# ---- Match collection + offset translation --------------------------------


def collect_spans(text: str) -> tuple[list[dict], Counter]:
    """Find all regex matches in `text`, dedupe overlaps (keep longest),
    convert char offsets -> byte offsets, validate round-trip.

    Returns (entities_list, pattern_counter) where each entity has
    {start, end, label: 'secret', _pattern: <name>} (pattern stripped before write).
    """
    raw_spans: list[tuple[int, int, str]] = []  # (char_start, char_end, pattern_name)
    for name, pat in PATTERNS.items():
        for m in pat.finditer(text):
            s, e = m.span()
            if s == e:
                continue
            raw_spans.append((s, e, name))

    if not raw_spans:
        return [], Counter()

    # Dedupe overlaps: keep the longest span; on tie, keep the earliest start.
    raw_spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept: list[tuple[int, int, str]] = []
    for s, e, n in raw_spans:
        overlap_idx = None
        for i, (ks, ke, _) in enumerate(kept):
            if not (e <= ks or s >= ke):
                overlap_idx = i
                break
        if overlap_idx is None:
            kept.append((s, e, n))
        else:
            ks, ke, kn = kept[overlap_idx]
            # keep whichever is longer
            if (e - s) > (ke - ks):
                kept[overlap_idx] = (s, e, n)
            # else keep existing
    kept.sort(key=lambda x: x[0])

    # char -> byte offset using cumulative byte lengths
    encoded = text.encode("utf-8")
    # Build prefix-byte-length array for each char index
    char_to_byte = [0] * (len(text) + 1)
    acc = 0
    for i, ch in enumerate(text):
        char_to_byte[i] = acc
        acc += len(ch.encode("utf-8"))
    char_to_byte[len(text)] = acc
    assert acc == len(encoded), "byte length mismatch"

    counter: Counter = Counter()
    entities: list[dict] = []
    for cs, ce, name in kept:
        bs = char_to_byte[cs]
        be = char_to_byte[ce]
        # Round-trip check
        surface_char = text[cs:ce]
        surface_byte = encoded[bs:be]
        if surface_byte.decode("utf-8", errors="strict") != surface_char:
            # Shouldn't happen but guard anyway
            continue
        if not (0 <= bs < be <= len(encoded)):
            continue
        entities.append({"start": bs, "end": be, "label": "secret", "_pattern": name, "_surface": surface_char})
        counter[name] += 1

    return entities, counter


# ---- Existing text-hash pool ----------------------------------------------


def load_existing_text_hashes(out_dir: Path) -> set[str]:
    seen: set[str] = set()
    if not out_dir.exists():
        return seen
    for p in sorted(out_dir.glob("*.jsonl")):
        # Skip our own batch outputs + audit sidecar so re-runs are idempotent
        if p.name.startswith(SHARD_PREFIX) or p.name == AUDIT_PATH.name:
            continue
        try:
            with p.open("rb") as f:
                for raw in f:
                    if not raw.strip():
                        continue
                    try:
                        rec = orjson.loads(raw)
                    except orjson.JSONDecodeError:
                        continue
                    t = rec.get("text")
                    if isinstance(t, str):
                        seen.add(hashlib.sha256(t.encode("utf-8")).hexdigest())
        except OSError:
            continue
    return seen


# ---- Main ------------------------------------------------------------------


def main() -> int:
    if not SRC_DIR.exists():
        print(f"FATAL: source dir not found: {SRC_DIR}", file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in SRC_DIR.iterdir() if p.suffix == ".jsonl")
    if not files:
        print(f"FATAL: no .jsonl files in {SRC_DIR}", file=sys.stderr)
        return 2

    rng = random.Random(SEED)
    rng.shuffle(files)

    existing_hashes = load_existing_text_hashes(OUT_DIR)
    print(f"loaded {len(existing_hashes)} existing text hashes for dedup", flush=True)

    batch_hashes: set[str] = set()
    collected: list[dict] = []
    pattern_counts: Counter = Counter()
    byte_lengths: list[int] = []
    dup_prior = 0
    dup_batch = 0
    scanned_files = 0
    scanned_turns = 0
    private_key_count = 0
    audit_records: list[dict] = []

    # Shard state
    shard_idx = 0
    shard_written = 0
    shard_path = OUT_DIR / f"{SHARD_PREFIX}_{shard_idx:03d}.jsonl"
    shard_fh = shard_path.open("wb")

    def rotate_shard():
        nonlocal shard_idx, shard_written, shard_path, shard_fh
        shard_fh.close()
        print(
            f"{shard_path.name}: written {PER_SHARD}, total {len(collected)}, scanned {scanned_files}",
            flush=True,
        )
        shard_idx += 1
        shard_written = 0
        shard_path = OUT_DIR / f"{SHARD_PREFIX}_{shard_idx:03d}.jsonl"
        shard_fh = shard_path.open("wb")

    try:
        for fp in files:
            if len(collected) >= MAX_TOTAL:
                break
            scanned_files += 1
            for rec in _iter_records(fp):
                texts = extract_texts(rec)
                if not texts:
                    continue
                for text in texts:
                    scanned_turns += 1
                    nbytes = len(text.encode("utf-8"))
                    if nbytes < MIN_BYTES or nbytes > MAX_BYTES:
                        continue
                    spans, counts = collect_spans(text)
                    if not spans:
                        continue
                    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if h in existing_hashes:
                        dup_prior += 1
                        continue
                    if h in batch_hashes:
                        dup_batch += 1
                        continue
                    batch_hashes.add(h)

                    # Strip internal fields before write; record audit
                    public_entities: list[dict] = []
                    for ent in spans:
                        public_entities.append(
                            {"start": ent["start"], "end": ent["end"], "label": "secret"}
                        )
                        surf = ent["_surface"]
                        # Sanitize surface preview: first 8 + last 4
                        first8 = surf[:8]
                        last4 = surf[-4:] if len(surf) >= 4 else surf
                        audit_records.append(
                            {
                                "file_id": fp.stem,
                                "pattern_type": ent["_pattern"],
                                "surface_first_8": first8,
                                "surface_last_4": last4,
                            }
                        )
                    record = {"text": text, "entities": public_entities}
                    shard_fh.write(orjson.dumps(record) + b"\n")
                    collected.append(record)
                    pattern_counts.update(counts)
                    byte_lengths.append(nbytes)
                    private_key_count += counts.get("private_key_hdr", 0)
                    shard_written += 1
                    if shard_written == PER_SHARD:
                        rotate_shard()
                    if len(collected) >= MAX_TOTAL:
                        break
                if len(collected) >= MAX_TOTAL:
                    break
            if len(collected) >= MAX_TOTAL:
                break
    finally:
        shard_fh.close()
        # If the last opened shard had < PER_SHARD rows, rename it to _partial
        if shard_written > 0 and shard_written < PER_SHARD:
            partial_path = OUT_DIR / f"{SHARD_PREFIX}_partial.jsonl"
            shard_path.replace(partial_path)
            print(
                f"{partial_path.name}: written {shard_written}, total {len(collected)}, scanned {scanned_files}",
                flush=True,
            )
        elif shard_written == 0:
            # Empty shard — drop it
            try:
                shard_path.unlink()
            except OSError:
                pass
        else:
            # Shard hit PER_SHARD exactly and was already announced inside rotate_shard;
            # but if we exited via the break BEFORE rotate (shouldn't happen because
            # rotate runs immediately on hit), do nothing extra.
            pass

    # Write audit sidecar
    with AUDIT_PATH.open("wb") as af:
        for a in audit_records:
            af.write(orjson.dumps(a) + b"\n")

    # ---- Verification ------------------------------------------------------

    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Re-scan every produced shard (exclude audit sidecar)
    shard_files = sorted(
        p
        for p in OUT_DIR.glob(f"{SHARD_PREFIX}_*.jsonl")
        if p.name != AUDIT_PATH.name
    )
    total_lines = 0
    for sp in shard_files:
        with sp.open("rb") as f:
            n = 0
            for raw in f:
                if not raw.strip():
                    continue
                rec = orjson.loads(raw)
                assert isinstance(rec, dict), f"{sp}: not a dict"
                ents = rec.get("entities")
                assert isinstance(ents, list) and ents, f"{sp}: empty entities"
                t = rec["text"]
                enc = t.encode("utf-8")
                for ent in ents:
                    s, e = ent["start"], ent["end"]
                    assert 0 <= s < e <= len(enc), f"{sp}: bad offsets"
                    assert ent["label"] == "secret", f"{sp}: bad label"
                    # round-trip
                    enc[s:e].decode("utf-8")
                n += 1
            if sp.name.endswith("_partial.jsonl"):
                assert n > 0 and n < PER_SHARD, f"{sp}: partial expected 1..{PER_SHARD - 1}, got {n}"
            else:
                assert n == PER_SHARD, f"{sp}: expected {PER_SHARD} lines, got {n}"
            total_lines += n
    print(f"  total rows across {len(shard_files)} shards: {total_lines}")

    # Round-trip 5 random spans
    if collected:
        rt_rng = random.Random(SEED + 1)
        # Build flat list of (text, ent) pairs
        flat = []
        for rec in collected:
            for ent in rec["entities"]:
                flat.append((rec["text"], ent))
        if flat:
            sample = rt_rng.sample(flat, min(5, len(flat)))
            for tt, ent in sample:
                enc = tt.encode("utf-8")
                seg = enc[ent["start"]:ent["end"]].decode("utf-8")
                assert seg, "empty span"
            print(f"  round-trip 5 random spans OK")

    print()
    print("PATTERN HISTOGRAM:")
    for k, v in pattern_counts.most_common():
        print(f"  {k}: {v}")

    print()
    if private_key_count:
        print(f"  *** PRIVATE_KEY_HDR matches: {private_key_count} ***")
        print(f"  *** These are likely REAL private keys leaked in transcripts. ***")
    else:
        print("  private_key_hdr matches: 0")

    print()
    if byte_lengths:
        sorted_lens = sorted(byte_lengths)
        median = sorted_lens[len(sorted_lens) // 2]
        print(f"  median byte length: {median}")
        print(f"  min={sorted_lens[0]}  max={sorted_lens[-1]}  n={len(sorted_lens)}")

    print()
    print(f"  dup vs prior curated: {dup_prior}")
    print(f"  dup within batch:     {dup_batch}")
    print(f"  scanned files: {scanned_files}")
    print(f"  scanned turns: {scanned_turns}")

    # 3 surface samples (audit format)
    if audit_records:
        spl_rng = random.Random(SEED + 2)
        n = min(3, len(audit_records))
        smp = spl_rng.sample(audit_records, n)
        print()
        print("  3 audit surface samples (first8...last4):")
        for a in smp:
            print(f"    [{a['pattern_type']:<20}] {a['surface_first_8']}...{a['surface_last_4']}")

    # One sample line (first record) — written to a local-only file is fine,
    # but per the spec, just print the leading text snippet (not the secret bytes themselves).
    if collected:
        first = collected[0]
        first_text = first["text"]
        first_ent = first["entities"][0]
        enc0 = first_text.encode("utf-8")
        surf0 = enc0[first_ent["start"]:first_ent["end"]].decode("utf-8")
        # Mask the surface for stdout display: keep first 8 + last 4
        masked = (
            surf0[:8] + "..." + surf0[-4:] if len(surf0) >= 12 else "<short>"
        )
        # Use a context snippet that doesn't include the full secret
        before_ctx = first_text[:max(0, first_ent["start"] - 0)][:80]
        # Show only the text before the first entity span to avoid leaking
        prefix_chars = 0
        # find char start corresponding to byte start (approximate by re-decoding)
        # safest: just show first 160 chars of text with the secret bytes redacted
        secret_str = surf0
        redacted_text = first_text.replace(secret_str, "<REDACTED>")
        print()
        print("  SAMPLE LINE (redacted secret):")
        print(f"    text[:300]: {redacted_text[:300]!r}")
        print(f"    masked surface: {masked}")

    print()
    print(
        f"DONE  total={len(collected)}  scanned_files={scanned_files}  "
        f"scanned_turns={scanned_turns}  audit={AUDIT_PATH.name}({len(audit_records)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
