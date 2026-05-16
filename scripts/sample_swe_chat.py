#!/usr/bin/env -S uv run --with orjson python
"""Curate SALT-NLP/SWE-chat transcripts into clean negative (entities=[]) shards.

This is the third batch of LoRA training data for openai/privacy-filter. Earlier
shards (nemotron_*, gretel_*) cover positive PII examples in document/prose
register. This batch teaches the model the *agent-transcript register* — CLI,
code blocks, and conversational prose taken from real coding-agent sessions —
all labeled empty so it stops false-positiving on transcript artifacts.

Source: /Volumes/P4510/.cache/datasets/SWE-chat/transcripts/*.jsonl
        Each file = one session. Many top-level types (progress, file-history-
        snapshot, system, attachment, queue-operation, event_msg, etc.) — we
        only care about turns carrying real natural-language/code text:

        * Claude-Code style: {"type": "user"|"assistant", "message": {
              "role": ..., "content": str | [parts]}}
            parts: {type: text, text}, {type: thinking, thinking}, ...
            (skip tool_use/tool_result — schema noisy, often contains paths)
        * Codex style:        {"type": "response_item", "payload": {
              "type": "message", "role": ..., "content": [parts]}}
            parts: {type: input_text|output_text, text}

Filtering (all turns rejected if any rule fires):
    1. 100 <= len(utf8) <= 4000
    2. No Presidio placeholder (<[A-Z][A-Z_]+>) — these are artifacts
    3. No leaked secret patterns (TruffleHog backstop)
    4. No bare email / loose phone
    5. Non-empty after strip
    6. <= 3 turns per source file (diversity)

Output: 5 shards of 100 lines each => 500 examples to
/Volumes/P4510/.cache/datasets/curated/swe_chat_{000..004}.jsonl
Schema per line: {"text": "<str>", "entities": []}

Deterministic with seed=42.
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

import orjson

# ---- Config ----------------------------------------------------------------

SEED = 42
SRC_DIR = Path("/Volumes/P4510/.cache/datasets/SWE-chat/transcripts")
OUT_DIR = Path("/Volumes/P4510/.cache/datasets/curated")
SHARD_PREFIX = "swe_chat"
SHARDS = 5
PER_SHARD = 100
TOTAL = SHARDS * PER_SHARD  # 500

MIN_BYTES = 100
MAX_BYTES = 4000
MAX_TURNS_PER_FILE = 2

# ---- Filters ---------------------------------------------------------------

# Presidio inserts placeholder labels like <PERSON>, <EMAIL_ADDRESS>, ...
# Require >=2 uppercase chars so we don't false-flag generics like <T> or HTML <DIV>
# isn't matched because <DIV> would be 3 chars but isn't a Presidio label anyway.
# Note: HTML tags are usually lowercase; uppercase HTML is rare in transcripts.
PRESIDIO_RE = re.compile(r"<[A-Z][A-Z_]+>")

SECRET_RES = [
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"gho_[A-Za-z0-9]{36}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer [A-Za-z0-9._\-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{20,}\.eyJ[A-Za-z0-9_\-]{20,}"),
]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b\+?\d[\d \-]{8,}\d\b")

# ---- Register heuristics ---------------------------------------------------

CLI_PREFIXES = (
    "$ ", "> ", "pnpm", "npm ", "yarn", "git ", "curl", "kubectl", "docker",
    "aws ", "python -m", "python3 -m", "uv ", "uvx ", "cargo ", "go ", "rustc",
    "make", "cmake", "bash ", "sh ", "node ", "deno ", "bun ", "pip ", "brew ",
    "ssh ", "scp ", "ls ", "cd ", "rm ", "mv ", "cp ", "mkdir", "touch ",
    "echo ", "cat ", "grep ", "sed ", "awk ", "find ", "tar ", "wget", "ps ",
)


def classify_register(text: str) -> str:
    """One of: 'cli', 'code', 'prose'. Cheap heuristics."""
    fence_blocks = text.count("```")
    # Count CLI-flavored lines
    cli_lines = 0
    total_lines = 0
    for ln in text.splitlines():
        stripped = ln.lstrip()
        if not stripped:
            continue
        total_lines += 1
        if stripped.startswith(CLI_PREFIXES):
            cli_lines += 1
    cli_ratio = cli_lines / total_lines if total_lines else 0.0

    if fence_blocks >= 2:
        return "code"
    if cli_ratio >= 0.25 or cli_lines >= 4:
        return "cli"
    # Fenced single block opener with code body counts as code too
    if fence_blocks >= 1 and total_lines > 3:
        return "code"
    return "prose"


# ---- Turn extraction -------------------------------------------------------


def _flatten_parts(parts) -> str:
    """Concatenate text-bearing parts. Yields ONE concatenated string.

    Supported part types:
        Claude-Code: text, thinking
        Codex:       input_text, output_text
        OpenCode:    text, reasoning
        Tool-result content lists (Claude): {type:text, text}
    """
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
    """Render a Claude-Code tool_use block as a CLI-flavored line.

    Captures: Bash command (and description), file paths from Read/Edit/Write,
    Grep pattern. Skips noisy structured params.
    """
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
    """Render a Claude-Code tool_result block as plain text."""
    c = part.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return _flatten_parts(c) or None
    return None


def _claude_msg_text(msg: dict, *, include_tools: bool) -> str | None:
    """Flatten a Claude-Code message.content into a single text blob."""
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
    """Render one OpenCode message part."""
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
        # OpenCode 'bash' tool
        if tool == "bash" and isinstance(inp, dict):
            cmd = inp.get("command")
            if isinstance(cmd, str):
                if isinstance(out, str) and out.strip():
                    return f"$ {cmd}\n{out}"
                return f"$ {cmd}"
        # file tools
        if tool in ("read", "write", "edit") and isinstance(inp, dict):
            fp = inp.get("filePath") or inp.get("file_path") or inp.get("path")
            if isinstance(fp, str):
                if isinstance(out, str) and out.strip() and len(out) < 4000:
                    return f"{tool} {fp}\n{out}"
                return f"{tool} {fp}"
    return None


def _iter_records(file_path: Path):
    """Yield turn dicts from any of the four transcript formats.

    Detects up front whether the file is JSONL or a single JSON object.
    Yields normalized records of the form:
        {"_kind": "claude"|"codex"|"opencode"|"gemini", ...payload}
    """
    # Cheap detection: peek at first non-empty byte
    try:
        with file_path.open("rb") as f:
            chunk = f.read(2048)
    except OSError:
        return
    head = chunk.lstrip()
    # If the first non-whitespace chars are { followed by newline+indent, it's
    # likely a pretty-printed single object. JSONL has one full object per line.
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
        # OpenCode: each msg has {info: {role}, parts: [...]}
        # Gemini:   each msg has {type, content}
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if "parts" in m and isinstance(m["parts"], list):
                yield {"_kind": "opencode", "info": m.get("info") or {}, "parts": m["parts"]}
            elif "content" in m and "type" in m:
                yield {"_kind": "gemini", **m}
        return

    # JSONL: one record per line, mostly Claude-Code, occasionally Codex
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
                # else: progress, system, file-history-snapshot, etc. — skip
    except OSError:
        return


def extract_texts(rec: dict) -> list[str]:
    """Return a LIST of candidate text strings from a normalized record.

    A single record may produce multiple candidates — e.g. a Claude assistant
    turn that has both natural-language text and shell commands can yield two
    independent samples to balance register variety.
    """
    kind = rec.get("_kind")
    out: list[str] = []
    if kind == "claude":
        msg = rec.get("message")
        if not isinstance(msg, dict):
            return out
        # Natural-language pass (text + thinking only)
        nl = _claude_msg_text(msg, include_tools=False)
        if nl:
            out.append(nl)
        # Tool pass (commands + paths + tool outputs)
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
        # Group into a single text blob
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
            chunks: list[str] = []
            for p in c:
                if isinstance(p, dict):
                    txt = p.get("text")
                    if isinstance(txt, str):
                        chunks.append(txt)
                elif isinstance(p, str):
                    chunks.append(p)
            if chunks:
                out.append("\n".join(chunks))
        return out
    return out


# ---- Filter pipeline -------------------------------------------------------


def filter_text(text: str, stats: Counter) -> bool:
    """Return True if the turn passes all filters."""
    if text is None:
        stats["null"] += 1
        return False
    t = text.strip()
    if not t:
        stats["empty_after_strip"] += 1
        return False
    nbytes = len(text.encode("utf-8"))
    if nbytes < MIN_BYTES:
        stats["too_short"] += 1
        return False
    if nbytes > MAX_BYTES:
        stats["too_long"] += 1
        return False
    if PRESIDIO_RE.search(text):
        stats["presidio_placeholder"] += 1
        return False
    for r in SECRET_RES:
        if r.search(text):
            stats["leaked_secret"] += 1
            return False
    if EMAIL_RE.search(text):
        stats["bare_email"] += 1
        return False
    if PHONE_RE.search(text):
        stats["bare_phone"] += 1
        return False
    return True


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

    stats: Counter = Counter()
    register_counts: Counter = Counter()
    byte_lengths: list[int] = []
    used_source_files: set[str] = set()

    collected: list[dict] = []
    scanned_files = 0
    scanned_turns = 0

    shard_idx = 0
    shard_written = 0
    # Open first shard
    out_path = OUT_DIR / f"{SHARD_PREFIX}_{shard_idx:03d}.jsonl"
    out_fh = out_path.open("wb")

    def close_and_advance(idx: int):
        nonlocal out_fh, out_path
        out_fh.close()
        print(
            f"{out_path.name}: written {PER_SHARD}, "
            f"total {len(collected)}, scanned_so_far {scanned_files}",
            flush=True,
        )
        next_path = OUT_DIR / f"{SHARD_PREFIX}_{idx:03d}.jsonl"
        out_fh_new = next_path.open("wb")
        return next_path, out_fh_new

    # Target register caps so we don't drown in 92% prose like the first run.
    # Soft caps; once a register is full we keep scanning but skip its samples
    # unless we're starved at the end.
    register_caps = {"cli": 175, "code": 175, "prose": 200}

    def try_accept(text: str, fp_name: str, file_taken: int):
        """Validate + classify + write one sample. Returns updated file_taken."""
        nonlocal shard_written, shard_idx, out_path, out_fh
        if file_taken >= MAX_TURNS_PER_FILE:
            return file_taken, False
        if not filter_text(text, stats):
            return file_taken, False
        reg = classify_register(text)
        # Soft cap unless we're running short overall
        remaining_overall = TOTAL - len(collected)
        if register_counts[reg] >= register_caps[reg] and remaining_overall > 50:
            stats[f"reg_cap_{reg}"] += 1
            return file_taken, False
        record = {"text": text, "entities": []}
        out_fh.write(orjson.dumps(record) + b"\n")
        collected.append(record)
        used_source_files.add(fp_name)
        byte_lengths.append(len(text.encode("utf-8")))
        register_counts[reg] += 1
        shard_written += 1
        if shard_written == PER_SHARD:
            shard_idx += 1
            if shard_idx < SHARDS:
                out_path, out_fh = close_and_advance(shard_idx)
                shard_written = 0
            else:
                out_fh.close()
                print(
                    f"{out_path.name}: written {PER_SHARD}, "
                    f"total {len(collected)}, "
                    f"scanned_so_far {scanned_files}",
                    flush=True,
                )
                out_fh = None
        return file_taken + 1, True

    try:
        for fp in files:
            if len(collected) >= TOTAL or out_fh is None:
                break
            scanned_files += 1
            file_taken = 0
            for rec in _iter_records(fp):
                if file_taken >= MAX_TURNS_PER_FILE:
                    break
                texts = extract_texts(rec)
                if not texts:
                    continue
                scanned_turns += 1
                for text in texts:
                    if file_taken >= MAX_TURNS_PER_FILE:
                        break
                    file_taken, _ = try_accept(text, fp.name, file_taken)
                    if len(collected) >= TOTAL:
                        break
                if len(collected) >= TOTAL:
                    break
    finally:
        if out_fh is not None and not out_fh.closed:
            out_fh.close()

    if len(collected) < TOTAL:
        print(
            f"FATAL: only collected {len(collected)} / {TOTAL} after scanning "
            f"{scanned_files} files. Stats: {dict(stats)}",
            file=sys.stderr,
        )
        return 3

    # ---- Verification ------------------------------------------------------

    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    total_lines = 0
    for i in range(SHARDS):
        sp = OUT_DIR / f"{SHARD_PREFIX}_{i:03d}.jsonl"
        with sp.open("rb") as f:
            n = 0
            for raw in f:
                if not raw.strip():
                    continue
                rec = orjson.loads(raw)
                assert isinstance(rec, dict), f"{sp}: not a dict"
                assert rec.get("entities") == [], f"{sp}: entities must be []"
                assert isinstance(rec.get("text"), str) and rec["text"].strip(), (
                    f"{sp}: empty text"
                )
                n += 1
            assert n == PER_SHARD, f"{sp}: expected {PER_SHARD} lines, got {n}"
            total_lines += n
    print(f"  shards: {SHARDS} x {PER_SHARD} = {total_lines} lines OK")

    # Spot-check 10 random lines for placeholders/secrets that slipped through
    spot_rng = random.Random(SEED + 1)
    spot_sample = spot_rng.sample(collected, 10)
    for rec in spot_sample:
        t = rec["text"]
        assert not PRESIDIO_RE.search(t), "PRESIDIO leak"
        assert not EMAIL_RE.search(t), "email leak"
        for r in SECRET_RES:
            assert not r.search(t), "secret leak"
    print("  spot check 10 random: no placeholders / secrets / emails leaked")

    # Length stats
    byte_lengths_sorted = sorted(byte_lengths)
    n = len(byte_lengths_sorted)

    def pct(p):
        return byte_lengths_sorted[max(0, min(n - 1, int(p * n)))]

    print(
        f"  byte length: p10={pct(0.10)}  median={pct(0.50)}  p90={pct(0.90)}  "
        f"min={byte_lengths_sorted[0]}  max={byte_lengths_sorted[-1]}"
    )
    print(f"  unique source transcripts used: {len(used_source_files)}")
    print(f"  register mix:")
    total = sum(register_counts.values())
    for k in ("cli", "code", "prose"):
        c = register_counts.get(k, 0)
        print(f"    {k:<6} {c:4d}  {100 * c / total:5.1f}%")

    print()
    print("FILTER REJECTIONS (top 5):")
    for k, v in stats.most_common(5):
        print(f"  {k}: {v}")

    print()
    print("SAMPLE LINE (first record):")
    sample_text = collected[0]["text"]
    print(f"  text[:300]: {sample_text[:300]!r}")

    print()
    print(
        f"DONE  total={len(collected)}  "
        f"scanned_files={scanned_files}  scanned_turns={scanned_turns}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
