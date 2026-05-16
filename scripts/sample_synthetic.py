#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Synthesize positive privacy-filter training examples in CLI / transcript register.

Uses `claude -p --model sonnet --output-format json` headless mode to drive
Anthropic's Sonnet through 1200 one-shot generation prompts. Each prompt asks
for a single JSON object describing one synthetic CLI command or one short LLM
transcript fragment with embedded PII / secrets. Responses are validated,
byte-offset-mapped, deduped, and written into per-batch staging files that are
finally sharded into 100-row JSONL files matching the project schema.

Run:
    uv run /Users/brooklyn/workspace/github/mlx-node/scripts/sample_synthetic.py

The script is deterministic (seed=45) and resumable: each accepted example is
fsync'd to the staging file, so re-running picks up from the existing count.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"
MODEL = "deepseek-chat"
SEED = 45
PARALLEL = 8
CALL_TIMEOUT = 60  # seconds

# Per-MTok pricing for cost estimate (USD). DeepSeek deepseek-chat (V4-Flash) as of 2026-05.
DEEPSEEK_INPUT_USD_PER_MTOK = 0.27
DEEPSEEK_OUTPUT_USD_PER_MTOK = 1.10

OUT_DIR = Path("/Volumes/P4510/.cache/datasets/curated")
CLI_STAGE = OUT_DIR / "synthetic_cli_stage.jsonl"
TRANSCRIPT_STAGE = OUT_DIR / "synthetic_transcript_stage.jsonl"

CLI_TARGET = 800
CLI_FILES = 8  # 100 rows each
TRANSCRIPT_TARGET = 400
TRANSCRIPT_FILES = 4

# Stop after this many _attempts_ (calls) per batch even if target not hit.
CLI_MAX_CALLS = 1600
TRANSCRIPT_MAX_CALLS = 800

# Canonical privacy-filter labels.
CANONICAL_LABELS = {
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
}

# Map model-emitted label synonyms onto canonical labels.
LABEL_SYNONYMS = {
    # all secret-y stuff collapses to "secret"
    "api_key": "secret",
    "api-key": "secret",
    "apikey": "secret",
    "token": "secret",
    "password": "secret",
    "passwd": "secret",
    "jwt": "secret",
    "bearer": "secret",
    "bearer_token": "secret",
    "github_pat": "secret",
    "github_token": "secret",
    "openai_key": "secret",
    "openai_api_key": "secret",
    "anthropic_key": "secret",
    "anthropic_api_key": "secret",
    "aws_secret": "secret",
    "aws_secret_access_key": "secret",
    "hf_token": "secret",
    "huggingface_token": "secret",
    "slack_token": "secret",
    "cloudflare_api_token": "secret",
    "cloudflare_token": "secret",
    "cf_api_token": "secret",
    "private_key": "secret",
    "ssh_key": "secret",
    "session_token": "secret",
    "auth_token": "secret",
    "access_token": "secret",
    "refresh_token": "secret",
    "client_secret": "secret",
    "secret_key": "secret",
    "credential": "secret",
    "credentials": "secret",
    "key": "secret",
    "aws_access_key_id": "secret",
    "access_key_id": "secret",
    # ids / account-y
    "account_id": "account_number",
    "cloudflare_account_id": "account_number",
    "aws_account_id": "account_number",
    "uid": "account_number",
    "user_id": "account_number",
    "id": "account_number",
    "account": "account_number",
    # email
    "email": "private_email",
    "email_address": "private_email",
    # url
    "url": "private_url",
    "link": "private_url",
    # name
    "name": "private_person",
    "person": "private_person",
    "person_name": "private_person",
    "username": "private_person",
    "user": "private_person",
    # date
    "date": "private_date",
    "datetime": "private_date",
    # phone
    "phone": "private_phone",
    "phone_number": "private_phone",
    "telephone": "private_phone",
    # address
    "address": "private_address",
    "street": "private_address",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CLI_PROMPT = """You generate ONE realistic shell command for a training dataset.

Output EXACTLY this JSON object on a single line (no prose, no markdown fence, no surrounding whitespace):

{{"text": "<command>", "entities": [{{"entity_string": "<EXACT substring>", "label": "<label>"}}, ...]}}

Rules:
- text MUST be a real, copy-pasteable shell command. Single line.
- Use utility: {utility}
- Include at least one of: {secret_types_csv}
- Include 1-3 PII spans total. Vary token shapes:
  - GitHub PAT: ghp_ followed by 36 alphanumeric chars
  - GitHub fine-grained PAT: github_pat_ followed by 22 alnum + _ + 59 alnum
  - OpenAI key: sk- followed by 40-48 alnum chars
  - OpenAI project key: sk-proj- followed by 48 alnum + _-
  - Anthropic: sk-ant-api03- followed by ~95 chars
  - AWS access key ID: AKIA + 16 uppercase alnum
  - AWS secret: 40 base64 chars
  - HF token: hf_ + 34 alnum chars
  - Cloudflare API token: 40 alnum chars
  - JWT: three base64url segments joined by dots
  - Bearer token: arbitrary 20-50 alnum string
  - Slack: xoxb- or xoxp- followed by 50+ chars with dashes
- Realistic flag-arg patterns. Use a MIX across the batch:
  --token VALUE     --token=VALUE     -t VALUE
  --api-key VALUE   --apikey=VALUE
  --password VALUE  -p VALUE
  -H "Authorization: Bearer VALUE"
  -H "X-API-Key: VALUE"
  -u user:VALUE
  FOO_TOKEN=VALUE   (env var assignment)
  export FOO_TOKEN=VALUE && ...
- Labels (use EXACTLY these strings):
  - All secrets/keys/tokens/passwords/JWTs/API keys -> "secret"
  - Email addresses -> "private_email"
  - URLs that include identifiers -> "private_url"
  - Cloudflare account hex IDs, AWS account IDs, IDs in URL paths -> "account_number"
- entity_string MUST be the EXACT substring as it appears in text (case + chars). I will compute offsets downstream by str.index - do NOT output offsets, do NOT add or remove characters.
- ONE entity per span. No nesting, no overlap.

Output the JSON object only, nothing else."""

TRANSCRIPT_PROMPT = """You generate ONE realistic LLM-agent transcript fragment for a training dataset.

Output EXACTLY this JSON object on a single line:

{{"text": "<fragment>", "entities": [{{"entity_string": "<EXACT substring>", "label": "<label>"}}, ...]}}

Rules:
- text MUST be a 200-1500 character fragment of a realistic agent transcript. Use this register: {register}
- Include 1-4 PII/secret spans. Mix in secrets from this set: {secret_types_csv}
- Maybe also: emails, GitHub URLs containing usernames, account IDs
- Realistic context. Patterns to include:
  - "TOKEN=ghp_XXX" lines in .env dumps
  - "Authorization: Bearer eyJ..." in HTTP request examples
  - "Set ANTHROPIC_API_KEY=sk-ant-... and run..." in instructions
  - Tool-result blocks like: "<tool_result>\\nGITHUB_TOKEN=ghp_XXX\\nrunning command..."
  - Code-block secrets: "```python\\nclient = OpenAI(api_key='sk-...')\\n```"
- Labels (use EXACTLY these strings):
  - All secrets/keys/tokens/passwords/JWTs/API keys -> "secret"
  - Email addresses -> "private_email"
  - URLs that include identifiers -> "private_url"
  - Cloudflare account hex IDs, AWS account IDs, IDs in URL paths -> "account_number"
  - Personal names -> "private_person"
- entity_string MUST be EXACT substring. No offsets. NO ESCAPING in entity_string beyond what JSON requires - it must match text.index(entity_string).

Output the JSON object only."""

# ---------------------------------------------------------------------------
# Variation tables
# ---------------------------------------------------------------------------

CLI_UTILITIES = [
    "curl", "wget", "pnpm", "npm", "yarn", "pip", "uv", "gh", "aws", "kubectl",
    "docker", "git", "ssh", "helm", "terraform", "gcloud", "az", "wrangler",
    "vercel", "railway",
]

SECRET_KINDS = [
    "GitHub PAT (ghp_...)",
    "GitHub fine-grained PAT (github_pat_...)",
    "OpenAI key (sk-...)",
    "OpenAI project key (sk-proj-...)",
    "Anthropic API key (sk-ant-api03-...)",
    "AWS access key ID (AKIA...)",
    "AWS secret access key (40-char base64)",
    "HuggingFace token (hf_...)",
    "Cloudflare API token",
    "JWT (eyJ...)",
    "generic Bearer token",
    "Slack token (xoxb-... or xoxp-...)",
]

TRANSCRIPT_REGISTERS = [
    'Tool result dump: agent shows the output of running a command, with env vars/tokens visible',
    'User asking for help: "I\'m getting a 401 from {API}, here\'s my config..." with secrets pasted in',
    'Agent showing config: ".env file content" or "here\'s how to set this up: export TOKEN=..."',
    'Fenced code block in agent response (markdown ``` ... ```) with secrets inline',
]

# ---------------------------------------------------------------------------
# Variation plan (deterministic)
# ---------------------------------------------------------------------------


@dataclass
class CallSpec:
    kind: str            # "cli" or "transcript"
    idx: int             # ordinal index in the batch
    utility: str | None  # CLI only
    register: str | None  # transcript only
    secret_kinds: list[str]  # 1-2 kinds to mention
    nonce: int           # random per-call salt to discourage repeats


def build_cli_specs(n: int, rng: random.Random) -> list[CallSpec]:
    specs: list[CallSpec] = []
    for i in range(n):
        utility = CLI_UTILITIES[i % len(CLI_UTILITIES)]
        k = rng.choice([1, 2])
        secret_kinds = rng.sample(SECRET_KINDS, k=k)
        specs.append(CallSpec(
            kind="cli", idx=i, utility=utility, register=None,
            secret_kinds=secret_kinds, nonce=rng.randint(0, 10**9),
        ))
    return specs


def build_transcript_specs(n: int, rng: random.Random) -> list[CallSpec]:
    specs: list[CallSpec] = []
    for i in range(n):
        register = TRANSCRIPT_REGISTERS[i % len(TRANSCRIPT_REGISTERS)]
        k = rng.choice([2, 3])
        secret_kinds = rng.sample(SECRET_KINDS, k=k)
        specs.append(CallSpec(
            kind="transcript", idx=i, utility=None, register=register,
            secret_kinds=secret_kinds, nonce=rng.randint(0, 10**9),
        ))
    return specs


def render_prompt(spec: CallSpec) -> str:
    secret_types_csv = ", ".join(spec.secret_kinds)
    # The nonce is appended as a comment to encourage variation without
    # actually being mentioned in the output text. Models tend to vary outputs
    # when the prompt differs even slightly.
    suffix = f"\n\n(internal variation salt #{spec.nonce}; do not include this in your output)"
    if spec.kind == "cli":
        return CLI_PROMPT.format(
            utility=spec.utility, secret_types_csv=secret_types_csv,
        ) + suffix
    else:
        return TRANSCRIPT_PROMPT.format(
            register=spec.register, secret_types_csv=secret_types_csv,
        ) + suffix


# ---------------------------------------------------------------------------
# claude -p invocation
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    spec: CallSpec
    raw_text: str | None
    cost_usd: float
    error: str | None = None


def invoke_llm(prompt: str) -> tuple[str, float, str | None]:
    """Call DeepSeek's OpenAI-compatible chat completions endpoint once.
    Returns (result_text, cost_usd_estimate, error)."""
    api_key = os.environ.get(DEEPSEEK_KEY_ENV)
    if not api_key:
        return "", 0.0, f"missing_env:{DEEPSEEK_KEY_ENV}"

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 1024,
        "temperature": 1.0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=CALL_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        snippet = exc.read().decode("utf-8", errors="replace")[:200]
        return "", 0.0, f"http_{exc.code}:{snippet}"
    except urllib.error.URLError as exc:
        return "", 0.0, f"url_error:{exc.reason!r}"
    except TimeoutError:
        return "", 0.0, "timeout"
    except Exception as exc:  # noqa: BLE001
        return "", 0.0, f"transport_error:{exc!r}"

    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return "", 0.0, "envelope_not_json"

    choices = envelope.get("choices") or []
    if not choices:
        return "", 0.0, "no_choices"
    content = (choices[0].get("message") or {}).get("content") or ""

    usage = envelope.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    cost = (
        prompt_tokens * (DEEPSEEK_INPUT_USD_PER_MTOK / 1_000_000)
        + completion_tokens * (DEEPSEEK_OUTPUT_USD_PER_MTOK / 1_000_000)
    )
    return content, cost, None


# ---------------------------------------------------------------------------
# Response parsing & validation
# ---------------------------------------------------------------------------

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_obj(raw: str) -> dict | None:
    """Try strict json.loads, then a permissive greedy fallback."""
    raw = raw.strip()
    if not raw:
        return None
    # Strip code-fence wrappers if any.
    if raw.startswith("```"):
        # remove ```json\n ... \n``` or ``` ... ```
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Greedy fallback: first '{' to last '}'.
    m = JSON_OBJECT_RE.search(raw)
    if m is None:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def normalize_label(label: str) -> str | None:
    if not isinstance(label, str):
        return None
    key = label.strip().lower().replace("-", "_").replace(" ", "_")
    if key in CANONICAL_LABELS:
        return key
    return LABEL_SYNONYMS.get(key)


@dataclass
class RowDropStats:
    counts: Counter = field(default_factory=Counter)

    def incr(self, reason: str) -> None:
        self.counts[reason] += 1

    def top(self, n: int = 5) -> list[tuple[str, int]]:
        return self.counts.most_common(n)


def build_row(obj: dict, drops: RowDropStats) -> dict | None:
    """Validate + offset-map. Returns the final {text, entities} row or None."""
    if not isinstance(obj, dict):
        drops.incr("obj_not_dict")
        return None
    text = obj.get("text")
    ents = obj.get("entities")
    if not isinstance(text, str) or not text:
        drops.incr("text_missing_or_empty")
        return None
    if not isinstance(ents, list):
        drops.incr("entities_not_list")
        return None

    text_bytes = text.encode("utf-8")
    spans: list[tuple[int, int, str]] = []  # (byte_start, byte_end, label)

    for ent in ents:
        if not isinstance(ent, dict):
            drops.incr("ent_not_dict")
            continue
        es = ent.get("entity_string")
        lbl_raw = ent.get("label")
        if not isinstance(es, str) or not es:
            drops.incr("ent_string_missing")
            continue
        lbl = normalize_label(lbl_raw)
        if lbl is None:
            drops.incr("bad_label")
            continue
        try:
            char_start = text.index(es)
        except ValueError:
            drops.incr("hallucinated_substring")
            continue
        byte_start = len(text[:char_start].encode("utf-8"))
        byte_end = byte_start + len(es.encode("utf-8"))
        if not (0 <= byte_start < byte_end <= len(text_bytes)):
            drops.incr("byte_offset_oob")
            continue
        if text_bytes[byte_start:byte_end].decode("utf-8", errors="replace") != es:
            drops.incr("byte_roundtrip_mismatch")
            continue
        spans.append((byte_start, byte_end, lbl))

    if not spans:
        drops.incr("retry_no_entity")
        return None

    # Sort, dedupe overlaps (drop the later one to keep sort stable).
    spans.sort(key=lambda s: (s[0], s[1]))
    kept: list[tuple[int, int, str]] = []
    last_end = -1
    for s in spans:
        if s[0] < last_end:
            drops.incr("overlap")
            continue
        kept.append(s)
        last_end = s[1]
    if not kept:
        drops.incr("all_overlapped")
        return None

    return {
        "text": text,
        "entities": [{"start": s, "end": e, "label": lbl} for s, e, lbl in kept],
    }


# ---------------------------------------------------------------------------
# Staging IO with dedupe
# ---------------------------------------------------------------------------


def load_existing(stage_path: Path) -> tuple[list[dict], set[str]]:
    """Read existing accepted rows from a staging file (for resume)."""
    rows: list[dict] = []
    hashes: set[str] = set()
    if not stage_path.exists():
        return rows, hashes
    with stage_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if not isinstance(text, str):
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in hashes:
                continue
            rows.append(row)
            hashes.add(h)
    return rows, hashes


def append_row(stage_path: Path, row: dict) -> None:
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with stage_path.open("ab") as fh:
        fh.write(line.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_batch(
    *,
    label: str,
    spec_builder,
    target: int,
    max_calls: int,
    stage_path: Path,
    rng: random.Random,
) -> tuple[list[dict], int, float, RowDropStats]:
    """Run one batch until target accepted rows or max_calls attempts."""

    rows, hashes = load_existing(stage_path)
    if len(rows) >= target:
        print(f"[{label}] already at target ({len(rows)}/{target}); skipping generation.")
        return rows[:target], 0, 0.0, RowDropStats()

    drops = RowDropStats()
    total_cost = 0.0
    calls_made = 0

    print(f"[{label}] resuming from {len(rows)} accepted rows, target {target}, max_calls {max_calls}")

    # Pre-build a large pool of specs we can pop from; we'll request more if we
    # run out. The pool is just an index counter — variation is per-call random
    # against the local rng (kept deterministic by seeding outside).
    spec_idx = 0

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        in_flight: dict = {}  # future -> spec

        # Keep the pool topped up.
        def submit_next():
            nonlocal spec_idx, calls_made
            if calls_made >= max_calls:
                return
            spec = spec_builder(spec_idx, rng)
            spec_idx += 1
            calls_made += 1
            prompt = render_prompt(spec)
            fut = pool.submit(invoke_llm, prompt)
            in_flight[fut] = spec

        # Prime
        for _ in range(PARALLEL):
            if len(rows) + len(in_flight) >= target:
                break
            submit_next()

        while in_flight and len(rows) < target:
            done_iter = as_completed(in_flight)
            try:
                fut = next(done_iter)
            except StopIteration:
                break
            spec = in_flight.pop(fut)

            try:
                result_text, cost_usd, err = fut.result()
            except Exception as exc:  # noqa: BLE001
                drops.incr("future_exception")
                # top up
                if len(rows) + len(in_flight) < target and calls_made < max_calls:
                    submit_next()
                continue

            total_cost += cost_usd

            if err is not None:
                drops.incr(f"call_err:{err.split(':')[0]}")
                if len(rows) + len(in_flight) < target and calls_made < max_calls:
                    submit_next()
                continue

            obj = extract_json_obj(result_text)
            if obj is None:
                drops.incr("response_not_json")
                if len(rows) + len(in_flight) < target and calls_made < max_calls:
                    submit_next()
                continue

            row = build_row(obj, drops)
            if row is None:
                if len(rows) + len(in_flight) < target and calls_made < max_calls:
                    submit_next()
                continue

            text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            if text_hash in hashes:
                drops.incr("duplicate_text")
                if len(rows) + len(in_flight) < target and calls_made < max_calls:
                    submit_next()
                continue
            hashes.add(text_hash)
            rows.append(row)
            append_row(stage_path, row)

            if len(rows) % 50 == 0:
                top = drops.top(1)
                top_str = f"{top[0][0]}={top[0][1]}" if top else "-"
                print(
                    f"[{label}] {len(rows)}/{target} accepted, "
                    f"{sum(drops.counts.values())} dropped, "
                    f"{calls_made} calls, ${total_cost:.2f}, "
                    f"top drop: {top_str}",
                    flush=True,
                )

            if len(rows) + len(in_flight) < target and calls_made < max_calls:
                submit_next()

        # Drain remaining in_flight in case we overshot or hit max_calls.
        for fut in in_flight:
            try:
                _ = fut.result(timeout=CALL_TIMEOUT)
            except Exception:  # noqa: BLE001
                pass

    return rows[:target], calls_made, total_cost, drops


# ---------------------------------------------------------------------------
# Final shard split + verification
# ---------------------------------------------------------------------------


def shard_to_files(rows: list[dict], prefix: str, n_files: int) -> list[Path]:
    rows_per_file = len(rows) // n_files
    out_paths: list[Path] = []
    for i in range(n_files):
        start = i * rows_per_file
        end = start + rows_per_file
        chunk = rows[start:end]
        path = OUT_DIR / f"{prefix}_{i:03d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in chunk:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_paths.append(path)
    return out_paths


def label_histogram(rows: list[dict]) -> Counter:
    c: Counter = Counter()
    for row in rows:
        for ent in row["entities"]:
            c[ent["label"]] += 1
    return c


def verify_outputs(prefix: str, n_files: int, rows_per_file: int, rng: random.Random) -> None:
    print(f"\n=== Verification: {prefix} ===")
    all_rows: list[dict] = []
    for i in range(n_files):
        path = OUT_DIR / f"{prefix}_{i:03d}.jsonl"
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) == rows_per_file, (
            f"{path} has {len(lines)} lines, expected {rows_per_file}"
        )
        for line in lines:
            row = json.loads(line)
            assert len(row["entities"]) >= 1, f"empty entities in {path}"
            all_rows.append(row)
    print(f"  files: {n_files}, rows/file: {rows_per_file}, total: {len(all_rows)}")

    hist = label_histogram(all_rows)
    print("  label histogram:")
    for lbl, cnt in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"    {lbl}: {cnt}")

    # Round-trip 10 random spans.
    print("  round-trip check (10 random spans):")
    ok = 0
    for _ in range(10):
        row = rng.choice(all_rows)
        if not row["entities"]:
            continue
        ent = rng.choice(row["entities"])
        text_bytes = row["text"].encode("utf-8")
        span = text_bytes[ent["start"]:ent["end"]]
        try:
            decoded = span.decode("utf-8")
            ok += 1
        except UnicodeDecodeError:
            decoded = "<invalid-utf8>"
        # print just one per call for brevity
    print(f"    {ok}/10 spans decode to valid UTF-8")

    # Sample 5 example lines.
    print(f"  sample rows ({prefix}):")
    samples = rng.sample(all_rows, k=min(5, len(all_rows)))
    for s in samples:
        text_preview = s["text"][:200].replace("\n", "\\n")
        print(f"    text[:200]={text_preview!r}")
        print(f"    entities={s['entities']}")


# ---------------------------------------------------------------------------
# Spec builders (callable from inside run_batch)
# ---------------------------------------------------------------------------


def cli_spec_builder(idx: int, rng: random.Random) -> CallSpec:
    utility = CLI_UTILITIES[idx % len(CLI_UTILITIES)]
    k = rng.choice([1, 2])
    secret_kinds = rng.sample(SECRET_KINDS, k=k)
    return CallSpec(
        kind="cli", idx=idx, utility=utility, register=None,
        secret_kinds=secret_kinds, nonce=rng.randint(0, 10**9),
    )


def transcript_spec_builder(idx: int, rng: random.Random) -> CallSpec:
    register = TRANSCRIPT_REGISTERS[idx % len(TRANSCRIPT_REGISTERS)]
    k = rng.choice([2, 3])
    secret_kinds = rng.sample(SECRET_KINDS, k=k)
    return CallSpec(
        kind="transcript", idx=idx, utility=None, register=register,
        secret_kinds=secret_kinds, nonce=rng.randint(0, 10**9),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng_master = random.Random(SEED)
    # Per-batch RNGs so re-running a single batch is reproducible.
    rng_cli = random.Random(rng_master.randint(0, 2**30))
    rng_transcript = random.Random(rng_master.randint(0, 2**30))
    rng_verify = random.Random(rng_master.randint(0, 2**30))

    start = time.time()

    cli_rows, cli_calls, cli_cost, cli_drops = run_batch(
        label="cli",
        spec_builder=cli_spec_builder,
        target=CLI_TARGET,
        max_calls=CLI_MAX_CALLS,
        stage_path=CLI_STAGE,
        rng=rng_cli,
    )
    print(
        f"\n[cli] final: {len(cli_rows)}/{CLI_TARGET} accepted, "
        f"{cli_calls} calls, ${cli_cost:.2f}, "
        f"top drops: {cli_drops.top(5)}\n"
    )

    transcript_rows, transcript_calls, transcript_cost, transcript_drops = run_batch(
        label="transcript",
        spec_builder=transcript_spec_builder,
        target=TRANSCRIPT_TARGET,
        max_calls=TRANSCRIPT_MAX_CALLS,
        stage_path=TRANSCRIPT_STAGE,
        rng=rng_transcript,
    )
    print(
        f"\n[transcript] final: {len(transcript_rows)}/{TRANSCRIPT_TARGET} accepted, "
        f"{transcript_calls} calls, ${transcript_cost:.2f}, "
        f"top drops: {transcript_drops.top(5)}\n"
    )

    if len(cli_rows) < CLI_TARGET or len(transcript_rows) < TRANSCRIPT_TARGET:
        print("WARNING: did not hit target counts; emitting shards from what we have.")

    cli_paths = shard_to_files(cli_rows[:CLI_TARGET], "synthetic_cli", CLI_FILES) if len(cli_rows) >= CLI_TARGET else []
    transcript_paths = shard_to_files(transcript_rows[:TRANSCRIPT_TARGET], "synthetic_transcript", TRANSCRIPT_FILES) if len(transcript_rows) >= TRANSCRIPT_TARGET else []

    if cli_paths:
        verify_outputs("synthetic_cli", CLI_FILES, CLI_TARGET // CLI_FILES, rng_verify)
    if transcript_paths:
        verify_outputs("synthetic_transcript", TRANSCRIPT_FILES, TRANSCRIPT_TARGET // TRANSCRIPT_FILES, rng_verify)

    elapsed = time.time() - start
    print("\n=== FINAL REPORT ===")
    print(f"runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"cli: {len(cli_rows)}/{CLI_TARGET} accepted, {cli_calls} calls, ${cli_cost:.2f}")
    print(f"transcript: {len(transcript_rows)}/{TRANSCRIPT_TARGET} accepted, {transcript_calls} calls, ${transcript_cost:.2f}")
    print(f"total cost: ${cli_cost + transcript_cost:.2f}")
    print(f"cli drop reasons:")
    for reason, cnt in cli_drops.top(10):
        print(f"  {reason}: {cnt}")
    print(f"transcript drop reasons:")
    for reason, cnt in transcript_drops.top(10):
        print(f"  {reason}: {cnt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
