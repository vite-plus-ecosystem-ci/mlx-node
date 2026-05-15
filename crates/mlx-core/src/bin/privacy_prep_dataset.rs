//! `mlx-prep-dataset` — preprocess raw `{text, entities[]}` JSONL into
//! pre-tokenized `{ids, labels, len}` JSONL for the privacy-filter
//! token-classification LoRA trainer.
//!
//! Pipeline:
//!
//! 1. Read raw JSONL line-by-line (one example per line).
//! 2. Tokenize the `text` field using the model's own `tokenizer.json`
//!    (`add_special_tokens=false`, matching the inference path).
//! 3. Convert byte-offset entity spans to per-token BIOES labels via
//!    [`mlx_core::training::align_bioes`].
//! 4. Truncate to `--max-seq-len` (default 256).
//! 5. Emit `{"ids":[…],"labels":[…],"len":N}` on stdout-formatted JSONL.
//!
//! Examples whose alignment fails (unknown labels, overlapping spans,
//! off-by-one offsets) are skipped with a warning to stderr; the binary
//! exits zero unless a fatal IO error occurs. This lets a malformed
//! line in a 50k-example dataset cost a warning, not the whole run.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde::Deserialize;

use mlx_core::tokenizer::Qwen3Tokenizer;
use mlx_core::training::{EntitySpan, align_bioes, default_label_to_id, truncate_bioes_inplace};

#[derive(Parser, Debug)]
#[command(
    name = "mlx-prep-dataset",
    about = "Tokenize and BIOES-align raw {text, entities[]} JSONL for the privacy-filter LoRA trainer.",
    version
)]
struct Cli {
    /// Path to the privacy-filter model directory (must contain
    /// `tokenizer.json`). Re-using the model dir guarantees the
    /// tokenization matches what the trainer/inference path sees.
    #[arg(long)]
    model: PathBuf,

    /// Input JSONL with one `{"text": …, "entities": [{"start", "end",
    /// "label"}]}` object per line.
    #[arg(long)]
    input: PathBuf,

    /// Output JSONL — one `{"ids": [...], "labels": [...], "len": N}`
    /// per line.
    #[arg(long)]
    output: PathBuf,

    /// Truncate examples whose token count exceeds this. Defaults to
    /// the trainer's default `max_seq_len`.
    #[arg(long, default_value_t = 256)]
    max_seq_len: usize,
}

#[derive(Debug, Deserialize)]
struct RawEntity {
    start: usize,
    end: usize,
    label: String,
}

#[derive(Debug, Deserialize)]
struct RawExample {
    text: String,
    #[serde(default)]
    entities: Vec<RawEntity>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok((written, skipped)) => {
            eprintln!(
                "mlx-prep-dataset: wrote {written} examples, skipped {skipped} (output: {})",
                cli.output.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("mlx-prep-dataset: fatal error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli) -> Result<(usize, usize), String> {
    let tokenizer_path = cli.model.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(format!(
            "tokenizer.json not found in model dir: {}",
            tokenizer_path.display()
        ));
    }
    let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        format!(
            "failed to load tokenizer at {}: {e}",
            tokenizer_path.display()
        )
    })?;

    let label_to_id = default_label_to_id();

    let input = File::open(&cli.input)
        .map_err(|e| format!("failed to open input {}: {e}", cli.input.display()))?;
    let reader = BufReader::new(input);

    if let Some(parent) = cli.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create output dir {}: {e}", parent.display()))?;
    }
    let output_file = File::create(&cli.output)
        .map_err(|e| format!("failed to create output {}: {e}", cli.output.display()))?;
    let mut writer = BufWriter::new(output_file);

    let mut written = 0usize;
    let mut skipped = 0usize;
    // After the 10th skip warning, suppress subsequent skip-warnings to
    // keep stderr useful on bad datasets. A final summary counts every
    // skip regardless.
    const SKIP_WARN_LIMIT: usize = 10;

    for (idx, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                // I/O failures are categorically different from "bad
                // input on this line" — surface them as ERROR rather
                // than WARN.
                eprintln!(
                    "ERROR: mlx-prep-dataset: read error at {} line {}: {e}",
                    cli.input.display(),
                    idx + 1
                );
                return Err(format!("read error at line {}: {e}", idx + 1));
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match process_line(trimmed, &tokenizer, &label_to_id, cli.max_seq_len) {
            Ok(json_line) => {
                writeln!(writer, "{json_line}")
                    .map_err(|e| format!("write error at line {}: {e}", idx + 1))?;
                written += 1;
            }
            Err(reason) => {
                if skipped < SKIP_WARN_LIMIT {
                    eprintln!(
                        "WARN: mlx-prep-dataset: skipping {} line {}: {reason}",
                        cli.input.display(),
                        idx + 1
                    );
                } else if skipped == SKIP_WARN_LIMIT {
                    eprintln!(
                        "WARN: mlx-prep-dataset: further skip warnings suppressed; see final summary for total count"
                    );
                }
                skipped += 1;
            }
        }
    }

    writer.flush().map_err(|e| format!("flush error: {e}"))?;

    Ok((written, skipped))
}

fn process_line(
    line: &str,
    tokenizer: &Qwen3Tokenizer,
    label_to_id: &std::collections::HashMap<String, i32>,
    max_seq_len: usize,
) -> Result<String, String> {
    let raw: RawExample = serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;

    let entities: Vec<EntitySpan> = raw
        .entities
        .into_iter()
        .map(|e| EntitySpan {
            start: e.start,
            end: e.end,
            label: e.label,
        })
        .collect();

    let aligned = align_bioes(&raw.text, &entities, tokenizer, label_to_id)
        .map_err(|e| format!("alignment failed: {e}"))?;

    // Apply max-seq-len truncation here so the trainer's per-batch
    // truncation in `make_batch` is a no-op for the common case.
    // Truncation may chop a multi-token entity mid-run; the helper
    // rewrites any trailing `B-X / I-X` run (one without its `E-X`
    // terminator) back to `O` so the surviving labels stay
    // BIOES-valid.
    let mut ids = aligned.ids;
    let mut labels = aligned.labels;
    truncate_bioes_inplace(&mut ids, &mut labels, max_seq_len);
    let n = ids.len();

    Ok(serialize_example(&ids, &labels, n))
}

/// Hand-roll the JSONL output to avoid pulling in serde_json's
/// per-line allocations and to guarantee a stable field order.
fn serialize_example(ids: &[i64], labels: &[i32], len: usize) -> String {
    let mut out = String::with_capacity(16 + 6 * ids.len() + 4 * labels.len());
    out.push_str("{\"ids\":[");
    push_i64_list(&mut out, ids);
    out.push_str("],\"labels\":[");
    push_i32_list(&mut out, labels);
    out.push_str("],\"len\":");
    out.push_str(&len.to_string());
    out.push('}');
    out
}

fn push_i64_list(out: &mut String, values: &[i64]) {
    let mut first = true;
    for v in values {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&v.to_string());
    }
}

fn push_i32_list(out: &mut String, values: &[i32]) {
    let mut first = true;
    for v in values {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&v.to_string());
    }
}
