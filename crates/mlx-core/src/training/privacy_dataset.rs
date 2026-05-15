//! Token-classification dataset reader for privacy-filter LoRA training.
//!
//! Reads pre-tokenized JSONL examples and assembles padded mini-batches.
//! Each line of the JSONL file is one example:
//!
//! ```json
//! {"ids": [142, 89, ...], "labels": [0, 0, 29, 30, ...], "len": 78}
//! ```
//!
//! - `ids` are token ids that the privacy-filter tokenizer would produce
//!   for the source text.
//! - `labels` are integer BIOES class ids (`0..num_classes`). Positions
//!   the trainer should ignore (e.g. continuation pieces of a wordpiece)
//!   must be stored as `-100` so [`Losses::cross_entropy`]'s
//!   `ignore_index` masking drops them from the loss.
//! - `len` is the unpadded length; redundant with `ids.len()` but kept
//!   for compatibility with mlx-lm-style data prep scripts.
//!
//! [`make_batch`] truncates over-length examples to `max_seq_len`, then
//! right-pads ids with `pad_id` and labels with `-100` so all examples
//! share a common length within the batch.

use std::path::Path;

use napi::bindgen_prelude::*;
use serde::Deserialize;

use crate::array::MxArray;

/// One pre-tokenized example: token ids plus per-token integer labels.
///
/// `labels.len()` is allowed to differ from `ids.len()` only if the
/// data-prep pipeline padded labels separately — typical JSONL files
/// have `ids.len() == labels.len() == len`. [`make_batch`] truncates
/// both arrays to `min(len, max_seq_len)` so any pre-padding the
/// dataset already applied is ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenClassificationExample {
    pub ids: Vec<i64>,
    pub labels: Vec<i32>,
    pub len: usize,
}

#[derive(Debug, Deserialize)]
struct TokenClassificationJson {
    ids: Vec<i64>,
    labels: Vec<i32>,
    #[serde(default)]
    len: Option<usize>,
}

/// Read every line of a JSONL file into a `Vec<TokenClassificationExample>`.
///
/// Empty lines are skipped. Each non-empty line must parse as a JSON
/// object with the keys `ids` and `labels`. If `len` is absent it
/// defaults to `ids.len()`. Returns the first parse error encountered
/// with the line number for easy debugging.
pub fn read_jsonl(path: &Path) -> Result<Vec<TokenClassificationExample>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| Error::from_reason(format!("read_jsonl: failed to read {path:?}: {e}")))?;

    let mut out = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed: TokenClassificationJson = serde_json::from_str(trimmed).map_err(|e| {
            Error::from_reason(format!(
                "read_jsonl: parse error on line {} of {path:?}: {e}",
                idx + 1
            ))
        })?;
        let len = parsed.len.unwrap_or(parsed.ids.len());
        out.push(TokenClassificationExample {
            ids: parsed.ids,
            labels: parsed.labels,
            len,
        });
    }
    Ok(out)
}

/// Assemble a padded mini-batch from a slice of examples.
///
/// - Truncates each example's `ids` / `labels` to `max_seq_len`.
/// - The batch's `T` is the longest example after truncation, but never
///   larger than `max_seq_len`.
/// - `ids` are right-padded with `pad_id` (i32) and `labels` with
///   `-100` (i32). The trainer's cross-entropy call must use
///   `ignore_index=-100` to drop those positions from the loss.
/// - Returns `(ids, labels)` as `int32` MxArrays of shape `[B, T]`.
///
/// Errors when `examples` is empty (callers usually fold a check at
/// their batching layer; we still guard here for symmetry).
pub fn make_batch(
    examples: &[TokenClassificationExample],
    pad_id: i64,
    max_seq_len: usize,
) -> Result<(MxArray, MxArray)> {
    if examples.is_empty() {
        return Err(Error::from_reason(
            "make_batch: examples slice is empty".to_string(),
        ));
    }
    if max_seq_len == 0 {
        return Err(Error::from_reason(
            "make_batch: max_seq_len must be > 0".to_string(),
        ));
    }

    let batch = examples.len();

    // Find the truncated max length across the batch.
    let mut t = 0usize;
    for ex in examples {
        let n = ex.ids.len().min(max_seq_len);
        if n > t {
            t = n;
        }
    }
    if t == 0 {
        return Err(Error::from_reason(
            "make_batch: all examples are empty".to_string(),
        ));
    }

    let pad_id_i32 = pad_id as i32;

    let mut flat_ids: Vec<i32> = Vec::with_capacity(batch * t);
    let mut flat_labels: Vec<i32> = Vec::with_capacity(batch * t);

    for ex in examples {
        // Truncate input ids to min(ex.ids.len(), max_seq_len).
        let n_ids = ex.ids.len().min(max_seq_len);
        for &id in &ex.ids[..n_ids] {
            flat_ids.push(id as i32);
        }
        for _ in n_ids..t {
            flat_ids.push(pad_id_i32);
        }

        // Truncate labels — they may have a different length from ids
        // in malformed datasets, but we always emit `t` labels per row.
        let n_labels = ex.labels.len().min(t);
        for &lab in &ex.labels[..n_labels] {
            flat_labels.push(lab);
        }
        for _ in n_labels..t {
            flat_labels.push(-100);
        }
    }

    let shape = [batch as i64, t as i64];
    let ids_arr = MxArray::from_int32(&flat_ids, &shape)?;
    let labels_arr = MxArray::from_int32(&flat_labels, &shape)?;
    Ok((ids_arr, labels_arr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    fn write_tmp(contents: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mlx_privacy_dataset_{}.jsonl",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn read_jsonl_basic() {
        let body = "\
{\"ids\":[1,2,3],\"labels\":[0,1,2],\"len\":3}
{\"ids\":[4,5],\"labels\":[-100,1],\"len\":2}

{\"ids\":[6],\"labels\":[0]}
";
        let path = write_tmp(body);
        let exs = read_jsonl(&path).expect("read");
        assert_eq!(exs.len(), 3);
        assert_eq!(exs[0].ids, vec![1, 2, 3]);
        assert_eq!(exs[0].labels, vec![0, 1, 2]);
        assert_eq!(exs[0].len, 3);
        assert_eq!(exs[1].ids, vec![4, 5]);
        assert_eq!(exs[1].labels, vec![-100, 1]);
        assert_eq!(exs[1].len, 2);
        // Third example: len omitted, defaults to ids.len().
        assert_eq!(exs[2].ids, vec![6]);
        assert_eq!(exs[2].labels, vec![0]);
        assert_eq!(exs[2].len, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn make_batch_pads_and_truncates() {
        let exs = vec![
            // 5 tokens, will be truncated to 4 (max_seq_len=4).
            TokenClassificationExample {
                ids: vec![10, 11, 12, 13, 14],
                labels: vec![0, 1, 2, 3, 4],
                len: 5,
            },
            // 2 tokens, will be padded to 4.
            TokenClassificationExample {
                ids: vec![20, 21],
                labels: vec![-100, 7],
                len: 2,
            },
        ];

        let (ids, labels) = make_batch(&exs, 99, 4).expect("make_batch");

        // Shape check.
        let ids_shape = ids.shape().unwrap().to_vec();
        let labels_shape = labels.shape().unwrap().to_vec();
        assert_eq!(ids_shape, vec![2, 4]);
        assert_eq!(labels_shape, vec![2, 4]);

        assert_eq!(ids.dtype().unwrap(), DType::Int32);
        assert_eq!(labels.dtype().unwrap(), DType::Int32);

        // Pull bytes back to verify pad placement.
        let ids_vec = ids.to_int32().unwrap().to_vec();
        let labels_vec = labels.to_int32().unwrap().to_vec();

        // Row 0: truncated to first 4 ids/labels (no padding).
        assert_eq!(&ids_vec[..4], &[10, 11, 12, 13]);
        assert_eq!(&labels_vec[..4], &[0, 1, 2, 3]);

        // Row 1: padded ids=[20,21,99,99], labels=[-100,7,-100,-100].
        assert_eq!(&ids_vec[4..], &[20, 21, 99, 99]);
        assert_eq!(&labels_vec[4..], &[-100, 7, -100, -100]);
    }
}
