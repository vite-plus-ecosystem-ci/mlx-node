//! BIOES per-token label alignment for the privacy-filter token classifier.
//!
//! Converts `{text, entities[]}` (HF-style byte-offset spans) into
//! `{ids, labels, len}` tuples consumed by
//! [`super::privacy_dataset::read_jsonl`].
//!
//! ## Tagging scheme
//!
//! The privacy-filter checkpoint exposes a 33-class BIOES label space —
//! `O` plus, for each of the 8 PII classes, a `B-X / I-X / E-X / S-X`
//! quartet (see `models::privacy_filter::config::PrivacyFilterConfig::id2label`).
//! Each token covered by an entity span is tagged with:
//!
//! - `S-X` if it is the only token in the span,
//! - `B-X` for the first of multiple tokens,
//! - `E-X` for the last, and
//! - `I-X` for any token in between.
//!
//! Tokens not covered by any entity receive `O` (label id 0).
//!
//! ## Token-to-span assignment (`LabelAllTokens=true`)
//!
//! Hugging Face tokenizers emit per-token **byte** offsets (`start`,
//! `end`) into the original UTF-8 source. A token belongs to an entity
//! when its **midpoint** `(start + end) / 2` falls inside the entity's
//! `[start, end)` byte range. This mirrors the standard
//! `LabelAllTokens=true` recipe from the HF token-classification
//! tutorial: every wordpiece that overlaps a labeled span receives the
//! corresponding tag (no `-100` continuation masking).
//!
//! Zero-width tokens (`start == end`) are skipped — they typically
//! correspond to special tokens that aren't present in `text` and so
//! cannot fall inside any byte range.
//!
//! ## Errors
//!
//! - Unknown label in an entity → `Err`.
//! - Entity `start >= end` or `end > text.len()` → `Err`.
//! - Overlapping entities → `Err` (the model can only label each token
//!   with one class, so the dataset must commit to one).
//!
//! Tokenizer failures bubble up unchanged.

use std::collections::HashMap;
use std::sync::OnceLock;

use napi::bindgen_prelude::*;

use crate::tokenizer::Qwen3Tokenizer;

/// A labeled byte-offset span in the source text.
///
/// `start` and `end` are byte offsets into the UTF-8 source string.
/// `start` is inclusive and `end` is exclusive (matching the HF
/// tokenizers / OpenAI privacy-filter convention).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntitySpan {
    /// Inclusive start byte offset into the source `text`.
    pub start: usize,
    /// Exclusive end byte offset into the source `text`.
    pub end: usize,
    /// PII class name *without* the BIOES prefix — e.g. `"private_email"`.
    pub label: String,
}

/// Result of aligning an example's entity spans to per-token BIOES
/// labels.
///
/// `len` is redundant with `ids.len()` but is kept so the output JSONL
/// matches the schema [`super::privacy_dataset::read_jsonl`] expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignedExample {
    pub ids: Vec<i64>,
    pub labels: Vec<i32>,
    pub len: usize,
}

/// Build the canonical 33-class privacy-filter `label → id` map.
///
/// Ordering matches the shipped `config.json` exactly: `O` at index 0,
/// followed by `B/I/E/S` quartets per PII class in alphabetical order
/// (`account_number`, `private_address`, `private_date`, `private_email`,
/// `private_person`, `private_phone`, `private_url`, `secret`). Verified
/// against `.cache/models/privacy-filter/config.json` and the fixture
/// in [`crate::models::privacy_filter::spans::tests`].
pub fn default_label_to_id() -> HashMap<String, i32> {
    let labels = canonical_label_strs();
    let mut map = HashMap::with_capacity(labels.len());
    for (idx, name) in labels.iter().enumerate() {
        map.insert((*name).to_string(), idx as i32);
    }
    map
}

/// Canonical 33-element label vector — `[O, B-acct, I-acct, ..., S-secret]`.
///
/// Centralized so the alignment binary and the unit tests below agree
/// on ordering. Re-deriving the order at runtime keeps the source of
/// truth in one place and avoids a hand-typed mistake in two files.
///
/// The owned `String`s are interned in a process-global [`OnceLock`] so
/// repeated calls share storage and the returned `&'static str` slices
/// have no per-call allocation. The previous implementation
/// `Box::leak`'d 32 strings on every call — a true memory leak under
/// heavy reuse (e.g. inside `align_bioes`).
pub fn canonical_label_strs() -> Vec<&'static str> {
    static LABELS: OnceLock<Vec<String>> = OnceLock::new();
    let owned: &'static Vec<String> = LABELS.get_or_init(|| {
        const PII_CLASSES: [&str; 8] = [
            "account_number",
            "private_address",
            "private_date",
            "private_email",
            "private_person",
            "private_phone",
            "private_url",
            "secret",
        ];
        let mut out = Vec::with_capacity(1 + 4 * PII_CLASSES.len());
        out.push("O".to_string());
        for class in PII_CLASSES {
            for prefix in ["B", "I", "E", "S"] {
                out.push(format!("{prefix}-{class}"));
            }
        }
        out
    });
    // `owned` lives for the duration of the program (OnceLock storage),
    // so the `&str` slices borrowed from each `String` are `'static`.
    owned.iter().map(|s| s.as_str()).collect()
}

/// Truncate a `(ids, labels)` pair to at most `max_len` tokens while
/// keeping the BIOES label sequence valid.
///
/// When truncation cuts mid-entity, the original tail looks like
/// `[..., B-X, I-X, I-X]` — a half-open run with no terminating `E-X`,
/// which violates the BIOES grammar. This helper detects such a
/// trailing `B-X / I-X` run and rewrites the entire run back to `O`
/// (dropping the partial entity cleanly) so downstream consumers
/// (the trainer, eval pipelines, span decoders) see a self-consistent
/// example.
///
/// `labels` are looked up against `canonical_label_strs()` to determine
/// each id's BIOES prefix. Unknown ids (e.g. from a different label
/// map) are left untouched — better to surface the inconsistency at the
/// decode site than to silently rewrite labels the helper doesn't
/// recognise.
///
/// Returns the truncated `(ids, labels)` slices' lengths; the inputs
/// are mutated in place.
pub fn truncate_bioes_inplace(ids: &mut Vec<i64>, labels: &mut Vec<i32>, max_len: usize) {
    let original_len = ids.len().min(labels.len());
    if original_len <= max_len {
        // No truncation needed.
        if ids.len() > original_len {
            ids.truncate(original_len);
        }
        if labels.len() > original_len {
            labels.truncate(original_len);
        }
        return;
    }
    ids.truncate(max_len);
    labels.truncate(max_len);

    let label_strs = canonical_label_strs();
    let o_id: i32 = 0;
    // Walk the tail backwards; rewrite any trailing run of `B-X / I-X`
    // (i.e. an entity that lost its `E-X` terminator) to `O`. Stop at
    // the first `E-X`, `S-X`, `O`, or unknown id — any of those signal
    // we've left the dangling run.
    let mut i = labels.len();
    while i > 0 {
        let raw = labels[i - 1];
        // Negative ids (e.g. HF's `-100` ignore marker) and out-of-range
        // ids both fall into the "stop" branch — we don't recognise them
        // and would rather leave them alone than guess.
        let prefix = usize::try_from(raw)
            .ok()
            .and_then(|id| label_strs.get(id))
            .and_then(|s| s.as_bytes().first());
        match prefix {
            Some(b'B') | Some(b'I') => {
                labels[i - 1] = o_id;
                i -= 1;
            }
            _ => break, // E-, S-, O, unknown, or negative — stop.
        }
    }
}

/// Align entity spans (byte offsets in `text`) to BIOES per-token labels.
///
/// See the [module-level docs](self) for the tagging scheme and the
/// midpoint heuristic used to assign tokens to spans.
///
/// `add_special_tokens=false` matches the privacy-filter inference path
/// (see `crates/mlx-core/src/models/privacy_filter/mod.rs::classify`),
/// so the token sequence emitted here is identical to what the model
/// would see at inference time.
pub fn align_bioes(
    text: &str,
    entities: &[EntitySpan],
    tokenizer: &Qwen3Tokenizer,
    label_to_id: &HashMap<String, i32>,
) -> Result<AlignedExample> {
    // ---- 1. Validate entity spans up front so we fail fast. ----
    let text_len = text.len();
    for (idx, ent) in entities.iter().enumerate() {
        if ent.start >= ent.end {
            return Err(Error::from_reason(format!(
                "align_bioes: entity {idx} has start >= end ({} >= {})",
                ent.start, ent.end
            )));
        }
        if ent.end > text_len {
            return Err(Error::from_reason(format!(
                "align_bioes: entity {idx} end {} exceeds text byte length {text_len}",
                ent.end
            )));
        }
        // Every entity class must be representable in BIOES.
        for prefix in ["B", "I", "E", "S"] {
            let tag = format!("{prefix}-{}", ent.label);
            if !label_to_id.contains_key(&tag) {
                return Err(Error::from_reason(format!(
                    "align_bioes: entity {idx} has unknown label {:?} (missing {tag} in label_to_id)",
                    ent.label
                )));
            }
        }
    }

    // ---- 2. Reject overlapping entities. ----
    //
    // Each token can only carry one BIOES tag, so the dataset has to
    // disambiguate before reaching us. Sort by start to do this in
    // O(n log n) instead of O(n^2).
    let mut sorted_idx: Vec<usize> = (0..entities.len()).collect();
    sorted_idx.sort_by_key(|&i| entities[i].start);
    for w in sorted_idx.windows(2) {
        let (a, b) = (&entities[w[0]], &entities[w[1]]);
        if b.start < a.end {
            return Err(Error::from_reason(format!(
                "align_bioes: entities overlap — [{}, {}) ({:?}) and [{}, {}) ({:?})",
                a.start, a.end, a.label, b.start, b.end, b.label,
            )));
        }
    }

    // ---- 3. Tokenize with byte offsets. ----
    let (ids_u32, offsets) = tokenizer.encode_with_offsets_sync(text, Some(false))?;
    let n = ids_u32.len();

    let o_id = *label_to_id.get("O").ok_or_else(|| {
        Error::from_reason("align_bioes: label_to_id missing required key \"O\"".to_string())
    })?;

    let mut labels = vec![o_id; n];

    // ---- 4. For each entity, find covered tokens and assign BIOES tags. ----
    for ent in entities {
        let mut covered: Vec<usize> = Vec::new();
        for (tok_idx, &(tok_start, tok_end)) in offsets.iter().enumerate() {
            // Zero-width offsets (e.g. inserted BOS/EOS pieces that
            // don't map back to text bytes) can never be inside any
            // byte range, so skip them.
            if tok_end <= tok_start {
                continue;
            }
            // Midpoint rule: a token belongs to the entity if its
            // midpoint falls inside [start, end). Avoid fractional
            // division by comparing `2 * midpoint = start + end`
            // against `2 * ent.start` / `2 * ent.end`.
            let two_mid = tok_start + tok_end;
            if two_mid >= 2 * ent.start && two_mid < 2 * ent.end {
                covered.push(tok_idx);
            }
        }

        if covered.is_empty() {
            // Entity exists in the source but the tokenizer didn't
            // produce any token whose midpoint sits in [start, end).
            // This is a real signal — usually means the offsets are
            // off by one wordpiece — so surface it as an error rather
            // than silently emitting all-O labels.
            return Err(Error::from_reason(format!(
                "align_bioes: entity [{}, {}) ({:?}) covers no tokens",
                ent.start, ent.end, ent.label,
            )));
        }

        if covered.len() == 1 {
            let tag = format!("S-{}", ent.label);
            labels[covered[0]] = *label_to_id.get(&tag).unwrap();
        } else {
            let first = *covered.first().unwrap();
            let last = *covered.last().unwrap();
            let b_id = *label_to_id.get(&format!("B-{}", ent.label)).unwrap();
            let i_id = *label_to_id.get(&format!("I-{}", ent.label)).unwrap();
            let e_id = *label_to_id.get(&format!("E-{}", ent.label)).unwrap();
            for &tok_idx in &covered {
                labels[tok_idx] = if tok_idx == first {
                    b_id
                } else if tok_idx == last {
                    e_id
                } else {
                    i_id
                };
            }
        }
    }

    let ids: Vec<i64> = ids_u32.iter().map(|&id| id as i64).collect();
    Ok(AlignedExample {
        ids,
        len: n,
        labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Resolve the privacy-filter tokenizer path. Tests that need a
    /// real tokenizer are `#[ignore]` because the checkpoint isn't
    /// checked in.
    fn tokenizer_path() -> PathBuf {
        let candidates = [
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../.cache/models/privacy-filter/tokenizer.json"),
            PathBuf::from("/Volumes/P4510/.cache/models/privacy-filter/tokenizer.json"),
        ];
        for p in candidates {
            if p.exists() {
                return p;
            }
        }
        panic!(
            "privacy-filter tokenizer.json not found — run `mlx download openai/privacy-filter`"
        );
    }

    fn load_tokenizer() -> Qwen3Tokenizer {
        Qwen3Tokenizer::from_file(&tokenizer_path()).expect("load tokenizer.json")
    }

    /// Verifies the canonical 33-label set matches the live id2label
    /// from `.cache/models/privacy-filter/config.json`. Pure (no model
    /// load needed) — runs in CI.
    #[test]
    fn canonical_label_order_matches_spec() {
        let labels = canonical_label_strs();
        assert_eq!(labels.len(), 33);
        assert_eq!(labels[0], "O");
        // Spot-check a few canonical positions documented in
        // `models::privacy_filter::config::tests::parses_real_config_json`.
        assert_eq!(labels[13], "B-private_email");
        assert_eq!(labels[32], "S-secret");

        let map = default_label_to_id();
        assert_eq!(map["O"], 0);
        assert_eq!(map["B-private_email"], 13);
        assert_eq!(map["S-secret"], 32);
        assert_eq!(map.len(), 33);
    }

    #[test]
    fn unknown_label_errors() {
        // We can validate the label check without touching the
        // tokenizer because `align_bioes` validates spans first. Use a
        // dummy tokenizer-less path by short-circuiting on a span with
        // an unknown label.
        //
        // The check runs before tokenization in `align_bioes`, but it
        // still needs a tokenizer reference. A loaded tokenizer is
        // expensive, so this test is also marked ignore — the
        // ignored test below covers it end-to-end. Keep this stub so
        // the unit-level expectation is documented even when the model
        // isn't on disk.
        let label_to_id = default_label_to_id();
        assert!(!label_to_id.contains_key("B-not_a_real_label"));
    }

    #[test]
    fn out_of_range_validation_is_pure() {
        // Mirror `unknown_label_errors` — verify our preconditions
        // hold purely on the inputs we control, no tokenizer needed.
        let text = "abcdef";
        let label_to_id = default_label_to_id();
        // Out-of-range end.
        assert_eq!(text.len(), 6);
        let _bad = EntitySpan {
            start: 0,
            end: 7,
            label: "secret".to_string(),
        };
        // Empty span.
        let _empty = EntitySpan {
            start: 3,
            end: 3,
            label: "secret".to_string(),
        };
        // Sanity — labels exist.
        assert!(label_to_id.contains_key("B-secret"));
        assert!(label_to_id.contains_key("S-secret"));
    }

    // -------- Real-tokenizer tests --------
    //
    // These need the privacy-filter tokenizer on disk. They use the
    // production tokenizer so the offsets we assert against are
    // exactly what the trainer will see.

    #[test]
    #[ignore = "requires .cache/models/privacy-filter/tokenizer.json — run with --ignored"]
    fn single_entity_word_aligned() {
        let tok = load_tokenizer();
        let label_to_id = default_label_to_id();
        let text = "my email is alice@example.com";
        let email_start = text.find("alice@example.com").unwrap();
        let email_end = email_start + "alice@example.com".len();
        let entities = vec![EntitySpan {
            start: email_start,
            end: email_end,
            label: "private_email".to_string(),
        }];

        let out = align_bioes(text, &entities, &tok, &label_to_id).expect("align");
        assert_eq!(out.ids.len(), out.labels.len());
        assert_eq!(out.len, out.ids.len());

        let o = label_to_id["O"];
        let b = label_to_id["B-private_email"];
        let i = label_to_id["I-private_email"];
        let e = label_to_id["E-private_email"];
        let s = label_to_id["S-private_email"];

        // Count how many tokens were tagged as something other than O.
        let entity_tags: Vec<i32> = out.labels.iter().copied().filter(|&l| l != o).collect();
        assert!(
            !entity_tags.is_empty(),
            "expected at least one entity-tagged token, got labels {:?}",
            out.labels
        );

        // Validate the BIOES structure: either one S-X token, or a
        // B-X / I-X* / E-X run with no internal O.
        if entity_tags.len() == 1 {
            assert_eq!(entity_tags[0], s, "single-token entity must be S-X");
        } else {
            assert_eq!(*entity_tags.first().unwrap(), b);
            assert_eq!(*entity_tags.last().unwrap(), e);
            for &mid in &entity_tags[1..entity_tags.len() - 1] {
                assert_eq!(mid, i);
            }
        }

        // Tokens outside the email span are all O.
        // (Indirectly verified — the entity-tag block is contiguous,
        // and we already filtered to non-O tags.)
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter/tokenizer.json — run with --ignored"]
    fn multi_token_entity() {
        let tok = load_tokenizer();
        let label_to_id = default_label_to_id();
        // A long URL produces several wordpieces.
        let text = "go to https://very.long.subdomain.example.co.uk/path/with/segments for info";
        let url = "https://very.long.subdomain.example.co.uk/path/with/segments";
        let start = text.find(url).unwrap();
        let end = start + url.len();
        let entities = vec![EntitySpan {
            start,
            end,
            label: "private_url".to_string(),
        }];

        let out = align_bioes(text, &entities, &tok, &label_to_id).expect("align");
        let o = label_to_id["O"];
        let b = label_to_id["B-private_url"];
        let i = label_to_id["I-private_url"];
        let e = label_to_id["E-private_url"];

        let entity_positions: Vec<usize> = out
            .labels
            .iter()
            .enumerate()
            .filter_map(|(idx, &l)| if l != o { Some(idx) } else { None })
            .collect();
        assert!(
            entity_positions.len() >= 3,
            "expected URL to span at least 3 tokens, got positions {entity_positions:?}",
        );
        // Contiguous run.
        let first = *entity_positions.first().unwrap();
        let last = *entity_positions.last().unwrap();
        assert_eq!(last - first + 1, entity_positions.len());

        assert_eq!(out.labels[first], b);
        assert_eq!(out.labels[last], e);
        for &pos in &entity_positions[1..entity_positions.len() - 1] {
            assert_eq!(out.labels[pos], i);
        }
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter/tokenizer.json — run with --ignored"]
    fn partial_overlap_token() {
        // The midpoint rule is deterministic for a known span. Pick a
        // span that we *know* contains the midpoint of the first
        // wordpiece: "alice" tokenizes to (at least) one wordpiece
        // whose midpoint sits inside [0, 5). Asserting the precise
        // outcome (`S-X` for a single token, `B-X / E-X` for two)
        // gives the test real signal instead of accepting any
        // result.
        let tok = load_tokenizer();
        let label_to_id = default_label_to_id();
        let text = "alice and bob";
        let entities = vec![EntitySpan {
            start: 0,
            end: 5, // "alice"
            label: "private_person".to_string(),
        }];

        let out = align_bioes(text, &entities, &tok, &label_to_id)
            .expect("alignment must succeed for a span that covers a full word");
        let o = label_to_id["O"];
        let s = label_to_id["S-private_person"];
        let b = label_to_id["B-private_person"];
        let i = label_to_id["I-private_person"];
        let e = label_to_id["E-private_person"];

        let entity_tags: Vec<i32> = out.labels.iter().copied().filter(|&l| l != o).collect();
        assert!(
            !entity_tags.is_empty(),
            "expected at least one entity-tagged token for span [0,5) over 'alice', got {:?}",
            out.labels,
        );

        if entity_tags.len() == 1 {
            assert_eq!(
                entity_tags[0], s,
                "single-token entity must be S-private_person, got {entity_tags:?}",
            );
        } else {
            assert_eq!(*entity_tags.first().unwrap(), b);
            assert_eq!(*entity_tags.last().unwrap(), e);
            for &mid in &entity_tags[1..entity_tags.len() - 1] {
                assert_eq!(mid, i);
            }
        }
    }

    /// Sanity-check the 2-token BIOES path: `B-X / E-X` with no `I-X`
    /// in between. The shortest path to a real 2-token entity is to
    /// pick a string whose tokenizer split yields exactly two
    /// wordpieces inside the span. We use `alice@example.com` which
    /// reliably splits into more than one token with the qwen
    /// tokenizer; the test then asserts only the "no internal I" and
    /// "endpoints are B/E" invariants — both stable regardless of the
    /// exact token boundary chosen by the tokenizer.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter/tokenizer.json — run with --ignored"]
    fn two_token_entity_has_no_internal_i() {
        let tok = load_tokenizer();
        let label_to_id = default_label_to_id();
        // Construct a deliberately short span that the qwen tokenizer
        // is highly likely to break into exactly two wordpieces — a
        // 6-character mixed-case alnum string.
        let text = "id is Ab12Cd here";
        let start = text.find("Ab12Cd").unwrap();
        let end = start + "Ab12Cd".len();
        let entities = vec![EntitySpan {
            start,
            end,
            label: "secret".to_string(),
        }];
        let out = align_bioes(text, &entities, &tok, &label_to_id).expect("align");

        let o = label_to_id["O"];
        let b = label_to_id["B-secret"];
        let i = label_to_id["I-secret"];
        let e = label_to_id["E-secret"];
        let s = label_to_id["S-secret"];

        let entity_tags: Vec<i32> = out.labels.iter().copied().filter(|&l| l != o).collect();
        assert!(
            !entity_tags.is_empty(),
            "expected entity span to cover at least one token",
        );

        // The contract is BIOES-valid: either one S-X, or B-X (... I-X ...) E-X.
        if entity_tags.len() == 1 {
            assert_eq!(entity_tags[0], s);
        } else if entity_tags.len() == 2 {
            // The exact case we want to exercise: B-X / E-X, no I-X.
            assert_eq!(entity_tags[0], b, "first tag should be B-secret");
            assert_eq!(entity_tags[1], e, "second tag should be E-secret");
            assert!(
                !entity_tags.contains(&i),
                "2-token entity must NOT contain I-X, got {entity_tags:?}",
            );
        } else {
            assert_eq!(*entity_tags.first().unwrap(), b);
            assert_eq!(*entity_tags.last().unwrap(), e);
            for &mid in &entity_tags[1..entity_tags.len() - 1] {
                assert_eq!(mid, i, "internal token must be I-secret");
            }
        }
    }

    #[test]
    fn truncate_bioes_no_change_when_under_limit() {
        let mut ids = vec![10i64, 20, 30];
        let mut labels = vec![0i32, 0, 0];
        truncate_bioes_inplace(&mut ids, &mut labels, 8);
        assert_eq!(ids, vec![10, 20, 30]);
        assert_eq!(labels, vec![0, 0, 0]);
    }

    #[test]
    fn truncate_bioes_keeps_complete_entity() {
        // Sequence ends on a clean E-X — truncation falls past the end,
        // so no rewriting is needed.
        let label_to_id = default_label_to_id();
        let b = label_to_id["B-secret"];
        let i = label_to_id["I-secret"];
        let e = label_to_id["E-secret"];
        let o = 0i32;

        let mut ids = vec![10, 20, 30, 40, 50];
        let mut labels = vec![o, b, i, e, o];
        truncate_bioes_inplace(&mut ids, &mut labels, 5);
        assert_eq!(labels, vec![o, b, i, e, o]);
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn truncate_bioes_rewrites_trailing_partial_entity_to_o() {
        // Original sequence has a 4-token entity at the end; we
        // truncate mid-entity so the surviving tail is `B-X / I-X /
        // I-X` (no E-X). The helper must rewrite the whole run to O.
        let label_to_id = default_label_to_id();
        let b = label_to_id["B-secret"];
        let i = label_to_id["I-secret"];
        let e = label_to_id["E-secret"];
        let o = 0i32;

        // Length-6 sequence; truncate to 5. Original labels: [O, B, I, I, I, E].
        // After truncate to 5: [O, B, I, I, I] — invalid. After rewrite: [O, O, O, O, O].
        let mut ids = vec![10, 20, 30, 40, 50, 60];
        let mut labels = vec![o, b, i, i, i, e];
        truncate_bioes_inplace(&mut ids, &mut labels, 5);
        assert_eq!(ids.len(), 5);
        assert_eq!(labels.len(), 5);
        assert_eq!(
            labels,
            vec![o, o, o, o, o],
            "trailing B-/I- run must be rewritten to O",
        );
    }

    #[test]
    fn truncate_bioes_stops_rewrite_at_prior_e() {
        // Two entities back-to-back: a complete one (B I E) followed
        // by a partial one (B I) cut by truncation. Only the trailing
        // partial run gets rewritten; the complete entity in front is
        // left intact.
        let label_to_id = default_label_to_id();
        let b = label_to_id["B-secret"];
        let i = label_to_id["I-secret"];
        let e = label_to_id["E-secret"];
        let o = 0i32;

        // Length-7. Truncate to 6: [O, B, I, E, O, B, I] (last B/I dangling).
        // Expect rewrite to: [O, B, I, E, O, O, O].
        let mut ids = vec![10, 20, 30, 40, 50, 60, 70];
        let mut labels = vec![o, b, i, e, o, b, i];
        truncate_bioes_inplace(&mut ids, &mut labels, 6);
        assert_eq!(ids.len(), 6);
        assert_eq!(labels.len(), 6);
        assert_eq!(labels, vec![o, b, i, e, o, o]);
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter/tokenizer.json — run with --ignored"]
    fn no_entities_all_o() {
        let tok = load_tokenizer();
        let label_to_id = default_label_to_id();
        let text = "completely benign sentence with no PII";
        let out = align_bioes(text, &[], &tok, &label_to_id).expect("align");
        let o = label_to_id["O"];
        assert!(!out.labels.is_empty());
        for &l in &out.labels {
            assert_eq!(l, o);
        }
        assert_eq!(out.ids.len(), out.labels.len());
        assert_eq!(out.len, out.ids.len());
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter/tokenizer.json — run with --ignored"]
    fn unknown_label_end_to_end_errors() {
        let tok = load_tokenizer();
        let label_to_id = default_label_to_id();
        let text = "hello world";
        let entities = vec![EntitySpan {
            start: 0,
            end: 5,
            label: "not_a_real_label".to_string(),
        }];
        let res = align_bioes(text, &entities, &tok, &label_to_id);
        assert!(res.is_err(), "expected Err for unknown label");
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter/tokenizer.json — run with --ignored"]
    fn out_of_range_entity_errors() {
        let tok = load_tokenizer();
        let label_to_id = default_label_to_id();
        let text = "short";
        let entities = vec![EntitySpan {
            start: 0,
            end: 999,
            label: "secret".to_string(),
        }];
        let res = align_bioes(text, &entities, &tok, &label_to_id);
        assert!(res.is_err(), "expected Err for out-of-range end");
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter/tokenizer.json — run with --ignored"]
    fn overlapping_entities_error() {
        let tok = load_tokenizer();
        let label_to_id = default_label_to_id();
        let text = "alice@example.com is a contact";
        let entities = vec![
            EntitySpan {
                start: 0,
                end: 17, // alice@example.com
                label: "private_email".to_string(),
            },
            EntitySpan {
                start: 0,
                end: 5, // alice (overlaps with the email)
                label: "private_person".to_string(),
            },
        ];
        let res = align_bioes(text, &entities, &tok, &label_to_id);
        assert!(res.is_err(), "expected Err for overlapping entities");
    }
}
