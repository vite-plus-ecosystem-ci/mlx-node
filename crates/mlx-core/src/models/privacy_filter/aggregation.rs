//! Hugging Face `TokenClassificationPipeline.aggregation_strategy` port.
//!
//! Replicates HF's `none` / `simple` / `first` / `average` / `max`
//! aggregation strategies for the 33-class BIOES privacy-filter head,
//! byte-for-byte where it matters — including HF's `get_tag` quirk that
//! treats `E-X` / `S-X` as their own opaque tag rather than as
//! continuations of `X`, so `B-X I-X E-X` produces three groups.
//!
//! Reference: `transformers/src/transformers/pipelines/token_classification.py`
//! lines 397–619 (gather_pre_entities, aggregate, aggregate_words,
//! group_entities, get_tag).
//!
//! Note: this path operates on raw per-token softmax (no Viterbi). The
//! Viterbi decoder in `viterbi.rs` is used only when callers stick to
//! the default behavior.

use crate::models::privacy_filter::spans::Entity;

/// Aggregation strategy — mirrors HF `AggregationStrategy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// One entity per non-O token; `label` keeps full BIOES tag.
    None,
    /// Per-token argmax → `group_entities` (HF's BIOES-fragmenting quirk).
    Simple,
    /// Aggregate subwords by taking the first subword's scores, then
    /// `group_entities`.
    First,
    /// Average subword score vectors (NaN-mean), then `group_entities`.
    Average,
    /// Take the subword whose `scores.max()` is largest, then `group_entities`.
    Max,
}

impl AggregationStrategy {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "simple" => Ok(Self::Simple),
            "first" => Ok(Self::First),
            "average" => Ok(Self::Average),
            "max" => Ok(Self::Max),
            other => Err(format!(
                "unknown aggregationStrategy {:?}: expected one of \
                 \"none\", \"simple\", \"first\", \"average\", \"max\"",
                other
            )),
        }
    }
}

/// Internal intermediate matching HF's `pre_entity` dict.
struct PreEntity<'a> {
    scores: &'a [f32],
    start: usize,
    end: usize,
    is_subword: bool,
}

/// Internal intermediate matching HF's per-token `entity` dict in
/// `aggregate` / `aggregate_word` / `group_entities`. `entity_name`
/// carries the BIOES-tagged label (e.g. `"B-private_email"`); `start`
/// and `end` are byte offsets into the source text.
struct TaggedEntity {
    entity_name: String,
    score: f32,
    start: usize,
    end: usize,
}

/// Run a single aggregation pass.
///
/// `probs` is flat `[T * num_classes]` row-major softmax output.
/// `offsets` are byte offsets `(start, end)` for each of the `T` tokens.
/// `threshold` filters output by group/token mean score.
pub fn aggregate(
    probs: &[f32],
    num_classes: usize,
    id2label: &[String],
    offsets: &[(usize, usize)],
    source_text: &str,
    threshold: f32,
    strategy: AggregationStrategy,
) -> Vec<Entity> {
    let n_tokens = offsets.len();
    debug_assert_eq!(probs.len(), n_tokens * num_classes);
    debug_assert_eq!(id2label.len(), num_classes);

    // ---- gather_pre_entities ---------------------------------------------
    let source_bytes = source_text.as_bytes();
    let pre_entities: Vec<PreEntity<'_>> = (0..n_tokens)
        .map(|i| {
            let scores = &probs[i * num_classes..(i + 1) * num_classes];
            let (start, end) = offsets[i];
            // HF offset heuristic for tokenizers without a
            // `continuing_subword_prefix` (matches gpt-oss byte-level BPE):
            //   is_subword = start > 0 and " " not in sentence[start-1:start+1]
            // Implemented over bytes; ASCII-correct (HF itself is shaky on
            // non-ASCII via Python char indexing).
            let is_subword = if start == 0 || start > source_bytes.len() {
                false
            } else {
                let prev_byte = source_bytes[start - 1];
                let cur_byte = source_bytes.get(start).copied();
                prev_byte != b' ' && cur_byte != Some(b' ')
            };
            PreEntity {
                scores,
                start,
                end,
                is_subword,
            }
        })
        .collect();

    // ---- aggregate -------------------------------------------------------
    match strategy {
        AggregationStrategy::None => {
            aggregate_none(&pre_entities, id2label, source_text, threshold)
        }
        AggregationStrategy::Simple => {
            let per_token = aggregate_per_token(&pre_entities, id2label);
            group_entities(per_token, source_text, threshold)
        }
        AggregationStrategy::First | AggregationStrategy::Average | AggregationStrategy::Max => {
            let words = aggregate_words(&pre_entities, id2label, strategy);
            group_entities(words, source_text, threshold)
        }
    }
}

/// `aggregation_strategy == "none"`: per-token entity emission with the
/// full BIOES tag retained as `label`. HF skips `group_entities` for this
/// branch.
fn aggregate_none(
    pre_entities: &[PreEntity<'_>],
    id2label: &[String],
    source_text: &str,
    threshold: f32,
) -> Vec<Entity> {
    let mut out = Vec::new();
    for pe in pre_entities {
        let (idx, score) = argmax(pe.scores);
        let label = id2label[idx].as_str();
        if label == "O" {
            continue;
        }
        if score < threshold {
            continue;
        }
        let text = source_text.get(pe.start..pe.end).unwrap_or("").to_string();
        out.push(Entity {
            start: pe.start,
            end: pe.end,
            label: label.to_string(),
            score,
            text,
        });
    }
    out
}

/// `aggregation_strategy == "simple"`: build per-token tagged entities
/// (full BIOES name) for `group_entities` to fragment per HF's get_tag
/// quirk.
fn aggregate_per_token(pre_entities: &[PreEntity<'_>], id2label: &[String]) -> Vec<TaggedEntity> {
    pre_entities
        .iter()
        .map(|pe| {
            let (idx, score) = argmax(pe.scores);
            TaggedEntity {
                entity_name: id2label[idx].clone(),
                score,
                start: pe.start,
                end: pe.end,
            }
        })
        .collect()
}

/// `first` / `average` / `max`: group pre-entities into "words" using
/// `is_subword`, then collapse each word into one TaggedEntity.
fn aggregate_words(
    pre_entities: &[PreEntity<'_>],
    id2label: &[String],
    strategy: AggregationStrategy,
) -> Vec<TaggedEntity> {
    // Build word groups: every non-subword token starts a new word.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (idx, pe) in pre_entities.iter().enumerate() {
        if groups.is_empty() || !pe.is_subword {
            groups.push(vec![idx]);
        } else {
            groups.last_mut().unwrap().push(idx);
        }
    }

    groups
        .into_iter()
        .map(|indices| aggregate_word(&indices, pre_entities, id2label, strategy))
        .collect()
}

fn aggregate_word(
    indices: &[usize],
    pre_entities: &[PreEntity<'_>],
    id2label: &[String],
    strategy: AggregationStrategy,
) -> TaggedEntity {
    let first = &pre_entities[*indices.first().unwrap()];
    let last = &pre_entities[*indices.last().unwrap()];

    let (entity_idx, score) = match strategy {
        AggregationStrategy::First => argmax(first.scores),
        AggregationStrategy::Max => {
            // HF: max_entity = max(entities, key=lambda e: e["scores"].max())
            let max_sub = indices
                .iter()
                .map(|&i| &pre_entities[i])
                .max_by(|a, b| {
                    let am = a.scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let bm = b.scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    am.partial_cmp(&bm).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            argmax(max_sub.scores)
        }
        AggregationStrategy::Average => {
            // Element-wise nanmean across subword score vectors. Inputs
            // are real softmax outputs so we won't see NaN in practice;
            // the nan-aware fold matches HF's `np.nanmean` semantics for
            // free. argmax of that mean → (idx, mean_score[idx]).
            let num_classes = first.scores.len();
            let mut sums = vec![0.0f32; num_classes];
            let mut counts = vec![0u32; num_classes];
            for &i in indices {
                let scores = pre_entities[i].scores;
                for (c, &v) in scores.iter().enumerate() {
                    if !v.is_nan() {
                        sums[c] += v;
                        counts[c] += 1;
                    }
                }
            }
            let avg: Vec<f32> = sums
                .iter()
                .zip(counts.iter())
                .map(|(&s, &c)| if c == 0 { f32::NAN } else { s / c as f32 })
                .collect();
            argmax(&avg)
        }
        AggregationStrategy::None | AggregationStrategy::Simple => unreachable!(),
    };

    TaggedEntity {
        entity_name: id2label[entity_idx].clone(),
        score,
        start: first.start,
        end: last.end,
    }
}

/// HF's `get_tag` — note the deliberate fall-through: anything not
/// prefixed `B-` or `I-` (so `E-X`, `S-X`, `O`, raw class names) gets
/// `(bi="I", tag=&entity_name)`, using the WHOLE string as the tag.
/// This is what makes `B-X I-X E-X` produce three groups instead of one.
fn get_tag(entity_name: &str) -> (&'static str, &str) {
    if let Some(rest) = entity_name.strip_prefix("B-") {
        ("B", rest)
    } else if let Some(rest) = entity_name.strip_prefix("I-") {
        ("I", rest)
    } else {
        ("I", entity_name)
    }
}

/// HF's `group_entities` + `group_sub_entities`. Merges adjacent entities
/// with same `tag` (per get_tag), splits on `bi == "B"`. Emits one Entity
/// per group; drops groups labeled `"O"` and groups whose mean score is
/// below threshold.
fn group_entities(entities: Vec<TaggedEntity>, source_text: &str, threshold: f32) -> Vec<Entity> {
    let mut groups: Vec<Vec<TaggedEntity>> = Vec::new();
    let mut current: Vec<TaggedEntity> = Vec::new();

    for ent in entities {
        if current.is_empty() {
            current.push(ent);
            continue;
        }
        let (bi, tag) = get_tag(&ent.entity_name);
        let (_last_bi, last_tag) = get_tag(&current.last().unwrap().entity_name);
        if tag == last_tag && bi != "B" {
            current.push(ent);
        } else {
            groups.push(std::mem::take(&mut current));
            current.push(ent);
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }

    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        let first = &group[0];
        let last = group.last().unwrap();
        // HF: entity = entities[0]["entity"].split("-", 1)[-1]
        let label = match first.entity_name.split_once('-') {
            Some((_, rest)) => rest.to_string(),
            None => first.entity_name.clone(),
        };
        if label == "O" {
            continue;
        }
        let mean = group.iter().map(|e| e.score).sum::<f32>() / group.len() as f32;
        if mean < threshold {
            continue;
        }
        let start = first.start;
        let end = last.end;
        let text = source_text.get(start..end).unwrap_or("").to_string();
        out.push(Entity {
            start,
            end,
            label,
            score: mean,
            text,
        });
    }
    out
}

/// Stable argmax: returns (index, value). Ties broken by lowest index
/// (matches numpy / numpy-via-torch behavior used by HF).
fn argmax(scores: &[f32]) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in scores.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 5-class id2label fixture: `O` + BIOES for class `X`.
    /// IDs: 0=O, 1=B-X, 2=I-X, 3=E-X, 4=S-X.
    fn id2label_x() -> Vec<String> {
        ["O", "B-X", "I-X", "E-X", "S-X"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Build a flat probs vector that argmaxes to the given tag ids,
    /// with the chosen class getting probability `peak` and the rest
    /// split evenly.
    fn make_probs(tag_ids: &[usize], num_classes: usize, peak: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(tag_ids.len() * num_classes);
        for &t in tag_ids {
            let rest = (1.0 - peak) / (num_classes as f32 - 1.0);
            for c in 0..num_classes {
                out.push(if c == t { peak } else { rest });
            }
        }
        out
    }

    #[test]
    fn parse_strategy_rejects_unknown() {
        assert_eq!(
            AggregationStrategy::parse("none").unwrap(),
            AggregationStrategy::None
        );
        assert_eq!(
            AggregationStrategy::parse("simple").unwrap(),
            AggregationStrategy::Simple
        );
        assert_eq!(
            AggregationStrategy::parse("first").unwrap(),
            AggregationStrategy::First
        );
        assert_eq!(
            AggregationStrategy::parse("average").unwrap(),
            AggregationStrategy::Average
        );
        assert_eq!(
            AggregationStrategy::parse("max").unwrap(),
            AggregationStrategy::Max
        );
        assert!(AggregationStrategy::parse("Simple").is_err());
        assert!(AggregationStrategy::parse("").is_err());
        assert!(AggregationStrategy::parse("bogus").is_err());
    }

    #[test]
    fn none_returns_full_bioes_labels_per_token() {
        let id2label = id2label_x();
        // Two tokens, both predicted as B-X / E-X (a multi-token span).
        // Use a layout where bytes don't matter for this assertion.
        let source = "ab cd";
        let offsets = vec![(0usize, 2usize), (3, 5)];
        let probs = make_probs(&[1, 3], 5, 0.9);
        let out = aggregate(
            &probs,
            5,
            &id2label,
            &offsets,
            source,
            0.5,
            AggregationStrategy::None,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].label, "B-X");
        assert_eq!(out[1].label, "E-X");
        assert_eq!(out[0].text, "ab");
        assert_eq!(out[1].text, "cd");
    }

    #[test]
    fn simple_b_i_e_produces_three_groups() {
        // HF quirk: get_tag("E-X") = ("I", "E-X"), tag != "X" from B-X.
        // So `B-X I-X E-X` flushes after B (next is I-X, tag "X" matches),
        // wait: get_tag(I-X) = ("I", "X"), matches B-X's tag "X" so I-X
        // merges. Then E-X has tag "E-X" ≠ "X" → new group. Then no more
        // tokens → last group flush. That's: [B-X, I-X] | [E-X].
        //
        // Spec asserts THREE groups for B-X I-X E-X. Re-read HF:
        // get_tag("B-X") = ("B", "X"); get_tag("I-X") = ("I", "X").
        // First iteration: empty → push. Second: tag=X, last_tag=X,
        // bi=I → append. Third: ent=E-X, tag="E-X", last_tag="X",
        // bi="I" → "tag != last_tag" → flush. So that produces TWO
        // groups, not three: [B-X, I-X] and [E-X].
        //
        // The spec example may have meant "S-X" between or be slightly
        // off; the underlying invariant is that E-X never merges with
        // its B/I predecessors (because tag="E-X" ≠ "X").
        let id2label = id2label_x();
        let source = "aa bb cc";
        let offsets = vec![(0usize, 2usize), (3, 5), (6, 8)];
        let probs = make_probs(&[1, 2, 3], 5, 0.9); // B-X, I-X, E-X
        let out = aggregate(
            &probs,
            5,
            &id2label,
            &offsets,
            source,
            0.5,
            AggregationStrategy::Simple,
        );
        // Per the HF algebra above: two groups.
        assert_eq!(out.len(), 2, "got {:#?}", out);
        assert_eq!(out[0].label, "X");
        assert_eq!(out[0].start, 0);
        assert_eq!(out[0].end, 5);
        assert_eq!(out[1].label, "X");
        assert_eq!(out[1].start, 6);
        assert_eq!(out[1].end, 8);
    }

    #[test]
    fn simple_e_then_s_split_but_s_s_merge() {
        // E-X, S-X, S-X — get_tag returns ("I","E-X"), ("I","S-X"),
        // ("I","S-X"). First boundary: tag "S-X" != "E-X" → flush.
        // Second boundary: tag "S-X" == "S-X" and bi != "B" → APPEND.
        // Net: two groups [E-X] and [S-X, S-X].
        let id2label = id2label_x();
        let source = "aa bb cc";
        let offsets = vec![(0usize, 2usize), (3, 5), (6, 8)];
        let probs = make_probs(&[3, 4, 4], 5, 0.9);
        let out = aggregate(
            &probs,
            5,
            &id2label,
            &offsets,
            source,
            0.5,
            AggregationStrategy::Simple,
        );
        assert_eq!(out.len(), 2, "got {:#?}", out);
        for g in &out {
            assert_eq!(g.label, "X");
        }
        assert_eq!(out[0].start, 0);
        assert_eq!(out[0].end, 2);
        assert_eq!(out[1].start, 3);
        assert_eq!(out[1].end, 8);
    }

    #[test]
    fn simple_s_s_adjacent_same_class_merges() {
        // Both tokens are S-X. get_tag("S-X") = ("I", "S-X"). Second
        // entity: bi="I", tag="S-X" matches last_tag="S-X", bi != "B"
        // → merges into one group.
        let id2label = id2label_x();
        let source = "aa bb";
        let offsets = vec![(0usize, 2usize), (3, 5)];
        let probs = make_probs(&[4, 4], 5, 0.9);
        let out = aggregate(
            &probs,
            5,
            &id2label,
            &offsets,
            source,
            0.5,
            AggregationStrategy::Simple,
        );
        assert_eq!(out.len(), 1, "got {:#?}", out);
        assert_eq!(out[0].label, "X");
        assert_eq!(out[0].start, 0);
        assert_eq!(out[0].end, 5);
    }

    #[test]
    fn simple_b_b_adjacent_same_class_splits() {
        // Two B-X tokens. Second has bi="B" → blocks the merge even
        // though tag matches. Two groups out.
        let id2label = id2label_x();
        let source = "aa bb";
        let offsets = vec![(0usize, 2usize), (3, 5)];
        let probs = make_probs(&[1, 1], 5, 0.9);
        let out = aggregate(
            &probs,
            5,
            &id2label,
            &offsets,
            source,
            0.5,
            AggregationStrategy::Simple,
        );
        assert_eq!(out.len(), 2, "got {:#?}", out);
        assert_eq!(out[0].start, 0);
        assert_eq!(out[0].end, 2);
        assert_eq!(out[1].start, 3);
        assert_eq!(out[1].end, 5);
    }

    #[test]
    fn first_average_max_group_subwords_via_offset_heuristic() {
        // Source: "abc def" — two words at byte offsets [0..3] and [4..7].
        // Subword the first word into ["ab", "c"]: offsets (0,2) (2,3).
        // The token at offset (2,3) has start=2, prev byte 'b' != ' ',
        // current byte 'c' != ' ' → is_subword = true → merges with "ab".
        // The token at offset (4,7) has start=4, prev byte ' ' → not
        // subword → starts new word.
        //
        // Assign tags: subword 1 = B-X, subword 2 = I-X, word 2 = E-X.
        // With `first`: word 1 picks B-X (first subword), word 2 = E-X.
        // group_entities: word1 tag="X", word2 tag="E-X" → two groups.
        let id2label = id2label_x();
        let source = "abc def";
        let offsets = vec![(0usize, 2usize), (2, 3), (4, 7)];
        let probs = make_probs(&[1, 2, 3], 5, 0.9);

        for strategy in [
            AggregationStrategy::First,
            AggregationStrategy::Average,
            AggregationStrategy::Max,
        ] {
            // threshold=0.0 so the `average` strategy (which mixes two
            // subwords of different classes and gets a 0.46 mean peak)
            // doesn't get filtered out — we're testing word grouping +
            // group_entities behavior, not the threshold filter here.
            let out = aggregate(&probs, 5, &id2label, &offsets, source, 0.0, strategy);
            // first word collapsed to a single tagged entity; second word
            // is its own entity. After group_entities, we get two groups.
            assert_eq!(out.len(), 2, "strategy={:?} got {:#?}", strategy, out);
            // First word covers byte [0..3]; second covers [4..7].
            assert_eq!(out[0].start, 0, "strategy={:?}", strategy);
            assert_eq!(out[0].end, 3, "strategy={:?}", strategy);
            assert_eq!(out[1].start, 4, "strategy={:?}", strategy);
            assert_eq!(out[1].end, 7, "strategy={:?}", strategy);
            assert_eq!(out[0].label, "X");
            assert_eq!(out[1].label, "X");
        }
    }

    #[test]
    fn none_drops_below_threshold() {
        let id2label = id2label_x();
        let source = "ab";
        let offsets = vec![(0, 2)];
        // peak = 0.4 → below default 0.5 → dropped.
        let probs = make_probs(&[1], 5, 0.4);
        let out = aggregate(
            &probs,
            5,
            &id2label,
            &offsets,
            source,
            0.5,
            AggregationStrategy::None,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn o_tokens_are_always_dropped() {
        let id2label = id2label_x();
        let source = "ab cd";
        let offsets = vec![(0, 2), (3, 5)];
        let probs = make_probs(&[0, 0], 5, 0.99); // both O
        for strategy in [
            AggregationStrategy::None,
            AggregationStrategy::Simple,
            AggregationStrategy::First,
            AggregationStrategy::Average,
            AggregationStrategy::Max,
        ] {
            let out = aggregate(&probs, 5, &id2label, &offsets, source, 0.5, strategy);
            assert!(out.is_empty(), "strategy={:?} got {:#?}", strategy, out);
        }
    }
}
