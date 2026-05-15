//! Trainable-parameter registry for LoRA fine-tuning.
//!
//! [`TrainableParams`] collects every gradient target for a privacy-filter
//! fine-tuning run — that's `(2 × num_layers × 4)` LoRA matrices plus the
//! classifier head's `score.weight` and `score.bias`. For the shipped
//! 8-layer checkpoint that's **66 params**.
//!
//! ## Why a registry?
//!
//! [`crate::autograd::value_and_grad`] takes a positional list of
//! [`MxArray`] references and returns gradients in the same order. The
//! optimizer (Phase D) keeps per-name moment state. We need a single
//! source of truth that pairs names with arrays so the two sides stay
//! aligned — that's [`TrainableParams`].
//!
//! ## Naming convention
//!
//! Matches what `LoRALinear::params()` and the adapter checkpoint use:
//! - LoRA: `model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_a` / `.lora_b`
//! - Classifier: `score.weight`, `score.bias`
//!
//! ## Mutation flow
//!
//! Construction takes `&ModelWeights` and stores [`MxArray`] *clones* —
//! cheap because `MxArray::clone()` is an `Arc` bump. After autograd +
//! optimizer produce a `name → new array` map, call
//! [`write_back_trainable_params`] with `&mut ModelWeights` to install
//! the new values. No interior-mutability machinery is needed: the
//! trainer owns the `&mut` and drives the cycle.
//!
//! Base (non-trainable) weights are never touched by the write-back
//! path. The `value_and_grad_only_updates_lora` test below asserts that
//! invariant byte-for-byte.

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::privacy_filter::{LoadedProj, ModelWeights};
use crate::nn::lora::LoRALinear;

/// Role of a trainable parameter — used for diagnostics and to let the
/// optimizer (Phase D) special-case different param families if needed.
///
/// Only the four variants below appear in the registry. The registry
/// has no notion of "frozen" params — base weights are simply absent
/// from [`TrainableParams::from_privacy_filter`]'s output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamRole {
    LoraA,
    LoraB,
    ClassifierWeight,
    ClassifierBias,
}

/// One trainable parameter — a fully-qualified name, its role, and an
/// `Arc`-shared handle to the underlying array.
pub struct TrainableParam {
    pub name: String,
    pub role: ParamRole,
    pub array: MxArray,
}

/// Ordered collection of trainable parameters.
///
/// The order is stable across rebuilds for a given model topology: all
/// per-layer LoRA params first (layer 0 → N-1, each layer in the order
/// `q.a, q.b, k.a, k.b, v.a, v.b, o.a, o.b`), then the classifier head
/// (`score.weight`, `score.bias`). This is the order returned by
/// [`Self::arrays`] and is what [`crate::autograd::value_and_grad`]
/// consumes positionally.
pub struct TrainableParams {
    params: Vec<TrainableParam>,
}

impl TrainableParams {
    /// Walk a privacy-filter [`ModelWeights`] and collect every LoRA A/B
    /// matrix plus the classifier head's `score.weight` and `score.bias`.
    ///
    /// Errors if *no* LoRA adapter is attached to any attention
    /// projection — running gradient descent against a checkpoint with
    /// no adapter is almost always a mistake (the only way to make
    /// progress would be to update the classifier head alone, and we'd
    /// rather force the caller to opt into that explicitly later).
    pub fn from_privacy_filter(weights: &ModelWeights) -> Result<Self> {
        let mut params = Vec::with_capacity(weights.layers.len() * 4 * 2 + 2);

        let proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"];
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let attn = &layer.self_attn;
            let projs: [&LoadedProj; 4] = [&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj];
            for (proj_name, proj) in proj_names.iter().zip(projs.iter()) {
                let lora = proj.lora().ok_or_else(|| {
                    Error::from_reason(format!(
                        "TrainableParams::from_privacy_filter: no LoRA adapter attached on \
                         model.layers.{layer_idx}.self_attn.{proj_name} \
                         (call PrivacyFilterModel::load_adapter or attach a zero-B adapter \
                         before building the registry)"
                    ))
                })?;
                let [(key_a, a), (key_b, b)] = lora.params();
                params.push(TrainableParam {
                    name: key_a.to_string(),
                    role: ParamRole::LoraA,
                    array: a.clone(),
                });
                params.push(TrainableParam {
                    name: key_b.to_string(),
                    role: ParamRole::LoraB,
                    array: b.clone(),
                });
            }
        }

        // Classifier head. Always trainable when fine-tuning the
        // privacy-filter — every published adapter ships a fresh head.
        params.push(TrainableParam {
            name: "score.weight".to_string(),
            role: ParamRole::ClassifierWeight,
            array: weights.score_weight.clone(),
        });
        params.push(TrainableParam {
            name: "score.bias".to_string(),
            role: ParamRole::ClassifierBias,
            array: weights.score_bias.clone(),
        });

        Ok(Self { params })
    }

    /// Number of trainable parameters.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// `true` when the registry holds no parameters.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Borrow each parameter's fully-qualified name, in the registry's
    /// canonical order.
    pub fn names(&self) -> Vec<&str> {
        self.params.iter().map(|p| p.name.as_str()).collect()
    }

    /// Borrow each parameter's [`MxArray`] in the registry's canonical
    /// order — ready to be passed to
    /// [`crate::autograd::value_and_grad`].
    pub fn arrays(&self) -> Vec<&MxArray> {
        self.params.iter().map(|p| &p.array).collect()
    }

    /// Borrow the full ordered slice of parameter records.
    pub fn params(&self) -> &[TrainableParam] {
        &self.params
    }

    /// Iterate over the parameter records in canonical order.
    pub fn iter(&self) -> impl Iterator<Item = &TrainableParam> {
        self.params.iter()
    }

    /// Look up the role of the parameter named `name`. Returns `None`
    /// when no such parameter is in the registry.
    pub fn role(&self, name: &str) -> Option<ParamRole> {
        self.params.iter().find(|p| p.name == name).map(|p| p.role)
    }
}

/// Install optimizer-produced new values back into the model.
///
/// `updates` is a `name → new array` map. Names must match the canonical
/// keys used by [`TrainableParams::from_privacy_filter`] — anything else
/// is rejected with a `Error::from_reason` so silent typos don't cause
/// the optimizer step to be quietly dropped.
///
/// LoRA writes are batched per-projection: if a projection receives only
/// one of its two arrays (e.g. only `lora_a`) the other side keeps its
/// current value. This lets the caller do partial updates without
/// triggering [`LoRALinear::set_weights`]'s shape check on the missing
/// side.
///
/// Both **shape and dtype** are validated for the classifier head before
/// the swap; a mismatch is rejected loudly so an f32 optimizer update
/// can't silently upcast a bf16 base tensor (which would in turn drag
/// the entire residual stream up to f32 — see the cast logic in
/// `LoRALinear::forward`). For LoRA arrays, shape validation is done by
/// [`LoRALinear::set_weights`]; an explicit dtype check here catches
/// mismatches before that call so the error message names the offending
/// parameter and dtypes directly.
///
/// Base weights (embeddings, layer norms, expert MLPs, attention sinks,
/// etc.) are never touched by this function — only the LoRA matrices
/// and the classifier head.
pub fn write_back_trainable_params(
    weights: &mut ModelWeights,
    updates: &HashMap<String, MxArray>,
) -> Result<()> {
    // First pass: validate every key against the schema. Reject unknown
    // names up front so a typo doesn't leave the model in a half-updated
    // state.
    let num_layers = weights.layers.len();
    for name in updates.keys() {
        if !is_known_trainable_name(name, num_layers) {
            return Err(Error::from_reason(format!(
                "write_back_trainable_params: unknown parameter name {name:?} \
                 (expected one of model.layers.{{0..{n}}}.self_attn.{{q,k,v,o}}_proj.lora_{{a,b}}, \
                 score.weight, score.bias)",
                n = num_layers.saturating_sub(1),
            )));
        }
    }

    // Classifier head — always direct, no batching needed.
    if let Some(new_w) = updates.get("score.weight") {
        let cur_shape = weights.score_weight.shape()?;
        let new_shape = new_w.shape()?;
        if cur_shape.as_ref() != new_shape.as_ref() {
            return Err(Error::from_reason(format!(
                "write_back_trainable_params: score.weight shape mismatch — expected {:?}, got {:?}",
                cur_shape.as_ref(),
                new_shape.as_ref(),
            )));
        }
        let cur_dtype = weights.score_weight.dtype()?;
        let new_dtype = new_w.dtype()?;
        if cur_dtype != new_dtype {
            return Err(Error::from_reason(format!(
                "write_back_trainable_params: score.weight dtype mismatch — expected {cur_dtype:?}, got {new_dtype:?} \
                 (cast the optimizer-produced update to match the base param dtype before write-back)"
            )));
        }
        weights.score_weight = new_w.clone();
    }
    if let Some(new_b) = updates.get("score.bias") {
        let cur_shape = weights.score_bias.shape()?;
        let new_shape = new_b.shape()?;
        if cur_shape.as_ref() != new_shape.as_ref() {
            return Err(Error::from_reason(format!(
                "write_back_trainable_params: score.bias shape mismatch — expected {:?}, got {:?}",
                cur_shape.as_ref(),
                new_shape.as_ref(),
            )));
        }
        let cur_dtype = weights.score_bias.dtype()?;
        let new_dtype = new_b.dtype()?;
        if cur_dtype != new_dtype {
            return Err(Error::from_reason(format!(
                "write_back_trainable_params: score.bias dtype mismatch — expected {cur_dtype:?}, got {new_dtype:?} \
                 (cast the optimizer-produced update to match the base param dtype before write-back)"
            )));
        }
        weights.score_bias = new_b.clone();
    }

    // LoRA writes: batch per-projection so we issue at most one
    // `set_weights` call per (layer, proj). For sides not present in
    // `updates`, fall back to the current adapter value to satisfy
    // `set_weights`'s "both arrays required" contract without changing
    // the missing side.
    let proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"];
    for layer_idx in 0..num_layers {
        for proj_name in proj_names.iter() {
            let key_a = format!("model.layers.{layer_idx}.self_attn.{proj_name}.lora_a");
            let key_b = format!("model.layers.{layer_idx}.self_attn.{proj_name}.lora_b");
            let new_a = updates.get(&key_a);
            let new_b = updates.get(&key_b);
            if new_a.is_none() && new_b.is_none() {
                continue;
            }

            let lora_slot = projection_lora_slot_mut(weights, layer_idx, proj_name);
            let lora = lora_slot.as_mut().ok_or_else(|| {
                Error::from_reason(format!(
                    "write_back_trainable_params: update targets \
                     model.layers.{layer_idx}.self_attn.{proj_name} but no LoRA adapter \
                     is attached there"
                ))
            })?;

            // Explicit dtype check up front. `LoRALinear::set_weights`
            // already validates shapes, but it does NOT check dtype —
            // so an f32 grad update against a bf16 base would silently
            // change the adapter dtype here. Catching it before the
            // swap gives a parameter-named error message.
            if let Some(a) = new_a {
                let cur_dtype = lora.a().dtype()?;
                let new_dtype = a.dtype()?;
                if cur_dtype != new_dtype {
                    return Err(Error::from_reason(format!(
                        "write_back_trainable_params: {key_a} dtype mismatch — \
                         expected {cur_dtype:?}, got {new_dtype:?} \
                         (cast the optimizer-produced update to match the base param dtype before write-back)"
                    )));
                }
            }
            if let Some(b) = new_b {
                let cur_dtype = lora.b().dtype()?;
                let new_dtype = b.dtype()?;
                if cur_dtype != new_dtype {
                    return Err(Error::from_reason(format!(
                        "write_back_trainable_params: {key_b} dtype mismatch — \
                         expected {cur_dtype:?}, got {new_dtype:?} \
                         (cast the optimizer-produced update to match the base param dtype before write-back)"
                    )));
                }
            }

            // Snapshot the current sides so we can fill in the
            // unchanged one. Both are cheap Arc bumps.
            let a_next = new_a.cloned().unwrap_or_else(|| lora.a().clone());
            let b_next = new_b.cloned().unwrap_or_else(|| lora.b().clone());
            lora.set_weights(a_next, b_next)?;
        }
    }

    Ok(())
}

/// Install the given arrays into the model's trainable slots **in
/// registry order** without dtype/shape coercion.
///
/// Unlike [`write_back_trainable_params`] (which performs validation and
/// supports partial updates), this is a fast, positional swap meant for
/// the autograd hot path: the loss closure receives MLX-side tracer
/// arrays for every trainable parameter and must install them into the
/// model's LoRA / classifier slots before running forward, otherwise the
/// trace doesn't connect the inputs to the output and all gradients come
/// back as zero.
///
/// `arrays` must match the order produced by
/// [`TrainableParams::from_privacy_filter`] (all LoRA params first in
/// layer-major / `q,k,v,o` order, then `score.weight`, `score.bias`).
///
/// Returns an `Error::from_reason` if the array count doesn't match the
/// model topology. Shape/dtype mismatches surface inside `set_weights`
/// for LoRA slots; the classifier head is swapped without shape checks
/// because the caller (autograd) is the only path here.
pub(crate) fn install_trainable_arrays_in_order(
    weights: &mut ModelWeights,
    arrays: &[MxArray],
) -> Result<()> {
    let num_layers = weights.layers.len();
    let expected = num_layers * 4 * 2 + 2;
    if arrays.len() != expected {
        return Err(Error::from_reason(format!(
            "install_trainable_arrays_in_order: expected {expected} arrays, got {}",
            arrays.len()
        )));
    }

    let mut cursor = 0usize;
    let proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"];
    for layer_idx in 0..num_layers {
        for proj_name in proj_names.iter() {
            let a_new = arrays[cursor].clone();
            let b_new = arrays[cursor + 1].clone();
            cursor += 2;
            let slot = projection_lora_slot_mut(weights, layer_idx, proj_name);
            let lora = slot.as_mut().ok_or_else(|| {
                Error::from_reason(format!(
                    "install_trainable_arrays_in_order: no LoRA on layer {layer_idx} {proj_name}"
                ))
            })?;
            lora.set_weights(a_new, b_new)?;
        }
    }
    weights.score_weight = arrays[cursor].clone();
    weights.score_bias = arrays[cursor + 1].clone();
    Ok(())
}

/// Recognises every key emitted by [`TrainableParams::from_privacy_filter`].
fn is_known_trainable_name(name: &str, num_layers: usize) -> bool {
    if name == "score.weight" || name == "score.bias" {
        return true;
    }
    // model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_{a,b}
    let rest = match name.strip_prefix("model.layers.") {
        Some(r) => r,
        None => return false,
    };
    let Some(dot) = rest.find('.') else {
        return false;
    };
    let (idx_str, tail) = rest.split_at(dot);
    let Ok(idx) = idx_str.parse::<usize>() else {
        return false;
    };
    if idx >= num_layers {
        return false;
    }
    matches!(
        tail,
        ".self_attn.q_proj.lora_a"
            | ".self_attn.q_proj.lora_b"
            | ".self_attn.k_proj.lora_a"
            | ".self_attn.k_proj.lora_b"
            | ".self_attn.v_proj.lora_a"
            | ".self_attn.v_proj.lora_b"
            | ".self_attn.o_proj.lora_a"
            | ".self_attn.o_proj.lora_b"
    )
}

/// Borrow the `Option<LoRALinear>` slot for the named projection.
fn projection_lora_slot_mut<'a>(
    weights: &'a mut ModelWeights,
    layer_idx: usize,
    proj_name: &str,
) -> &'a mut Option<LoRALinear> {
    let attn = &mut weights.layers[layer_idx].self_attn;
    match proj_name {
        "q_proj" => attn.q_proj.lora_mut(),
        "k_proj" => attn.k_proj.lora_mut(),
        "v_proj" => attn.v_proj.lora_mut(),
        "o_proj" => attn.o_proj.lora_mut(),
        // `is_known_trainable_name` guarantees we never reach this branch.
        _ => unreachable!("projection_lora_slot_mut: unexpected proj_name {proj_name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{DType, MxArray};
    use crate::autograd::value_and_grad;
    use crate::models::privacy_filter::{
        AttnWeights, LayerWeights, LoadedProj, LoadedRouter, MlpWeights, ModelWeights,
        PrivacyFilterModel,
    };
    use crate::moe::{RouterConfig, RoutingMode, TopKRouter};
    use crate::nn::lora::{LoRAConfig, LoRALinear};
    use std::path::PathBuf;

    fn checkpoint_dir() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter")
    }

    /// Mirror of `forward.rs::attach_zero_adapter_and_classify` — attach a
    /// zero-B LoRA adapter to every attention projection so the registry
    /// has something to find.
    fn attach_zero_adapter(model: &mut PrivacyFilterModel, rank: i64) {
        let cfg = LoRAConfig {
            rank,
            alpha: rank as f32 * 2.0,
            dropout: 0.0,
        };
        let num_layers = model.loaded.weights.layers.len();
        let hidden = model.loaded.config.hidden_size as i64;
        let num_q = model.loaded.config.num_attention_heads as i64;
        let num_kv = model.loaded.config.num_key_value_heads as i64;
        let head_dim = model.loaded.config.head_dim as i64;
        let q_out = num_q * head_dim;
        let kv_out = num_kv * head_dim;

        for i in 0..num_layers {
            let attn = &mut model.loaded.weights.layers[i].self_attn;
            *attn.q_proj.lora_mut() = Some(
                LoRALinear::new_zero_b(
                    hidden,
                    q_out,
                    &cfg,
                    &format!("model.layers.{i}.self_attn.q_proj"),
                    DType::BFloat16,
                )
                .unwrap(),
            );
            *attn.k_proj.lora_mut() = Some(
                LoRALinear::new_zero_b(
                    hidden,
                    kv_out,
                    &cfg,
                    &format!("model.layers.{i}.self_attn.k_proj"),
                    DType::BFloat16,
                )
                .unwrap(),
            );
            *attn.v_proj.lora_mut() = Some(
                LoRALinear::new_zero_b(
                    hidden,
                    kv_out,
                    &cfg,
                    &format!("model.layers.{i}.self_attn.v_proj"),
                    DType::BFloat16,
                )
                .unwrap(),
            );
            *attn.o_proj.lora_mut() = Some(
                LoRALinear::new_zero_b(
                    q_out,
                    hidden,
                    &cfg,
                    &format!("model.layers.{i}.self_attn.o_proj"),
                    DType::BFloat16,
                )
                .unwrap(),
            );
        }
    }

    /// Build a tiny `ModelWeights` with zero LoRA adapters and a plain
    /// single layer. Used by [`errors_when_no_adapter_attached`].
    fn build_stub_weights_without_lora() -> ModelWeights {
        let hidden = 4i64;
        let num_classes = 3i64;
        let intermediate = 8i64;
        let num_experts = 2i64;
        let top_k = 1i64;

        let plain = |shape: &[i64]| MxArray::zeros(shape, Some(DType::Float32)).unwrap();
        let make_proj = |shape: &[i64]| LoadedProj::Plain {
            weight: plain(shape),
            bias: None,
            lora: None,
        };

        let router = TopKRouter::new(
            RouterConfig {
                num_experts: num_experts as usize,
                hidden: hidden as usize,
                top_k: top_k as usize,
                mode: RoutingMode::GptOss,
            },
            plain(&[num_experts, hidden]),
            plain(&[num_experts]),
        )
        .unwrap();

        let layer = LayerWeights {
            input_layernorm: plain(&[hidden]),
            post_attention_layernorm: plain(&[hidden]),
            self_attn: AttnWeights {
                q_proj: make_proj(&[hidden, hidden]),
                k_proj: make_proj(&[hidden, hidden]),
                v_proj: make_proj(&[hidden, hidden]),
                o_proj: make_proj(&[hidden, hidden]),
                sinks: plain(&[1]),
            },
            mlp: MlpWeights {
                router: LoadedRouter::Plain(router),
                gate_up_proj: make_proj(&[num_experts, hidden, 2 * intermediate]),
                gate_up_bias: plain(&[num_experts, 2 * intermediate]),
                down_proj: make_proj(&[num_experts, intermediate, hidden]),
                down_bias: plain(&[num_experts, hidden]),
            },
        };

        ModelWeights {
            embed_tokens: plain(&[16, hidden]),
            final_norm: plain(&[hidden]),
            score_weight: plain(&[num_classes, hidden]),
            score_bias: plain(&[num_classes]),
            layers: vec![layer],
            score_weight_backup: None,
            score_bias_backup: None,
        }
    }

    #[test]
    fn errors_when_no_adapter_attached() {
        let weights = build_stub_weights_without_lora();
        let err = match TrainableParams::from_privacy_filter(&weights) {
            Ok(_) => panic!("expected from_privacy_filter to fail without LoRA"),
            Err(e) => e,
        };
        let msg = err.reason.clone();
        assert!(
            msg.contains("no LoRA adapter attached"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn counts_match_expected() {
        let mut model = PrivacyFilterModel::load_from_dir(&checkpoint_dir()).expect("load");
        attach_zero_adapter(&mut model, 16);

        let params = TrainableParams::from_privacy_filter(&model.loaded.weights).expect("collect");

        // 8 layers × 4 projections × 2 LoRA arrays + 2 classifier = 66.
        let num_layers = model.loaded.weights.layers.len();
        let expected = num_layers * 4 * 2 + 2;
        assert_eq!(expected, 66, "shipped checkpoint should have 8 layers");
        assert_eq!(
            params.len(),
            expected,
            "expected {expected} trainable params, got {}",
            params.len()
        );

        let names: Vec<&str> = params.names();

        // Spot-check a few canonical names.
        for layer_idx in 0..num_layers {
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let key_a = format!("model.layers.{layer_idx}.self_attn.{proj_name}.lora_a");
                let key_b = format!("model.layers.{layer_idx}.self_attn.{proj_name}.lora_b");
                assert!(
                    names.iter().any(|n| *n == key_a),
                    "missing LoRA-A name {key_a}",
                );
                assert!(
                    names.iter().any(|n| *n == key_b),
                    "missing LoRA-B name {key_b}",
                );
            }
        }
        assert!(names.contains(&"score.weight"));
        assert!(names.contains(&"score.bias"));

        // Roles look-up: pick a few representatives.
        assert_eq!(
            params.role("model.layers.0.self_attn.q_proj.lora_a"),
            Some(ParamRole::LoraA),
        );
        assert_eq!(
            params.role("model.layers.7.self_attn.o_proj.lora_b"),
            Some(ParamRole::LoraB),
        );
        assert_eq!(
            params.role("score.weight"),
            Some(ParamRole::ClassifierWeight)
        );
        assert_eq!(params.role("score.bias"), Some(ParamRole::ClassifierBias));
        assert_eq!(params.role("not.a.real.param"), None);
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn value_and_grad_only_updates_lora() -> Result<()> {
        let mut model = PrivacyFilterModel::load_from_dir(&checkpoint_dir()).expect("load");
        attach_zero_adapter(&mut model, 16);

        // Snapshot a base weight that must NOT be touched: layer 0
        // q_proj.weight (raw bf16 bytes via `to_uint16_native`).
        let base_q_proj_handle: MxArray = {
            let w = model.loaded.weights.layers[0].self_attn.q_proj.weight();
            w.clone()
        };
        let before_bytes: Vec<u16> = {
            base_q_proj_handle.eval();
            base_q_proj_handle
                .to_uint16_native()
                .expect("uint16 extract")
        };

        // Build the registry.
        let params = TrainableParams::from_privacy_filter(&model.loaded.weights).expect("collect");
        assert!(!params.is_empty());

        // Snapshot the layer-0 q_proj LoRA-A bytes (f32, since fp32-LoRA
        // was rolled in alongside this trainer). We expect this to change
        // after the SGD step below — if it doesn't, the test is silently
        // a no-op against `write_back_trainable_params` and would
        // continue to pass even if the write-back path were gutted.
        let lora_a_before: Vec<f32> = {
            let a = params.arrays()[0];
            a.eval();
            a.to_float32().expect("lora_a to_float32").to_vec()
        };

        // Drive `value_and_grad` against the registry. Sum every param
        // (cast to f32) so the loss depends on every input and every
        // gradient is non-zero — without this, MLX's autograd produces
        // zero grads for unused inputs (see the existing
        // `test_gradient_of_sum` in `autograd.rs`), and the SGD step
        // becomes a no-op for those params. We need *non-zero* grads on
        // LoRA-A specifically so the bit-change assertion below is real
        // signal.
        let (_loss, grads) = value_and_grad(params.arrays(), |arrays| {
            // Fold sum-of-elements (in f32) across every trainable param.
            let mut acc = arrays[0].astype(DType::Float32)?.sum(None, None)?;
            for arr in arrays.iter().skip(1) {
                let term = arr.astype(DType::Float32)?.sum(None, None)?;
                acc = acc.add(&term)?;
            }
            Ok(acc)
        })
        .expect("value_and_grad");

        // Apply a tiny SGD step using every gradient. Mirror the registry
        // order.
        let lr = 1e-3_f64;
        let mut updates: HashMap<String, MxArray> = HashMap::new();
        for (param, grad) in params.iter().zip(grads.iter()) {
            // new = param - lr * grad (in param's dtype)
            let param_dtype = param.array.dtype()?;
            let grad_cast = grad.astype(param_dtype)?;
            let step = grad_cast.mul_scalar(lr)?;
            let step_cast = step.astype(param_dtype)?;
            let new_arr = param.array.sub(&step_cast)?;
            let new_arr_cast = new_arr.astype(param_dtype)?;
            new_arr_cast.eval();
            updates.insert(param.name.clone(), new_arr_cast);
        }

        write_back_trainable_params(&mut model.loaded.weights, &updates).expect("write back");

        // The base layer-0 q_proj.weight must not have been touched —
        // the registry doesn't include it and the write-back path only
        // mutates LoRA slots and the classifier head.
        let after_bytes: Vec<u16> = {
            let w = model.loaded.weights.layers[0].self_attn.q_proj.weight();
            w.eval();
            w.to_uint16_native().expect("uint16 extract")
        };
        assert_eq!(
            before_bytes, after_bytes,
            "base q_proj.weight bytes changed after LoRA update — frozen invariant violated",
        );

        // The layer-0 q_proj LoRA-A *must* have changed — otherwise
        // we'd be passing the frozen-base assertion above with a no-op
        // write-back and the test wouldn't actually exercise the
        // gradient flow it claims to test.
        let lora_a_after: Vec<f32> = {
            let a = model.loaded.weights.layers[0]
                .self_attn
                .q_proj
                .lora()
                .expect("layer 0 q_proj lora present")
                .a()
                .clone();
            a.eval();
            a.to_float32().expect("lora_a after to_float32").to_vec()
        };
        assert_eq!(
            lora_a_before.len(),
            lora_a_after.len(),
            "lora_a length changed",
        );
        assert!(
            lora_a_before
                .iter()
                .zip(lora_a_after.iter())
                .any(|(b, a)| (a - b).abs() > 0.0),
            "lora_a bytes are bit-identical after the SGD step — write-back was a no-op",
        );
        Ok(())
    }

    #[test]
    fn write_back_handles_partial_lora_update() {
        // Build a small `ModelWeights` and attach a zero-B LoRA to the
        // layer-0 q_proj. Pass only `lora_a` in the updates map — the
        // missing `lora_b` slot must keep its current value.
        let mut weights = build_stub_weights_without_lora();
        let hidden = 4i64;
        let cfg = LoRAConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
        };
        let lora = LoRALinear::new_zero_b(
            hidden,
            hidden,
            &cfg,
            "model.layers.0.self_attn.q_proj",
            DType::Float32,
        )
        .expect("new_zero_b");
        *weights.layers[0].self_attn.q_proj.lora_mut() = Some(lora);

        let lora_a_before: Vec<f32> = weights.layers[0]
            .self_attn
            .q_proj
            .lora()
            .expect("attached")
            .a()
            .clone()
            .to_float32()
            .expect("to_float32 a")
            .to_vec();
        let lora_b_before: Vec<f32> = weights.layers[0]
            .self_attn
            .q_proj
            .lora()
            .expect("attached")
            .b()
            .clone()
            .to_float32()
            .expect("to_float32 b")
            .to_vec();

        // Build an `updates` map with ONLY lora_a (an array of all ones,
        // distinct from the random-uniform initialization so we can be
        // sure the change is visible).
        let mut updates: HashMap<String, MxArray> = HashMap::new();
        let new_a = MxArray::ones(&[hidden, cfg.rank], Some(DType::Float32)).expect("ones");
        updates.insert("model.layers.0.self_attn.q_proj.lora_a".to_string(), new_a);

        write_back_trainable_params(&mut weights, &updates).expect("write_back");

        // lora_a must have changed to all ones.
        let lora_a_after: Vec<f32> = weights.layers[0]
            .self_attn
            .q_proj
            .lora()
            .expect("attached")
            .a()
            .clone()
            .to_float32()
            .expect("to_float32 a")
            .to_vec();
        assert_eq!(lora_a_after.len(), lora_a_before.len());
        assert!(
            lora_a_after.iter().all(|v| (v - 1.0).abs() < 1e-6),
            "lora_a should be all 1.0 after partial update, got {lora_a_after:?}",
        );

        // lora_b must be byte-identical to before — the partial-update
        // contract is that the missing side is left alone.
        let lora_b_after: Vec<f32> = weights.layers[0]
            .self_attn
            .q_proj
            .lora()
            .expect("attached")
            .b()
            .clone()
            .to_float32()
            .expect("to_float32 b")
            .to_vec();
        assert_eq!(
            lora_b_before, lora_b_after,
            "lora_b changed during partial update — partial-update contract violated",
        );
    }

    #[test]
    fn write_back_rejects_unknown_name() {
        let mut weights = build_stub_weights_without_lora();
        let mut updates = HashMap::new();
        updates.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            MxArray::zeros(&[4, 4], Some(DType::Float32)).unwrap(),
        );
        let err = write_back_trainable_params(&mut weights, &updates)
            .expect_err("expected unknown-name rejection");
        let msg = err.reason.clone();
        assert!(
            msg.contains("unknown parameter name"),
            "unexpected error message: {msg}"
        );
    }
}
