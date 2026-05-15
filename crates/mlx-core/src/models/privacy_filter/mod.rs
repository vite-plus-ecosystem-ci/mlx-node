//! OpenAI Privacy Filter — token-classification PII detector.
pub mod aggregation;
pub mod attention;
pub mod classifier;
pub mod config;
pub mod experts;
pub mod forward;
pub mod persistence;
pub mod quantized_linear;
pub mod spans;
pub mod transformer;
pub mod viterbi;
pub mod yarn;
pub use aggregation::AggregationStrategy;
pub use attention::AttentionLayer;
pub use classifier::classifier_forward;
pub use config::{PrivacyFilterConfig, RopeParameters};
pub use experts::GptOssMlp;
pub use forward::PrivacyFilterModel;
pub use persistence::{
    AttnWeights, LayerWeights, LoadedModel, LoadedRouter, MlpWeights, ModelWeights,
    QuantizationConfig, load_from_directory,
};
pub use quantized_linear::{
    LoadedProj, PrivacyFilterQuantizedSwitchLinear, QuantizedLinear, TensorQuantParams, project_2d,
    project_moe,
};
pub use spans::{Entity, extract_spans};
pub use transformer::Block;
pub use viterbi::{Calibration, build_transition_matrix, label_id, viterbi_decode};
pub use yarn::compute_yarn_freqs;

use crate::array::{DType, MxArray};
use crate::nn::Activations;
use crate::nn::lora::{LoRAConfig, LoRALinear};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;
use std::path::Path;

/// A privacy entity detected by [`PrivacyFilterModelJs::classify`].
///
/// `start`/`end` are byte offsets into the input string (Hugging Face
/// `tokenizers` convention). `label` is the privacy class without the
/// BIOES prefix (e.g. `"private_email"`). `score` is the mean — across
/// the span's tokens — of the softmax probability of the Viterbi-emitted
/// tag at each token.
#[napi(object)]
pub struct PrivacyEntity {
    pub label: String,
    pub start: u32,
    pub end: u32,
    pub score: f64,
    pub text: String,
}

/// Per-token output emitted when [`PrivacyClassifyOptions::return_tokens`]
/// is `true`. `tag` is the full BIOES tag (`"O"` or `"B-..."`/`"I-..."`/
/// `"E-..."`/`"S-..."`) chosen by the Viterbi decoder. `score` is the
/// softmax probability of that emitted tag at this token, so `tag` and
/// `score` always share decoders (at boundary tokens the Viterbi tag can
/// differ from the local argmax).
#[napi(object)]
pub struct PrivacyToken {
    pub text: String,
    pub tag: String,
    pub score: f64,
    pub start: u32,
    pub end: u32,
}

/// Per-call Viterbi calibration overrides.
///
/// Any field set to `Some(_)` overrides the corresponding bias from the
/// model's default calibration (loaded from `viterbi_calibration.json`
/// at load time). Missing fields fall back to the default.
#[napi(object)]
pub struct PrivacyCalibration {
    pub transition_bias_background_stay: Option<f64>,
    pub transition_bias_background_to_start: Option<f64>,
    pub transition_bias_end_to_background: Option<f64>,
    pub transition_bias_end_to_start: Option<f64>,
    pub transition_bias_inside_to_continue: Option<f64>,
    pub transition_bias_inside_to_end: Option<f64>,
}

/// Options for [`PrivacyFilterModelJs::classify`].
///
/// - `threshold` (default `0.5`): minimum mean per-token probability for
///   an extracted span/group to be returned.
/// - `calibration`: per-call overrides on top of the checkpoint default.
///   Ignored when `aggregation_strategy` is set (no Viterbi pass is run).
/// - `return_tokens` (default `false`): when `true`, the result includes
///   a `tokens` array with one entry per input token. When combined with
///   `aggregation_strategy`, the per-token `tag` is the raw argmax BIOES
///   tag and `score` is the argmax probability (no Viterbi).
/// - `aggregation_strategy`: opt into Hugging Face
///   `TokenClassificationPipeline.aggregation_strategy` behavior. One of
///   `"none"`, `"simple"`, `"first"`, `"average"`, `"max"`. When set, the
///   Viterbi decoder is skipped and raw per-token argmax is used.
#[napi(object)]
pub struct PrivacyClassifyOptions {
    pub threshold: Option<f64>,
    pub calibration: Option<PrivacyCalibration>,
    pub return_tokens: Option<bool>,
    pub aggregation_strategy: Option<String>,
}

/// Result of [`PrivacyFilterModelJs::classify`].
#[napi(object)]
pub struct PrivacyClassifyResult {
    pub entities: Vec<PrivacyEntity>,
    pub tokens: Option<Vec<PrivacyToken>>,
}

/// NAPI-exported view of [`PrivacyFilterModel`].
///
/// Construct via [`PrivacyFilterModelJs::load`] and run end-to-end
/// classification via [`PrivacyFilterModelJs::classify`].
#[napi(js_name = "PrivacyFilterModel")]
pub struct PrivacyFilterModelJs {
    inner: PrivacyFilterModel,
}

#[napi]
impl PrivacyFilterModelJs {
    /// Load a privacy-filter checkpoint from a directory.
    ///
    /// The directory must contain `config.json`, `model.safetensors`,
    /// `tokenizer.json`, and optionally `viterbi_calibration.json` and
    /// `tokenizer_config.json`. Synchronous to match the loader pattern
    /// used by every other model in this crate (e.g. `TextDetModel`).
    #[napi(factory)]
    pub fn load(model_path: String) -> Result<Self> {
        let inner = PrivacyFilterModel::load_from_dir(Path::new(&model_path))?;
        Ok(Self { inner })
    }

    /// Classify `text` and return detected PII entities (and optionally
    /// per-token tags).
    ///
    /// Pipeline:
    /// 1. Tokenize the text with byte offsets, no special tokens.
    /// 2. Run the forward pass to get `[1, T, 33]` logits.
    /// 3. Compute softmax (for per-tag confidences) and log-softmax (for
    ///    Viterbi emissions) over the class axis.
    /// 4. Build the transition matrix from the default calibration
    ///    merged with any per-call overrides.
    /// 5. Viterbi-decode using log-softmax emissions to get the BIOES
    ///    tag sequence.
    /// 6. For each token, take the softmax probability of the
    ///    Viterbi-emitted tag (so per-token `tag` and `score` share
    ///    decoders), then walk the tags + offsets + those probabilities
    ///    to extract coherent spans whose mean probability clears
    ///    `threshold`.
    #[napi]
    pub fn classify(
        &self,
        text: String,
        opts: Option<PrivacyClassifyOptions>,
    ) -> Result<PrivacyClassifyResult> {
        let opts = opts.unwrap_or(PrivacyClassifyOptions {
            threshold: None,
            calibration: None,
            return_tokens: None,
            aggregation_strategy: None,
        });
        let threshold = opts.threshold.unwrap_or(0.5) as f32;
        let return_tokens = opts.return_tokens.unwrap_or(false);
        let aggregation_strategy = match opts.aggregation_strategy.as_deref() {
            Some(s) => Some(AggregationStrategy::parse(s).map_err(|e| {
                napi::Error::from_reason(format!("invalid aggregationStrategy: {e}"))
            })?),
            None => None,
        };

        // ---- 1. Tokenize with byte offsets (no special tokens). ----
        //
        // `add_special_tokens=false` matches the HF reference call in
        // `forward.rs`'s test and the gpt-oss / o200k expectation that
        // the token-classification head is fed the raw byte-pair tokens.
        let (ids_u32, offsets) = self
            .inner
            .loaded
            .tokenizer
            .encode_with_offsets_sync(&text, Some(false))?;
        let n_tokens = ids_u32.len();
        let num_classes = self.inner.loaded.label_strs.len();

        // Empty input → empty result. The forward pass would still work
        // on a [1, 0] tensor, but Viterbi would have nothing to do and
        // it's cleaner to short-circuit.
        if n_tokens == 0 {
            return Ok(PrivacyClassifyResult {
                entities: Vec::new(),
                tokens: if return_tokens {
                    Some(Vec::new())
                } else {
                    None
                },
            });
        }

        // ---- 2. Forward pass. ----
        let ids_i32: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
        let input_ids = MxArray::from_int32(&ids_i32, &[1, n_tokens as i64])?;
        let logits = self.inner.forward_logits(&input_ids)?;

        // Promote to f32 so all downstream math has stable headroom.
        // The shipped checkpoint runs in bf16, which makes softmax /
        // log-softmax less precise than the HF reference (which forces
        // f32 internally for its own softmax — see `modeling_opf.py`).
        let logits_f32 = logits.astype(DType::Float32)?;

        // ---- 3. Softmax (for per-tag probabilities, indexed by the
        //         Viterbi pick below) and log-softmax (for Viterbi
        //         emissions). ----
        //
        // Why log-softmax for emissions: the Viterbi decoder adds
        // emission + transition scalars, and transition biases come in
        // additive log-space form (see `viterbi_calibration.json`).
        // Feeding probabilities or raw logits would break that algebra.
        // Log-softmax also gives us numerical stability for free.
        let probs = Activations::softmax(&logits_f32, Some(-1))?;
        let log_probs = Activations::log_softmax(&logits_f32, Some(-1))?;

        // Eval once before two GPU→CPU pulls so we don't replay the
        // forward pass twice.
        probs.eval();
        log_probs.eval();

        let probs_flat: Vec<f32> = probs.to_float32()?.to_vec();
        let log_probs_flat: Vec<f32> = log_probs.to_float32()?.to_vec();
        debug_assert_eq!(probs_flat.len(), n_tokens * num_classes);
        debug_assert_eq!(log_probs_flat.len(), n_tokens * num_classes);

        // Drop GPU-side intermediates and reclaim MLX's buffer cache.
        // Without this, repeated `classify` calls grow the cache
        // unboundedly (one classify allocates ~per-layer hidden states,
        // attention K/V, MLP gate/up/down for 8 layers + the [1, T, 33]
        // logits + softmax/log-softmax tensors).
        drop(probs);
        drop(log_probs);
        drop(logits_f32);
        drop(logits);
        drop(input_ids);
        crate::array::memory::synchronize_and_clear_cache();

        let id2label = self.inner.loaded.label_strs.clone();

        // ---- HF aggregation-strategy branch (skips Viterbi). ----
        //
        // When the caller opts into one of the HF aggregation strategies,
        // we use raw per-token argmax + the HF `gather_pre_entities` →
        // `aggregate` → `group_entities` pipeline instead of our
        // constrained BIOES Viterbi decoder. `calibration` is silently
        // ignored on this path (the spec mandates this).
        if let Some(strategy) = aggregation_strategy {
            let entities_internal = aggregation::aggregate(
                &probs_flat,
                num_classes,
                &id2label,
                &offsets,
                &text,
                threshold,
                strategy,
            );

            let entities: Vec<PrivacyEntity> = entities_internal
                .into_iter()
                .map(|e| PrivacyEntity {
                    label: e.label,
                    start: e.start as u32,
                    end: e.end as u32,
                    score: e.score as f64,
                    text: e.text,
                })
                .collect();

            // Tokens output on this path uses raw per-token argmax — `tag`
            // is the full BIOES label of the argmax class, `score` is its
            // softmax probability. No Viterbi is run on this path.
            let tokens = if return_tokens {
                let mut out = Vec::with_capacity(n_tokens);
                for t in 0..n_tokens {
                    let (start, end) = offsets[t];
                    let tok_text = if start == 0 && end == 0 {
                        self.inner
                            .loaded
                            .tokenizer
                            .id_to_token(ids_u32[t])
                            .unwrap_or_default()
                    } else {
                        text.get(start..end).unwrap_or("").to_string()
                    };
                    let row = &probs_flat[t * num_classes..(t + 1) * num_classes];
                    let mut best_idx = 0usize;
                    let mut best_val = f32::NEG_INFINITY;
                    for (i, &v) in row.iter().enumerate() {
                        if v > best_val {
                            best_val = v;
                            best_idx = i;
                        }
                    }
                    out.push(PrivacyToken {
                        text: tok_text,
                        tag: id2label
                            .get(best_idx)
                            .cloned()
                            .unwrap_or_else(|| "O".to_string()),
                        score: best_val as f64,
                        start: start as u32,
                        end: end as u32,
                    });
                }
                Some(out)
            } else {
                None
            };

            return Ok(PrivacyClassifyResult { entities, tokens });
        }

        // ---- 4. Build the emission matrix [T, num_classes] from
        //         log-softmax. ----
        let mut emit: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let row = log_probs_flat[t * num_classes..(t + 1) * num_classes].to_vec();
            emit.push(row);
        }

        // ---- 5. Merge default calibration with per-call overrides. ----
        let cal = merge_calibration(
            &self.inner.loaded.calibration_default,
            opts.calibration.as_ref(),
        );
        let transitions = build_transition_matrix(&self.inner.loaded.label_strs, &cal);

        // ---- 6. Viterbi decode. ----
        let tags = viterbi_decode(&emit, &transitions);

        // Per-token confidence aligned to the Viterbi-emitted tag.
        //
        // We deliberately index `probs_flat` by `tags[t]` (the Viterbi
        // pick) rather than by the local argmax. At boundary tokens the
        // Viterbi tag can differ from the argmax tag (transition biases
        // win), so reporting the argmax-class probability would mix
        // semantics: the `tag` would come from Viterbi while the `score`
        // would come from a different class. Using the Viterbi tag's
        // softmax probability keeps both the per-token output and the
        // span-mean score consistent with the emitted tag sequence.
        let mut per_token_tag_probs: Vec<f32> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let tag_id = tags[t];
            per_token_tag_probs.push(probs_flat[t * num_classes + tag_id]);
        }

        // ---- 7. Extract spans. ----
        let entities_internal = extract_spans(
            &tags,
            &id2label,
            &per_token_tag_probs,
            &offsets,
            &text,
            threshold,
        );

        let entities: Vec<PrivacyEntity> = entities_internal
            .into_iter()
            .map(|e| PrivacyEntity {
                label: e.label,
                start: e.start as u32,
                end: e.end as u32,
                score: e.score as f64,
                text: e.text,
            })
            .collect();

        // ---- 8. Per-token output (optional). ----
        //
        // We report:
        // - `tag`:   the Viterbi-decoded BIOES tag (consistent with the
        //            sequence used to extract entities).
        // - `score`: the softmax probability of that Viterbi-emitted tag
        //            at this token — the same value that feeds into the
        //            span mean, so `tag` and `score` share semantics.
        // - `text`:  the byte slice from `source_text[start..end]`. For
        //            special tokens HF emits `(0, 0)`; we fall back to
        //            the tokenizer's `id_to_token` in that case so the
        //            output is never empty/ambiguous.
        let tokens = if return_tokens {
            let mut out = Vec::with_capacity(n_tokens);
            for t in 0..n_tokens {
                let (start, end) = offsets[t];
                let tok_text = if start == 0 && end == 0 {
                    // Special token or otherwise zero-width — fall back
                    // to the tokenizer's vocab string.
                    self.inner
                        .loaded
                        .tokenizer
                        .id_to_token(ids_u32[t])
                        .unwrap_or_default()
                } else {
                    text.get(start..end).unwrap_or("").to_string()
                };
                let tag_id = tags[t];
                out.push(PrivacyToken {
                    text: tok_text,
                    tag: id2label
                        .get(tag_id)
                        .cloned()
                        .unwrap_or_else(|| "O".to_string()),
                    score: per_token_tag_probs[t] as f64,
                    start: start as u32,
                    end: end as u32,
                });
            }
            Some(out)
        } else {
            None
        };

        Ok(PrivacyClassifyResult { entities, tokens })
    }

    /// Load a LoRA adapter from a directory containing `adapter.safetensors`
    /// and `adapter_config.json`. The adapter is attached to all layers'
    /// attention projections (q/k/v/o) and overrides the classifier head's
    /// `score.weight` / `score.bias`. Any existing adapter is cleared first.
    ///
    /// `adapter_config.json` must provide `rank` (i64), `alpha` (f32), and
    /// `dropout` (f32). `target_modules` must include all of `q_proj`,
    /// `k_proj`, `v_proj`, and `o_proj` — this is enforced at load time
    /// because the corresponding `lora_a` / `lora_b` tensors are looked up
    /// unconditionally for every layer/projection and a missing tensor
    /// raises a `missing key` error from
    /// [`crate::nn::lora::LoRALinear::from_safetensors`]. The
    /// `base_model_path` key is informational and may be absent.
    #[napi]
    pub fn load_adapter(&mut self, adapter_dir: String) -> Result<()> {
        // 1. Clear any previously loaded adapter so load is idempotent.
        self.clear_adapter()?;

        // 2. Parse adapter_config.json.
        let cfg_path = Path::new(&adapter_dir).join("adapter_config.json");
        let cfg_json = std::fs::read_to_string(&cfg_path).map_err(|e| {
            Error::from_reason(format!("failed to read {}: {e}", cfg_path.display()))
        })?;
        let cfg_val: serde_json::Value = serde_json::from_str(&cfg_json).map_err(|e| {
            Error::from_reason(format!("failed to parse {}: {e}", cfg_path.display()))
        })?;

        let rank = cfg_val
            .get("rank")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| Error::from_reason("adapter_config.json: 'rank' missing or not int"))?;
        let alpha = cfg_val
            .get("alpha")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                Error::from_reason("adapter_config.json: 'alpha' missing or not number")
            })? as f32;
        let dropout = cfg_val
            .get("dropout")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let lora_cfg = LoRAConfig {
            rank,
            alpha,
            dropout,
        };

        // 3. Load adapter.safetensors.
        let adapter_path = Path::new(&adapter_dir).join("adapter.safetensors");
        let tensors: HashMap<String, MxArray> =
            crate::utils::safetensors::load_safetensors_lazy(&adapter_path)?;

        // 4. Attach LoRA to each layer's attention projections.
        let num_layers = self.inner.loaded.weights.layers.len();
        for i in 0..num_layers {
            let layer = &mut self.inner.loaded.weights.layers[i];
            let attn = &mut layer.self_attn;

            let q_name = format!("model.layers.{i}.self_attn.q_proj");
            let k_name = format!("model.layers.{i}.self_attn.k_proj");
            let v_name = format!("model.layers.{i}.self_attn.v_proj");
            let o_name = format!("model.layers.{i}.self_attn.o_proj");

            *attn.q_proj.lora_mut() =
                Some(LoRALinear::from_safetensors(&tensors, &q_name, &lora_cfg)?);
            *attn.k_proj.lora_mut() =
                Some(LoRALinear::from_safetensors(&tensors, &k_name, &lora_cfg)?);
            *attn.v_proj.lora_mut() =
                Some(LoRALinear::from_safetensors(&tensors, &v_name, &lora_cfg)?);
            *attn.o_proj.lora_mut() =
                Some(LoRALinear::from_safetensors(&tensors, &o_name, &lora_cfg)?);
        }

        // 5. Override classifier head weights and back up originals.
        let new_score_weight = tensors
            .get("score.weight")
            .cloned()
            .ok_or_else(|| Error::from_reason("adapter.safetensors: missing 'score.weight'"))?;
        let new_score_bias = tensors
            .get("score.bias")
            .cloned()
            .ok_or_else(|| Error::from_reason("adapter.safetensors: missing 'score.bias'"))?;

        // Validate shapes match the model's classifier head.
        let old_sw_shape = self
            .inner
            .loaded
            .weights
            .score_weight
            .shape()?
            .as_ref()
            .to_vec();
        let new_sw_shape = new_score_weight.shape()?.as_ref().to_vec();
        if old_sw_shape != new_sw_shape {
            return Err(Error::from_reason(format!(
                "adapter score.weight shape mismatch: model has {:?}, adapter has {:?}",
                old_sw_shape, new_sw_shape
            )));
        }
        let old_sb_shape = self
            .inner
            .loaded
            .weights
            .score_bias
            .shape()?
            .as_ref()
            .to_vec();
        let new_sb_shape = new_score_bias.shape()?.as_ref().to_vec();
        if old_sb_shape != new_sb_shape {
            return Err(Error::from_reason(format!(
                "adapter score.bias shape mismatch: model has {:?}, adapter has {:?}",
                old_sb_shape, new_sb_shape
            )));
        }

        // Stash originals.
        let backup_w = self.inner.loaded.weights.score_weight.clone();
        let backup_b = self.inner.loaded.weights.score_bias.clone();
        self.inner.loaded.weights.score_weight_backup = Some(backup_w);
        self.inner.loaded.weights.score_bias_backup = Some(backup_b);

        // Install override.
        self.inner.loaded.weights.score_weight = new_score_weight;
        self.inner.loaded.weights.score_bias = new_score_bias;

        Ok(())
    }

    /// Drop all LoRA adapters and restore the original classifier weights.
    /// Idempotent — calling it on a model without an adapter is a no-op.
    #[napi]
    pub fn clear_adapter(&mut self) -> Result<()> {
        // Clear LoRA from every attention projection.
        for layer in &mut self.inner.loaded.weights.layers {
            let attn = &mut layer.self_attn;
            *attn.q_proj.lora_mut() = None;
            *attn.k_proj.lora_mut() = None;
            *attn.v_proj.lora_mut() = None;
            *attn.o_proj.lora_mut() = None;
        }

        // Restore classifier head from backups (if any).
        if let Some(backup) = self.inner.loaded.weights.score_weight_backup.take() {
            self.inner.loaded.weights.score_weight = backup;
        }
        if let Some(backup) = self.inner.loaded.weights.score_bias_backup.take() {
            self.inner.loaded.weights.score_bias = backup;
        }

        Ok(())
    }

    /// Bake the currently-loaded adapter into the base weights and write the
    /// result to `<out_dir>/model.safetensors`. Errors if any LoRA adapter
    /// sits on top of a quantized projection (dequantization is not supported
    /// in this phase). The in-memory model state is unchanged after this call.
    ///
    /// **Aux files are not copied.** Only `model.safetensors` is written.
    /// Callers must copy the following files from the base checkpoint
    /// directory into `out_dir` themselves (the fused output is otherwise
    /// not loadable as a standalone privacy-filter checkpoint):
    ///
    /// - `config.json`
    /// - `tokenizer.json`
    /// - `tokenizer_config.json`
    /// - `viterbi_calibration.json`
    ///
    /// This method only borrows `self` immutably — it reads the in-memory
    /// weights and serializes them; no mutation of the live model occurs.
    #[napi]
    pub fn fuse_adapter(&self, out_dir: String) -> Result<()> {
        let out_path = Path::new(&out_dir);
        std::fs::create_dir_all(out_path).map_err(|e| {
            Error::from_reason(format!(
                "failed to create output directory {}: {e}",
                out_path.display()
            ))
        })?;

        let weights = &self.inner.loaded.weights;
        let mut map: HashMap<String, MxArray> = HashMap::new();

        // Top-level weights.
        map.insert(
            "model.embed_tokens.weight".to_string(),
            weights.embed_tokens.clone(),
        );
        map.insert("model.norm.weight".to_string(), weights.final_norm.clone());
        // Classifier head: use current score_weight/score_bias (already overridden).
        map.insert("score.weight".to_string(), weights.score_weight.clone());
        map.insert("score.bias".to_string(), weights.score_bias.clone());

        // Per-layer weights.
        for (i, layer) in weights.layers.iter().enumerate() {
            let p = format!("model.layers.{i}");

            // Layer norms.
            map.insert(
                format!("{p}.input_layernorm.weight"),
                layer.input_layernorm.clone(),
            );
            map.insert(
                format!("{p}.post_attention_layernorm.weight"),
                layer.post_attention_layernorm.clone(),
            );

            // Attention sinks.
            map.insert(
                format!("{p}.self_attn.sinks"),
                layer.self_attn.sinks.clone(),
            );

            // Attention projections — fuse LoRA if attached.
            let proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"];
            let projs = [
                &layer.self_attn.q_proj,
                &layer.self_attn.k_proj,
                &layer.self_attn.v_proj,
                &layer.self_attn.o_proj,
            ];
            for (proj_name, proj) in proj_names.iter().zip(projs.iter()) {
                let prefix = format!("{p}.self_attn.{proj_name}");
                if let Some(lora) = proj.lora() {
                    if proj.is_quantized() {
                        return Err(Error::from_reason(format!(
                            "fuse_adapter: cannot fuse LoRA on quantized projection {prefix} \
                             (dequantization is not supported in Phase B — dequantize the \
                             checkpoint first)"
                        )));
                    }
                    let fused = lora.fuse_into_weight(proj.weight())?;
                    map.insert(format!("{prefix}.weight"), fused);
                } else {
                    map.insert(format!("{prefix}.weight"), proj.weight().clone());
                }
                // Bias (if present — always plain, never quantized companion).
                let bias_opt = match *proj {
                    LoadedProj::Plain { bias, .. } => bias.as_ref(),
                    LoadedProj::Quantized { bias, .. } => bias.as_ref(),
                };
                if let Some(b) = bias_opt {
                    map.insert(format!("{prefix}.bias"), b.clone());
                }
                // Quantization companions.
                if let LoadedProj::Quantized { scales, biases, .. } = *proj {
                    map.insert(format!("{prefix}.scales"), scales.clone());
                    if let Some(qb) = biases {
                        map.insert(format!("{prefix}.biases"), qb.clone());
                    }
                }
            }

            // Router.
            match &layer.mlp.router {
                LoadedRouter::Plain(router) => {
                    let router_prefix = format!("{p}.mlp.router");
                    map.insert(format!("{router_prefix}.weight"), router.weight.clone());
                    map.insert(format!("{router_prefix}.bias"), router.bias.clone());
                }
                LoadedRouter::Quantized { proj: rproj, .. } => {
                    let router_prefix = format!("{p}.mlp.router");
                    map.insert(format!("{router_prefix}.weight"), rproj.weight().clone());
                    if let LoadedProj::Quantized {
                        scales,
                        biases,
                        bias,
                        ..
                    } = rproj
                    {
                        map.insert(format!("{router_prefix}.scales"), scales.clone());
                        if let Some(qb) = biases {
                            map.insert(format!("{router_prefix}.biases"), qb.clone());
                        }
                        if let Some(b) = bias {
                            map.insert(format!("{router_prefix}.bias"), b.clone());
                        }
                    } else if let LoadedProj::Plain { bias: Some(b), .. } = rproj {
                        map.insert(format!("{router_prefix}.bias"), b.clone());
                    }
                }
            }

            // MoE expert weights (bare keys — no .weight suffix for plain tensors).
            let gate_up_prefix = format!("{p}.mlp.experts.gate_up_proj");
            let down_prefix = format!("{p}.mlp.experts.down_proj");

            match &layer.mlp.gate_up_proj {
                LoadedProj::Plain { weight, .. } => {
                    map.insert(gate_up_prefix.clone(), weight.clone());
                }
                LoadedProj::Quantized {
                    weight,
                    scales,
                    biases,
                    ..
                } => {
                    map.insert(format!("{gate_up_prefix}.weight"), weight.clone());
                    map.insert(format!("{gate_up_prefix}.scales"), scales.clone());
                    if let Some(qb) = biases {
                        map.insert(format!("{gate_up_prefix}.biases"), qb.clone());
                    }
                }
            }
            match &layer.mlp.down_proj {
                LoadedProj::Plain { weight, .. } => {
                    map.insert(down_prefix.clone(), weight.clone());
                }
                LoadedProj::Quantized {
                    weight,
                    scales,
                    biases,
                    ..
                } => {
                    map.insert(format!("{down_prefix}.weight"), weight.clone());
                    map.insert(format!("{down_prefix}.scales"), scales.clone());
                    if let Some(qb) = biases {
                        map.insert(format!("{down_prefix}.biases"), qb.clone());
                    }
                }
            }
            map.insert(
                format!("{p}.mlp.experts.gate_up_proj_bias"),
                layer.mlp.gate_up_bias.clone(),
            );
            map.insert(
                format!("{p}.mlp.experts.down_proj_bias"),
                layer.mlp.down_bias.clone(),
            );
        }

        let out_file = out_path.join("model.safetensors");
        crate::utils::safetensors::save_safetensors(&out_file, &map, None)?;
        Ok(())
    }
}

/// Merge per-call calibration overrides on top of the model's default
/// calibration. Any field left `None` in `overrides` is taken from
/// `default`.
fn merge_calibration(default: &Calibration, overrides: Option<&PrivacyCalibration>) -> Calibration {
    let Some(o) = overrides else {
        return *default;
    };
    Calibration {
        transition_bias_background_stay: o
            .transition_bias_background_stay
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_background_stay),
        transition_bias_background_to_start: o
            .transition_bias_background_to_start
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_background_to_start),
        transition_bias_end_to_background: o
            .transition_bias_end_to_background
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_end_to_background),
        transition_bias_end_to_start: o
            .transition_bias_end_to_start
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_end_to_start),
        transition_bias_inside_to_continue: o
            .transition_bias_inside_to_continue
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_inside_to_continue),
        transition_bias_inside_to_end: o
            .transition_bias_inside_to_end
            .map(|v| v as f32)
            .unwrap_or(default.transition_bias_inside_to_end),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_calibration_uses_default_when_no_overrides() {
        let default = Calibration {
            transition_bias_background_stay: 1.5,
            transition_bias_inside_to_end: -0.25,
            ..Calibration::default()
        };
        let merged = merge_calibration(&default, None);
        assert_eq!(merged.transition_bias_background_stay, 1.5);
        assert_eq!(merged.transition_bias_inside_to_end, -0.25);
    }

    #[test]
    fn merge_calibration_overrides_per_field() {
        let default = Calibration {
            transition_bias_background_stay: 1.5,
            transition_bias_inside_to_end: -0.25,
            ..Calibration::default()
        };
        let overrides = PrivacyCalibration {
            transition_bias_background_stay: Some(0.0),
            transition_bias_background_to_start: None,
            transition_bias_end_to_background: None,
            transition_bias_end_to_start: None,
            transition_bias_inside_to_continue: Some(2.5),
            transition_bias_inside_to_end: None,
        };
        let merged = merge_calibration(&default, Some(&overrides));
        // Overridden:
        assert_eq!(merged.transition_bias_background_stay, 0.0);
        assert_eq!(merged.transition_bias_inside_to_continue, 2.5);
        // Inherited from default:
        assert_eq!(merged.transition_bias_inside_to_end, -0.25);
        // Inherited zero-default:
        assert_eq!(merged.transition_bias_end_to_start, 0.0);
    }
}
