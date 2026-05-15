//! End-to-end forward pass for the OpenAI Privacy Filter model.
//!
//! Wires the 8 transformer [`Block`]s together with the input embedding,
//! the final RMSNorm, and the classifier head to produce per-token
//! logits ready for Viterbi decoding.
//!
//! ```text
//!   input_ids [B, T]
//!         │
//!         ├── embed_tokens[input_ids]   → hidden [B, T, 640]
//!         │
//!         ├── for layer in 0..8:
//!         │     Block { input_layernorm → AttentionLayer
//!         │             post_attention_layernorm → GptOssMlp }
//!         │
//!         ├── final RMSNorm (model.norm)
//!         └── classifier head (score.weight / score.bias)
//!                                       → logits [B, T, 33]
//! ```
//!
//! Shared per-model state (YaRN frequencies) is computed once at load
//! time and threaded through every block as a `&MxArray`. The runtime
//! call sequence is identical to mlx-lm's reference forward pass — see
//! `mlx-lm/mlx_lm/models/gpt_oss.py` — modulo bidirectional banded
//! attention (no KV cache) and the per-token classifier output instead
//! of an LM head.

use crate::array::MxArray;
use crate::nn::RMSNorm;
use napi::bindgen_prelude::*;
use std::path::Path;

use super::classifier::classifier_forward;
use super::persistence::{LoadedModel, load_from_directory};
use super::transformer::Block;
use super::yarn::compute_yarn_freqs;

/// A loaded privacy-filter model plus its precomputed YaRN frequencies.
///
/// The frequencies depend only on `head_dim` and `rope_parameters` so we
/// build them once at load time and share the resulting `[head_dim/2]`
/// f32 array across all 8 attention layers.
pub struct PrivacyFilterModel {
    pub loaded: LoadedModel,
    /// Shared YaRN frequencies (`[head_dim / 2]`, f32).
    pub yarn_freqs: MxArray,
}

impl PrivacyFilterModel {
    /// Load a privacy-filter checkpoint from a directory and precompute
    /// the shared YaRN frequencies.
    pub fn load_from_dir(path: &Path) -> Result<Self> {
        let loaded = load_from_directory(path)?;
        let yarn_freqs =
            compute_yarn_freqs(loaded.config.head_dim, &loaded.config.rope_parameters)?;
        Ok(Self { loaded, yarn_freqs })
    }

    /// Run the full forward pass and return per-token logits.
    ///
    /// - `input_ids` shape `[B, T]`, dtype int32 / int64 / uint32 (anything
    ///   `mx.take` accepts as an integer index).
    /// - Output shape `[B, T, num_classes]` (33 for the shipped
    ///   privacy-filter checkpoint).
    /// - Output dtype is the model's native weight dtype (bf16 for the
    ///   shipped checkpoint); callers wanting f32 should `astype` the
    ///   result themselves.
    pub fn forward_logits(&self, input_ids: &MxArray) -> Result<MxArray> {
        self.forward_logits_training(input_ids, false)
    }

    /// Training-mode variant of [`Self::forward_logits`].
    ///
    /// Identical to `forward_logits` in inference mode, but threads a
    /// `training` flag through the attention layers so LoRA dropout
    /// fires when an adapter with `dropout > 0` is attached.
    pub fn forward_logits_training(&self, input_ids: &MxArray, training: bool) -> Result<MxArray> {
        let weights = &self.loaded.weights;
        let cfg = &self.loaded.config;

        // 1. Embedding lookup: hidden = embed_tokens[input_ids].
        let mut hidden = weights.embed_tokens.take(input_ids, 0)?;

        // 2. Run all transformer blocks.
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let block = Block {
                weights: layer,
                config: cfg,
                yarn_freqs: &self.yarn_freqs,
                layer_idx,
            };
            hidden = block.forward(&hidden, training)?;
        }

        // 3. Final RMSNorm (model.norm).
        let final_norm = RMSNorm::from_weight(&weights.final_norm, Some(cfg.rms_norm_eps as f64))?;
        let hidden = final_norm.forward(&hidden)?;

        // 4. Classifier head — `score.weight` `[33, hidden]`,
        //    `score.bias` `[33]`. See [`classifier_forward`].
        classifier_forward(&hidden, &weights.score_weight, &weights.score_bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;

    fn checkpoint_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter")
    }

    /// Shape + finiteness check on a tiny real-tokenized input.
    /// Guards against accidental shape regressions and any NaN/Inf
    /// escaping the 8-block pipeline.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn forward_logits_shape_and_finite() {
        let model = PrivacyFilterModel::load_from_dir(&checkpoint_dir()).expect("load model");

        let text = "Hi I am Alice.";
        let ids_u32 = model
            .loaded
            .tokenizer
            .encode_sync(text, Some(false))
            .expect("encode");
        let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
        let t = ids.len() as i64;
        let input_ids = MxArray::from_int32(&ids, &[1, t]).expect("from_int32");

        let logits = model.forward_logits(&input_ids).expect("forward");
        let shape: Vec<i64> = logits.shape().unwrap().to_vec();
        assert_eq!(shape, vec![1, t, model.loaded.label_strs.len() as i64]);

        // Finiteness check via GPU reduction (promote to f32 first so
        // bf16's wider exponent range doesn't paper over an actual Inf).
        let logits_f32 = logits.astype(DType::Float32).expect("astype");
        assert!(
            !logits_f32.has_nan_or_inf().expect("has_nan_or_inf"),
            "forward_logits produced NaN or Inf"
        );
    }

    /// Tokenize `text`, run the forward pass through `model`, and
    /// return per-token (string, argmax_tag) pairs. Shared by the
    /// canonical Alice helper and the multi-fixture cross-mode parity
    /// test so every checkpoint sees byte-for-byte identical
    /// tokenization + decoding.
    fn argmax_tags_for_text(model: &PrivacyFilterModel, text: &str) -> (Vec<String>, Vec<String>) {
        let ids_u32 = model
            .loaded
            .tokenizer
            .encode_sync(text, Some(false))
            .expect("encode");
        let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
        let t = ids.len() as i64;
        let input_ids = MxArray::from_int32(&ids, &[1, t]).expect("from_int32");
        let logits = model.forward_logits(&input_ids).expect("forward");
        let pred = logits.argmax(-1, Some(false)).expect("argmax");
        let pred_i32 = pred.to_int32().expect("to_int32");
        let tok_strs: Vec<String> = ids_u32
            .iter()
            .map(|&id| {
                model
                    .loaded
                    .tokenizer
                    .decode_sync(&[id], false)
                    .unwrap_or_default()
            })
            .collect();
        let pred_tags: Vec<String> = pred_i32
            .iter()
            .map(|&i| model.loaded.label_strs[i as usize].clone())
            .collect();
        (tok_strs, pred_tags)
    }

    /// Run the same `forward_classify_alice_smith_email` body against
    /// an arbitrary checkpoint directory and return the (tokens, tags)
    /// pair. Shared by the bf16 reference and quantized variants so
    /// they all use byte-for-byte identical tokenization + decoding.
    fn run_classify_alice_smith_email(ckpt: &std::path::Path) -> (Vec<String>, Vec<String>) {
        let model = PrivacyFilterModel::load_from_dir(ckpt).expect("load model");
        argmax_tags_for_text(&model, "Hi I am Alice Smith, email alice@example.com")
    }

    /// The named-entity positions in the canonical test sentence.
    /// Quantized variants are allowed per-token confidence boundary
    /// noise OUTSIDE this set but must label these positions exactly
    /// as the bf16 reference does (per the Phase C correctness gate).
    const ALICE_NER_POSITIONS: &[(usize, &str)] = &[
        (3, "B-private_person"),
        (4, "E-private_person"),
        (7, "B-private_email"),
        (8, "I-private_email"),
        (9, "E-private_email"),
    ];

    /// End-to-end correctness: classify a PII-laden sentence and verify
    /// that the predicted tag sequence contains the expected entity
    /// starts (`B-private_person` for the name, `B-private_email` for
    /// the email).
    ///
    /// On failure the test prints the actual tokens and tags so the
    /// caller can diagnose whether the bug lives in YaRN, attention
    /// sinks, MoE routing, etc. **Do not loosen this assertion** — a
    /// failure here is a real algorithmic bug.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn forward_classify_alice_smith_email() {
        let model = PrivacyFilterModel::load_from_dir(&checkpoint_dir()).expect("load model");

        let text = "Hi I am Alice Smith, email alice@example.com";
        let ids_u32 = model
            .loaded
            .tokenizer
            .encode_sync(text, Some(false))
            .expect("encode");
        let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
        let t = ids.len() as i64;
        let input_ids = MxArray::from_int32(&ids, &[1, t]).expect("from_int32");

        let logits = model.forward_logits(&input_ids).expect("forward");

        // argmax over the last axis. `argmax(axis=-1)` reduces the
        // `num_classes` axis, leaving `[1, T]` int indices.
        let pred = logits.argmax(-1, Some(false)).expect("argmax");
        let pred_i32 = pred.to_int32().expect("to_int32");

        // Decode each token id back to a string for the debug print.
        let tok_strs: Vec<String> = ids_u32
            .iter()
            .map(|&id| {
                model
                    .loaded
                    .tokenizer
                    .decode_sync(&[id], false)
                    .unwrap_or_default()
            })
            .collect();

        let pred_tags: Vec<&str> = pred_i32
            .iter()
            .map(|&i| model.loaded.label_strs[i as usize].as_str())
            .collect();

        println!("Tokens: {:?}", tok_strs);
        println!("Tags:   {:?}", pred_tags);

        assert!(
            pred_tags.contains(&"B-private_person"),
            "expected B-private_person in tag sequence; got {:?}",
            pred_tags
        );
        assert!(
            pred_tags.contains(&"B-private_email"),
            "expected B-private_email in tag sequence; got {:?}",
            pred_tags
        );
    }

    // ---- LoRA Phase B tests ---- //

    /// Helper: attach a synthetic zero-B adapter (all attention projections,
    /// score head uses original weights) and run classify. Returns the
    /// per-token softmax probabilities as a flat Vec<f32>.
    fn attach_zero_adapter_and_classify(
        model: &mut PrivacyFilterModel,
        text: &str,
        rank: i64,
    ) -> Vec<f32> {
        use crate::array::DType;
        use crate::nn::lora::{LoRAConfig, LoRALinear};

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

        // Use the original score_weight / score_bias as the override —
        // so the head is unchanged.
        let orig_sw = model.loaded.weights.score_weight.clone();
        let orig_sb = model.loaded.weights.score_bias.clone();

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

        // Stash backups and install identity override for classifier head.
        model.loaded.weights.score_weight_backup = Some(model.loaded.weights.score_weight.clone());
        model.loaded.weights.score_bias_backup = Some(model.loaded.weights.score_bias.clone());
        model.loaded.weights.score_weight = orig_sw;
        model.loaded.weights.score_bias = orig_sb;

        // Run classify.
        let ids_u32 = model
            .loaded
            .tokenizer
            .encode_sync(text, Some(false))
            .unwrap();
        let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
        let n = ids.len() as i64;
        let input_ids = MxArray::from_int32(&ids, &[1, n]).unwrap();
        let logits = model.forward_logits(&input_ids).unwrap();
        let logits_f32 = logits.astype(DType::Float32).unwrap();
        let probs = crate::nn::Activations::softmax(&logits_f32, Some(-1)).unwrap();
        probs.eval();
        probs.to_float32().unwrap().to_vec()
    }

    /// With a zero-B LoRA adapter attached, the model forward pass should
    /// produce output bit-exactly (or near-bit-exactly) equal to the base
    /// model. Max-abs-diff < 1e-5 in softmax probability space is the gate.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn forward_parity_with_zero_adapter() {
        use crate::array::DType;
        use crate::nn::Activations;

        let text = "Hi I am Alice Smith, email alice@example.com.";
        let ckpt = checkpoint_dir();

        // Base model probs (no adapter).
        let base_probs = {
            let model = PrivacyFilterModel::load_from_dir(&ckpt).unwrap();
            let ids_u32 = model
                .loaded
                .tokenizer
                .encode_sync(text, Some(false))
                .unwrap();
            let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
            let n = ids.len() as i64;
            let input_ids = MxArray::from_int32(&ids, &[1, n]).unwrap();
            let logits = model.forward_logits(&input_ids).unwrap();
            let logits_f32 = logits.astype(DType::Float32).unwrap();
            let probs = Activations::softmax(&logits_f32, Some(-1)).unwrap();
            probs.eval();
            probs.to_float32().unwrap().to_vec()
        };

        // Adapter model probs (zero-B adapter).
        let adapter_probs = {
            let mut model = PrivacyFilterModel::load_from_dir(&ckpt).unwrap();
            attach_zero_adapter_and_classify(&mut model, text, 16)
        };

        assert_eq!(
            base_probs.len(),
            adapter_probs.len(),
            "prob vector length mismatch"
        );

        let max_diff = base_probs
            .iter()
            .zip(adapter_probs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "zero-B adapter changed softmax probs by {max_diff:.2e} (expected < 1e-5)"
        );
    }

    /// Save a synthetic adapter with known A weights to disk, load it via
    /// `load_adapter`, and verify the in-memory adapter weights match.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn adapter_save_load_roundtrip() {
        use crate::array::DType;
        use crate::nn::lora::{LoRAConfig, LoRALinear};
        use crate::utils::safetensors::save_safetensors;
        use std::collections::HashMap;

        let rank = 8i64;
        let cfg = LoRAConfig {
            rank,
            alpha: 16.0,
            dropout: 0.05,
        };

        let ckpt = checkpoint_dir();
        let model_ref = PrivacyFilterModel::load_from_dir(&ckpt).unwrap();
        let num_layers = model_ref.loaded.config.num_hidden_layers;
        let hidden = model_ref.loaded.config.hidden_size as i64;
        let num_q = model_ref.loaded.config.num_attention_heads as i64;
        let num_kv = model_ref.loaded.config.num_key_value_heads as i64;
        let head_dim = model_ref.loaded.config.head_dim as i64;
        let q_out = num_q * head_dim;
        let kv_out = num_kv * head_dim;
        let num_classes = model_ref.loaded.label_strs.len() as i64;
        drop(model_ref);

        // Build a synthetic adapter tensor map with deterministic A values.
        let mut tensors: HashMap<String, MxArray> = HashMap::new();
        let proj_specs: &[(&str, i64, i64)] = &[
            ("q_proj", hidden, q_out),
            ("k_proj", hidden, kv_out),
            ("v_proj", hidden, kv_out),
            ("o_proj", q_out, hidden),
        ];
        // We'll check layer 0 q_proj A after loading.
        let q0_a_data: Vec<f32> = (0..hidden * rank).map(|i| (i as f32) * 1e-4).collect();
        let q0_a_ref = MxArray::from_float32(&q0_a_data, &[hidden, rank]).unwrap();

        for i in 0..num_layers {
            for &(proj_name, in_dim, out_dim) in proj_specs {
                let name = format!("model.layers.{i}.self_attn.{proj_name}");
                let mut adapter =
                    LoRALinear::new_zero_b(in_dim, out_dim, &cfg, &name, DType::Float32).unwrap();
                if i == 0 && proj_name == "q_proj" {
                    adapter
                        .set_weights(
                            q0_a_ref.clone(),
                            MxArray::zeros(&[rank, out_dim], Some(DType::Float32)).unwrap(),
                        )
                        .unwrap();
                }
                for (key, arr) in adapter.params() {
                    tensors.insert(key.to_owned(), arr.clone());
                }
            }
        }
        // Classifier override: all-zero score.weight / score.bias for simplicity.
        tensors.insert(
            "score.weight".to_owned(),
            MxArray::zeros(&[num_classes, hidden], Some(DType::Float32)).unwrap(),
        );
        tensors.insert(
            "score.bias".to_owned(),
            MxArray::zeros(&[num_classes], Some(DType::Float32)).unwrap(),
        );

        // Write to tempdir.
        let tmp = std::env::temp_dir().join(format!(
            "mlx_lora_roundtrip_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let adapter_st = tmp.join("adapter.safetensors");
        save_safetensors(&adapter_st, &tensors, None).unwrap();

        // Write adapter_config.json.
        let adapter_cfg_json = serde_json::json!({
            "rank": rank,
            "alpha": cfg.alpha,
            "dropout": cfg.dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "base_model_path": ckpt.display().to_string()
        });
        std::fs::write(
            tmp.join("adapter_config.json"),
            serde_json::to_string_pretty(&adapter_cfg_json).unwrap(),
        )
        .unwrap();

        // Load adapter.
        let mut js_model =
            crate::models::privacy_filter::PrivacyFilterModelJs::load(ckpt.display().to_string())
                .unwrap();
        js_model
            .load_adapter(tmp.display().to_string())
            .expect("load_adapter");

        // Verify layer 0 q_proj A matches.
        let loaded_a = js_model.inner.loaded.weights.layers[0]
            .self_attn
            .q_proj
            .lora()
            .expect("q_proj should have LoRA")
            .a();
        let loaded_a_f32 = loaded_a
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        let ref_a_f32 = q0_a_ref.to_float32().unwrap().to_vec();
        assert_eq!(loaded_a_f32.len(), ref_a_f32.len());
        let max_diff = loaded_a_f32
            .iter()
            .zip(ref_a_f32.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "q_proj A weight mismatch after roundtrip: max_diff={max_diff:.2e}"
        );

        // Verify score_weight was overridden (should be zeros now).
        let sw = &js_model.inner.loaded.weights.score_weight;
        let sw_vals = sw
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        for &v in &sw_vals {
            assert!(
                v.abs() < 1e-6,
                "score_weight should be zeros after adapter load, got {v}"
            );
        }

        // Cleanup.
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Load the model, attach a small random adapter, run fuse_adapter to a
    /// tempdir, reload from that tempdir, and verify classify output matches
    /// (within 5e-3 in softmax probability).
    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn fuse_then_forward_matches_unfused() {
        use crate::array::DType;
        use crate::nn::Activations;
        use crate::nn::lora::{LoRAConfig, LoRALinear};
        use std::collections::HashMap;

        let text = "Hi I am Alice Smith, email alice@example.com.";
        let ckpt = checkpoint_dir();

        let rank = 4i64;
        let cfg = LoRAConfig {
            rank,
            alpha: 8.0,
            dropout: 0.0,
        };

        // Build adapter tensors with tiny random A values (small perturbation).
        let num_layers;
        let hidden;
        let q_out;
        let kv_out;
        let num_classes;
        {
            let model_ref = PrivacyFilterModel::load_from_dir(&ckpt).unwrap();
            num_layers = model_ref.loaded.config.num_hidden_layers;
            hidden = model_ref.loaded.config.hidden_size as i64;
            let num_q = model_ref.loaded.config.num_attention_heads as i64;
            let num_kv = model_ref.loaded.config.num_key_value_heads as i64;
            let head_dim = model_ref.loaded.config.head_dim as i64;
            q_out = num_q * head_dim;
            kv_out = num_kv * head_dim;
            num_classes = model_ref.loaded.label_strs.len() as i64;
        }

        let proj_specs: &[(&str, i64, i64)] = &[
            ("q_proj", hidden, q_out),
            ("k_proj", hidden, kv_out),
            ("v_proj", hidden, kv_out),
            ("o_proj", q_out, hidden),
        ];

        let mut tensors: HashMap<String, MxArray> = HashMap::new();
        for i in 0..num_layers {
            for &(proj_name, in_dim, out_dim) in proj_specs {
                let name = format!("model.layers.{i}.self_attn.{proj_name}");
                // Use small uniform random A, zero B — still a legit zero-delta adapter
                // but with a non-trivial A matrix for the roundtrip fidelity check.
                let small_a_data: Vec<f32> = (0..in_dim * rank)
                    .map(|k| ((k as f32 * 1.1 + i as f32) * 1e-6).sin() * 0.01)
                    .collect();
                let mut adapter =
                    LoRALinear::new_zero_b(in_dim, out_dim, &cfg, &name, DType::Float32).unwrap();
                adapter
                    .set_weights(
                        MxArray::from_float32(&small_a_data, &[in_dim, rank]).unwrap(),
                        MxArray::zeros(&[rank, out_dim], Some(DType::Float32)).unwrap(),
                    )
                    .unwrap();
                for (key, arr) in adapter.params() {
                    tensors.insert(key.to_owned(), arr.clone());
                }
            }
        }
        // Classifier override = identity (load original weights).
        let model_for_sw = PrivacyFilterModel::load_from_dir(&ckpt).unwrap();
        tensors.insert(
            "score.weight".to_owned(),
            model_for_sw.loaded.weights.score_weight.clone(),
        );
        tensors.insert(
            "score.bias".to_owned(),
            model_for_sw.loaded.weights.score_bias.clone(),
        );
        drop(model_for_sw);

        // Create tempdir for adapter and fused output.
        let tmp = std::env::temp_dir().join(format!(
            "mlx_lora_fuse_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let adapter_dir = tmp.join("adapter");
        let fused_dir = tmp.join("fused");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        std::fs::create_dir_all(&fused_dir).unwrap();

        // Save adapter.
        crate::utils::safetensors::save_safetensors(
            adapter_dir.join("adapter.safetensors"),
            &tensors,
            None,
        )
        .unwrap();
        let adapter_cfg_json = serde_json::json!({
            "rank": rank,
            "alpha": cfg.alpha,
            "dropout": cfg.dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        });
        std::fs::write(
            adapter_dir.join("adapter_config.json"),
            serde_json::to_string_pretty(&adapter_cfg_json).unwrap(),
        )
        .unwrap();

        // Copy config files to fused_dir so it can be loaded.
        for fname in &[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "viterbi_calibration.json",
        ] {
            let src = ckpt.join(fname);
            if src.exists() {
                std::fs::copy(&src, fused_dir.join(fname)).unwrap();
            }
        }

        // Load model, attach adapter, classify (unfused).
        let mut js_model =
            crate::models::privacy_filter::PrivacyFilterModelJs::load(ckpt.display().to_string())
                .unwrap();
        js_model
            .load_adapter(adapter_dir.display().to_string())
            .expect("load_adapter");

        let unfused_probs = {
            let ids_u32 = js_model
                .inner
                .loaded
                .tokenizer
                .encode_sync(text, Some(false))
                .unwrap();
            let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
            let n = ids.len() as i64;
            let input_ids = MxArray::from_int32(&ids, &[1, n]).unwrap();
            let logits = js_model.inner.forward_logits(&input_ids).unwrap();
            let logits_f32 = logits.astype(DType::Float32).unwrap();
            let probs = Activations::softmax(&logits_f32, Some(-1)).unwrap();
            probs.eval();
            probs.to_float32().unwrap().to_vec()
        };
        let n_tokens = unfused_probs.len() / num_classes as usize;

        // Fuse adapter.
        js_model
            .fuse_adapter(fused_dir.display().to_string())
            .expect("fuse_adapter");

        // Load fused model and classify.
        let fused_probs = {
            let fused_model =
                PrivacyFilterModel::load_from_dir(&fused_dir).expect("load fused model");
            let ids_u32 = fused_model
                .loaded
                .tokenizer
                .encode_sync(text, Some(false))
                .unwrap();
            let ids: Vec<i32> = ids_u32.iter().map(|&id| id as i32).collect();
            let n = ids.len() as i64;
            let input_ids = MxArray::from_int32(&ids, &[1, n]).unwrap();
            let logits = fused_model.forward_logits(&input_ids).unwrap();
            let logits_f32 = logits.astype(DType::Float32).unwrap();
            let probs = Activations::softmax(&logits_f32, Some(-1)).unwrap();
            probs.eval();
            probs.to_float32().unwrap().to_vec()
        };

        assert_eq!(
            unfused_probs.len(),
            fused_probs.len(),
            "prob vector length mismatch"
        );

        // Since B=0 the fused and unfused outputs should be identical to the base
        // model; allow 5e-3 tolerance for any floating-point rounding in the fuse.
        let max_diff = unfused_probs
            .iter()
            .zip(fused_probs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("fuse_then_forward: n_tokens={n_tokens}, max_diff={max_diff:.2e}");
        assert!(
            max_diff < 5e-3,
            "fused model softmax probs differ from unfused by {max_diff:.2e} (threshold 5e-3)"
        );

        // Cleanup.
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Run the canonical PII sentence through a quantized variant and
    /// assert the named-entity positions still match the bf16
    /// reference. Per-token confidence boundary noise on
    /// background-token positions is allowed; label flips at the
    /// NER positions are not.
    fn assert_quantized_ner_matches(variant: &str) {
        let ckpt = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(format!(".cache/models/privacy-filter-{variant}"));
        if !ckpt.exists() {
            eprintln!(
                "skipping quantized {variant} test — checkpoint not present at {}",
                ckpt.display()
            );
            return;
        }
        let (tokens, tags) = run_classify_alice_smith_email(&ckpt);
        println!("[{variant}] Tokens: {:?}", tokens);
        println!("[{variant}] Tags:   {:?}", tags);
        for &(pos, expected) in ALICE_NER_POSITIONS {
            assert_eq!(
                tags.get(pos).map(String::as_str),
                Some(expected),
                "[{variant}] tag at NER position {pos} flipped: expected {expected}, got {:?}",
                tags.get(pos)
            );
        }
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter-mxfp8 — run with --include-ignored"]
    fn forward_classify_alice_smith_email_mxfp8() {
        assert_quantized_ner_matches("mxfp8");
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter-mxfp4 — run with --include-ignored"]
    fn forward_classify_alice_smith_email_mxfp4() {
        assert_quantized_ner_matches("mxfp4");
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter-nvfp4 — run with --include-ignored"]
    fn forward_classify_alice_smith_email_nvfp4() {
        assert_quantized_ner_matches("nvfp4");
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter-affine — run with --include-ignored"]
    fn forward_classify_alice_smith_email_affine() {
        assert_quantized_ner_matches("affine");
    }

    /// Five-fixture cross-mode parity sweep.
    ///
    /// The bf16 checkpoint produces the reference argmax tag sequence
    /// for each of five PII-laden inputs spanning all eight label
    /// classes (person, email, phone, address, date, url, account,
    /// secret). Every quantized variant is then loaded in turn and
    /// must produce **the same tag** at every position where the bf16
    /// reference predicted a non-`O` tag. Per-token boundary noise on
    /// background positions is allowed; label flips at entity
    /// positions are not.
    ///
    /// This is a stronger version of the per-mode single-input
    /// `forward_classify_alice_smith_email_<mode>` tests above. It
    /// runs five inputs and compares the argmax decision (not Viterbi
    /// output) because raw argmax is the simpler invariant to verify
    /// — the decoder operates downstream on whatever logits the
    /// quantized forward emits, so quantization-induced tag flips
    /// will already show up here.
    #[test]
    #[ignore = "requires .cache/models/privacy-filter-{mxfp4,mxfp8,nvfp4,affine} — run with --include-ignored"]
    fn parity_across_quantized_modes() {
        const INPUTS: &[&str] = &[
            "Hi I am Alice Smith, email alice@example.com",
            "Call me at +1 555 123 4567 anytime",
            "Ship the package to 742 Evergreen Terrace, Springfield",
            "Born 1990-03-14, profile at https://example.org/profile/jdoe",
            "Account 1234567890; api key sk-ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ",
        ];

        let cache_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models");

        // Reference pass: bf16 argmax tags for each input.
        let bf16_dir = cache_dir.join("privacy-filter");
        if !bf16_dir.exists() {
            eprintln!(
                "skipping cross-mode parity — bf16 checkpoint missing at {}",
                bf16_dir.display()
            );
            return;
        }
        let bf16 = PrivacyFilterModel::load_from_dir(&bf16_dir).expect("load bf16");
        let mut bf16_tags: Vec<Vec<String>> = Vec::with_capacity(INPUTS.len());
        for &text in INPUTS {
            let (_tokens, tags) = argmax_tags_for_text(&bf16, text);
            bf16_tags.push(tags);
        }
        // Drop the bf16 model to free its weights before loading the
        // next checkpoint — keeps peak footprint bounded on a 36GB box.
        drop(bf16);

        let variants = ["mxfp8", "mxfp4", "nvfp4", "affine"];
        // Collect all divergences across every (variant, input) pair so a
        // single test failure reports the full picture instead of bailing
        // on the first mode that flips. Each report line stringifies one
        // (variant, input_idx, position, token, bf16_tag, quant_tag).
        let mut all_failures: Vec<String> = Vec::new();
        for variant in variants {
            let ckpt = cache_dir.join(format!("privacy-filter-{variant}"));
            if !ckpt.exists() {
                eprintln!(
                    "skipping cross-mode parity for {variant} — checkpoint missing at {}",
                    ckpt.display()
                );
                continue;
            }
            let model = PrivacyFilterModel::load_from_dir(&ckpt)
                .unwrap_or_else(|e| panic!("load {variant}: {e:?}"));
            for (i, &text) in INPUTS.iter().enumerate() {
                let (tokens, tags) = argmax_tags_for_text(&model, text);
                let mut mismatches: Vec<(usize, &str, &str, &str)> = Vec::new();
                for (j, (b, q)) in bf16_tags[i].iter().zip(tags.iter()).enumerate() {
                    if b != "O" && b != q {
                        let tok = tokens.get(j).map(String::as_str).unwrap_or("");
                        mismatches.push((j, tok, b.as_str(), q.as_str()));
                    }
                }
                if !mismatches.is_empty() {
                    all_failures.push(format!(
                        "[{variant}] input[{i}]={text:?}: entity-position tag flips vs bf16: \
                         {mismatches:?}\n  bf16 = {:?}\n  {variant} = {:?}",
                        bf16_tags[i], tags,
                    ));
                }
            }
        }
        assert!(
            all_failures.is_empty(),
            "cross-mode parity divergences found:\n{}",
            all_failures.join("\n\n"),
        );
    }
}
