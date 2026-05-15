use crate::array::{DType, MxArray};
use napi::bindgen_prelude::*;
use std::collections::HashMap;

/// Configuration for a LoRA adapter.
///
/// Typical values for privacy-filter fine-tuning:
/// - rank: 16, alpha: 32.0, dropout: 0.05
#[derive(Clone, Copy, Debug)]
pub struct LoRAConfig {
    /// Low-rank bottleneck dimension
    pub rank: i64,
    /// Scaling factor; effective scale = alpha / rank
    pub alpha: f32,
    /// Dropout probability applied to input before the adapter path
    pub dropout: f32,
}

impl LoRAConfig {
    /// Effective scale = alpha / rank.
    /// Mirrors mlx-lm's `scaling` field (`lora.py:64`).
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
        }
    }
}

/// Rank-decomposed LoRA adapter for a single linear projection.
///
/// Layout follows the HF convention: `base_w` is `[out, in]`.
/// - `a`: `[in_features, rank]` — "up" matrix, LeCun-uniform initialized
/// - `b`: `[rank, out_features]` — "down" matrix, zero-initialized
///
/// With `b = 0`, `delta(x)` is exactly zero at initialization, so the
/// adapter is a no-op before training begins.
pub struct LoRALinear {
    /// [in_features, rank] — first adapter matrix
    a: MxArray,
    /// [rank, out_features] — second adapter matrix (zero-initialized)
    b: MxArray,
    /// alpha / rank — precomputed to avoid repeated division
    pub scale: f32,
    /// Dropout probability (0.0 disables dropout entirely)
    pub dropout: f32,
    /// Fully-qualified weight name, e.g. "model.layers.0.self_attn.q_proj"
    pub name: String,
    /// Precomputed safetensors key for `a`: "{name}.lora_a"
    key_a: String,
    /// Precomputed safetensors key for `b`: "{name}.lora_b"
    key_b: String,
}

impl LoRALinear {
    /// Construct a fresh zero-output adapter.
    ///
    /// `a` is drawn from `Uniform(±1/√in_features)` (LeCun-uniform, matches mlx-lm),
    /// ensuring reasonable gradient flow at the start of training.
    /// `b` is zeros, so `A @ B = 0` and the adapter contributes nothing until
    /// the first gradient step.
    ///
    /// The `_base_dtype` argument names the dtype of the *base* linear weight
    /// the adapter is attached to. It is intentionally ignored when allocating
    /// `a` / `b`: per standard PEFT practice (mlx-lm, HuggingFace PEFT), LoRA
    /// matrices are always stored in `fp32` regardless of the base model's
    /// dtype. This prevents AdamW updates of magnitude `~lr * grad` (often
    /// well below bf16 epsilon ≈ 4e-3) from silently rounding to zero at
    /// write-back time. The `delta()` forward already casts the result back
    /// to `x.dtype()` so the residual stream stays in the base dtype.
    /// The argument is kept in the signature for backwards-compatibility
    /// with existing call sites and for documentation value at the call site.
    pub fn new_zero_b(
        in_features: i64,
        out_features: i64,
        cfg: &LoRAConfig,
        name: &str,
        _base_dtype: DType,
    ) -> Result<Self> {
        // LoRA storage is always f32 — see doc-comment above.
        let dtype = DType::Float32;
        let limit = 1.0 / (in_features as f64).sqrt();
        let a = MxArray::random_uniform(&[in_features, cfg.rank], -limit, limit, Some(dtype))?;
        let b = MxArray::zeros(&[cfg.rank, out_features], Some(dtype))?;
        Ok(Self {
            a,
            b,
            scale: cfg.scale(),
            dropout: cfg.dropout,
            key_a: format!("{}.lora_a", name),
            key_b: format!("{}.lora_b", name),
            name: name.to_owned(),
        })
    }

    /// Load adapter matrices from a flat tensor map produced by `params()`.
    ///
    /// Looks up `{name}.lora_a` and `{name}.lora_b`. Returns an error if
    /// either key is missing. The shapes of the loaded tensors determine
    /// `in_features` / `out_features`; no external dimensions are required.
    ///
    /// **Dtype handling.** Loaded tensors are *always* upcast to `fp32`
    /// regardless of the on-disk storage dtype. This makes legacy v1
    /// adapters (saved before the fp32-LoRA change, where A/B may be
    /// stored in bf16/fp16 matching the base model) reload cleanly while
    /// ensuring optimizer updates against the new in-memory copies don't
    /// round below bf16 epsilon. Adapters saved after this change are
    /// already fp32 on disk so the cast is a no-op.
    pub fn from_safetensors(
        tensors: &HashMap<String, MxArray>,
        name: &str,
        cfg: &LoRAConfig,
    ) -> Result<Self> {
        let key_a = format!("{}.lora_a", name);
        let key_b = format!("{}.lora_b", name);

        let a = tensors
            .get(&key_a)
            .ok_or_else(|| Error::from_reason(format!("LoRA tensor not found: {}", key_a)))?;
        let b = tensors
            .get(&key_b)
            .ok_or_else(|| Error::from_reason(format!("LoRA tensor not found: {}", key_b)))?;

        // Shape sanity: a must be [in, rank], b must be [rank, out]
        let a_shape = a.shape()?;
        let b_shape = b.shape()?;
        if a_shape.len() != 2 {
            return Err(Error::from_reason(format!(
                "LoRA a for '{}' must be 2D, got {} dims",
                name,
                a_shape.len()
            )));
        }
        if b_shape.len() != 2 {
            return Err(Error::from_reason(format!(
                "LoRA b for '{}' must be 2D, got {} dims",
                name,
                b_shape.len()
            )));
        }
        if a_shape[1] != b_shape[0] {
            return Err(Error::from_reason(format!(
                "LoRA rank mismatch for '{}': a has rank {}, b has rank {}",
                name, a_shape[1], b_shape[0]
            )));
        }

        // Auto-upcast to f32. `astype` is cheap if the tensor is already f32.
        let a_f32 = a.astype(DType::Float32)?;
        let b_f32 = b.astype(DType::Float32)?;

        Ok(Self {
            a: a_f32,
            b: b_f32,
            scale: cfg.scale(),
            dropout: cfg.dropout,
            key_a,
            key_b,
            name: name.to_owned(),
        })
    }

    /// Returns a reference to the `a` matrix (`[in_features, rank]`).
    pub fn a(&self) -> &MxArray {
        &self.a
    }

    /// Returns a reference to the `b` matrix (`[rank, out_features]`).
    pub fn b(&self) -> &MxArray {
        &self.b
    }

    /// Replace adapter weights, validating that shapes are unchanged.
    ///
    /// Returns an error with concrete shape values if `a` or `b` differ from
    /// the current shapes.
    pub fn set_weights(&mut self, a: MxArray, b: MxArray) -> Result<()> {
        let cur_a_shape = self.a.shape()?;
        let new_a_shape = a.shape()?;
        if cur_a_shape.as_ref() != new_a_shape.as_ref() {
            return Err(Error::from_reason(format!(
                "LoRALinear::set_weights shape mismatch for 'a': expected {:?}, got {:?}",
                cur_a_shape.as_ref(),
                new_a_shape.as_ref()
            )));
        }
        let cur_b_shape = self.b.shape()?;
        let new_b_shape = b.shape()?;
        if cur_b_shape.as_ref() != new_b_shape.as_ref() {
            return Err(Error::from_reason(format!(
                "LoRALinear::set_weights shape mismatch for 'b': expected {:?}, got {:?}",
                cur_b_shape.as_ref(),
                new_b_shape.as_ref()
            )));
        }
        self.a = a;
        self.b = b;
        Ok(())
    }

    /// Compute the adapter contribution: `scale * (dropout(x) @ A) @ B`.
    ///
    /// The result is cast back to `x.dtype()` before returning.
    /// This cast is required because MLX promotes mixed-dtype binary ops to
    /// f32 — without it, adding `delta` to a bf16 base output would silently
    /// upcast the entire residual stream (`mlx_lm/tuner/lora.py:98`).
    ///
    /// Dropout uses inverted scaling: surviving activations are divided by
    /// `(1 - p)` so the expected value of `dropout(x)` equals `x`, matching
    /// PyTorch's `F.dropout(inplace=False)`.
    pub fn delta(&self, x: &MxArray, training: bool) -> Result<MxArray> {
        let x_dtype = x.dtype()?;

        if !matches!(x_dtype, DType::Float32 | DType::Float16 | DType::BFloat16) {
            return Err(Error::from_reason(format!(
                "LoRALinear::delta requires floating-point input, got {:?}",
                x_dtype
            )));
        }

        // Apply inverted dropout during training.
        // Skipped entirely when dropout == 0 to avoid a pointless Bernoulli
        // sample that would make the output non-deterministic.
        let x_dropped = if training && self.dropout > 0.0 {
            let shape = x.shape()?;
            let shape_slice: Vec<i64> = shape.to_vec();
            let keep_prob = 1.0 - self.dropout as f64;
            // Bernoulli mask: 1 where we keep, 0 where we drop
            let mask = MxArray::random_bernoulli(&shape_slice, keep_prob)?;
            // Cast mask to same dtype as x for the mul (avoids f32 promotion)
            let mask_typed = mask.astype(x_dtype)?;
            // Inverted scaling: divide by keep_prob so E[output] = E[input]
            let scale_arr = MxArray::scalar_float(keep_prob)?;
            let scale_typed = scale_arr.astype(x_dtype)?;
            let scaled_mask = mask_typed.div(&scale_typed)?;
            x.mul(&scaled_mask)?
        } else {
            x.clone()
        };

        // (x @ A) @ B — both A and B are in their stored dtype
        let xa = x_dropped.matmul(&self.a)?;
        let xab = xa.matmul(&self.b)?;

        // Scale and cast back to x's dtype to prevent silent f32 promotion
        // when the caller adds delta to a bf16 base output.
        let scaled = xab.mul_scalar(self.scale as f64)?;
        scaled.astype(x_dtype)
    }

    /// Merge adapter into the base weight: `W' = base_w + scale * (A @ B)^T`.
    ///
    /// `base_w` is `[out_features, in_features]` (HF Linear convention).
    /// `(A @ B)` has shape `[in, out]`; transposing gives `[out, in]` to match.
    /// The result is cast to `base_w.dtype()`.
    pub fn fuse_into_weight(&self, base_w: &MxArray) -> Result<MxArray> {
        let base_dtype = base_w.dtype()?;

        // A: [in, rank], B: [rank, out] → A@B: [in, out]
        let ab = self.a.matmul(&self.b)?;
        // Transpose to [out, in] to match HF weight layout
        let ab_t = ab.transpose(Some(&[1, 0]))?;
        let delta_w = ab_t.mul_scalar(self.scale as f64)?;

        // Cast both sides to base_dtype before adding to avoid f32 promotion
        let base_f = base_w.astype(DType::Float32)?;
        let delta_f = delta_w.astype(DType::Float32)?;
        let merged_f = base_f.add(&delta_f)?;

        // Cast back to original weight dtype
        merged_f.astype(base_dtype)
    }

    /// Iterator-style accessor for the two trainable parameters.
    ///
    /// Returns `[("{name}.lora_a", &a), ("{name}.lora_b", &b)]`. Used by the
    /// (future) trainable-param registry to collect gradient targets.
    /// Keys are precomputed at construction — this is an O(1) operation.
    pub fn params(&self) -> [(&str, &MxArray); 2] {
        [
            (self.key_a.as_str(), &self.a),
            (self.key_b.as_str(), &self.b),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_no_dropout() -> LoRAConfig {
        LoRAConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
        }
    }

    fn allclose(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn to_vec(arr: &MxArray) -> Vec<f32> {
        arr.to_float32().unwrap().to_vec()
    }

    #[test]
    fn lora_config_scale_matches_alpha_over_rank() {
        let cfg = LoRAConfig {
            rank: 16,
            alpha: 32.0,
            dropout: 0.05,
        };
        assert!((cfg.scale() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn forward_with_zero_b_returns_zero_delta() {
        let cfg = cfg_no_dropout();
        let adapter = LoRALinear::new_zero_b(8, 4, &cfg, "test_layer", DType::Float32).unwrap();

        // Random input [2, 3, 8]
        let x = MxArray::random_normal(&[2, 3, 8], 0.0, 1.0, None).unwrap();
        let delta = adapter.delta(&x, false).unwrap();

        let vals = to_vec(&delta);
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "Expected zero delta at index {} (B=0 invariant), got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn forward_matches_manual_matmul() {
        // Build A and B with known values
        #[rustfmt::skip]
        let a_data: Vec<f32> = vec![
            0.1, 0.2,   // row 0: in=0
            0.3, 0.4,   // row 1: in=1
            0.5, 0.6,   // row 2: in=2
            0.7, 0.8,   // row 3: in=3
        ];
        #[rustfmt::skip]
        let b_data: Vec<f32> = vec![
            1.0, 0.0, -1.0,  // row 0: rank=0
            0.0, 1.0,  0.5,  // row 1: rank=1
        ];
        let a = MxArray::from_float32(&a_data, &[4, 2]).unwrap();
        let b = MxArray::from_float32(&b_data, &[2, 3]).unwrap();
        let scale = 2.0_f32;

        let mut adapter = LoRALinear::new_zero_b(
            4,
            3,
            &LoRAConfig {
                rank: 2,
                alpha: 4.0,
                dropout: 0.0,
            },
            "test",
            DType::Float32,
        )
        .unwrap();
        adapter.set_weights(a, b).unwrap();
        adapter.scale = scale;

        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = MxArray::from_float32(&x_data, &[2, 4]).unwrap();

        let delta = adapter.delta(&x, false).unwrap();

        // Manual: scale * (x @ A) @ B
        let xa = x.matmul(adapter.a()).unwrap();
        let xab = xa.matmul(adapter.b()).unwrap();
        let expected = xab.mul_scalar(scale as f64).unwrap();

        let got = to_vec(&delta);
        let exp = to_vec(&expected);
        assert!(
            allclose(&got, &exp, 1e-5),
            "delta mismatch: got={:?} expected={:?}",
            got,
            exp
        );
    }

    #[test]
    fn fuse_preserves_combined_output() {
        let in_feat: i64 = 8;
        let out_feat: i64 = 4;
        let cfg = cfg_no_dropout();

        // Give A and B non-zero values; use from_float32 directly
        let a_data: Vec<f32> = (0..in_feat * cfg.rank).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..cfg.rank * out_feat).map(|i| i as f32 * 0.01).collect();

        let mut adapter =
            LoRALinear::new_zero_b(in_feat, out_feat, &cfg, "test_fuse", DType::Float32).unwrap();
        adapter
            .set_weights(
                MxArray::from_float32(&a_data, &[in_feat, cfg.rank]).unwrap(),
                MxArray::from_float32(&b_data, &[cfg.rank, out_feat]).unwrap(),
            )
            .unwrap();

        // base_w: [out, in] — HF convention
        let w_data: Vec<f32> = (0..out_feat * in_feat)
            .map(|i| (i as f32 - 16.0) * 0.05)
            .collect();
        let base_w = MxArray::from_float32(&w_data, &[out_feat, in_feat]).unwrap();

        let x = MxArray::random_normal(&[3, in_feat], 0.0, 1.0, None).unwrap();

        // Unfused: x @ W^T + delta(x)
        let w_t = base_w.transpose(Some(&[1, 0])).unwrap();
        let y_base = x.matmul(&w_t).unwrap();
        let y_delta = adapter.delta(&x, false).unwrap();
        let y_unfused = y_base.add(&y_delta).unwrap();

        // Fused: x @ W'^T  where W' = fuse_into_weight(W)
        let w_fused = adapter.fuse_into_weight(&base_w).unwrap();
        let w_fused_t = w_fused.transpose(Some(&[1, 0])).unwrap();
        let y_fused = x.matmul(&w_fused_t).unwrap();

        let unfused_vals = to_vec(&y_unfused);
        let fused_vals = to_vec(&y_fused);
        assert!(
            allclose(&unfused_vals, &fused_vals, 5e-3),
            "fuse mismatch: unfused={:?} fused={:?}",
            unfused_vals,
            fused_vals
        );
    }

    #[test]
    fn dropout_zero_is_deterministic() {
        let cfg = LoRAConfig {
            dropout: 0.0,
            ..cfg_no_dropout()
        };
        let mut adapter = LoRALinear::new_zero_b(8, 4, &cfg, "det_test", DType::Float32).unwrap();
        adapter
            .set_weights(
                MxArray::random_uniform(&[8, 4], -0.5, 0.5, Some(DType::Float32)).unwrap(),
                MxArray::random_uniform(&[4, 4], -0.5, 0.5, Some(DType::Float32)).unwrap(),
            )
            .unwrap();

        let x = MxArray::random_normal(&[2, 3, 8], 0.0, 1.0, None).unwrap();

        let out1 = adapter.delta(&x, true).unwrap();
        let out2 = adapter.delta(&x, true).unwrap();

        let v1 = to_vec(&out1);
        let v2 = to_vec(&out2);
        assert_eq!(
            v1, v2,
            "With dropout=0.0, delta should be identical across calls"
        );
    }

    #[test]
    fn new_zero_b_always_stores_f32_even_when_bf16_requested() {
        // Per standard PEFT practice the LoRA A/B matrices must remain f32
        // regardless of the base model's dtype. Pass BFloat16 explicitly and
        // assert the stored arrays are still Float32.
        let cfg = cfg_no_dropout();
        let adapter = LoRALinear::new_zero_b(8, 4, &cfg, "force_f32", DType::BFloat16).unwrap();
        assert_eq!(adapter.a().dtype().unwrap(), DType::Float32);
        assert_eq!(adapter.b().dtype().unwrap(), DType::Float32);
    }

    #[test]
    fn from_safetensors_upcasts_bf16_to_f32() {
        // Simulate a legacy v1 adapter where A/B were saved in bf16.
        // The loader must auto-upcast to f32 so optimizer updates don't
        // round below bf16 epsilon during training.
        let cfg = cfg_no_dropout();
        let in_feat: i64 = 8;
        let out_feat: i64 = 4;
        let name = "model.layers.0.q_proj";

        let a_f32 =
            MxArray::random_uniform(&[in_feat, cfg.rank], -0.1, 0.1, Some(DType::Float32)).unwrap();
        let b_f32 = MxArray::zeros(&[cfg.rank, out_feat], Some(DType::Float32)).unwrap();
        let a_bf16 = a_f32.astype(DType::BFloat16).unwrap();
        let b_bf16 = b_f32.astype(DType::BFloat16).unwrap();

        let mut map: HashMap<String, MxArray> = HashMap::new();
        map.insert(format!("{}.lora_a", name), a_bf16);
        map.insert(format!("{}.lora_b", name), b_bf16);

        let loaded = LoRALinear::from_safetensors(&map, name, &cfg).unwrap();
        assert_eq!(loaded.a().dtype().unwrap(), DType::Float32);
        assert_eq!(loaded.b().dtype().unwrap(), DType::Float32);
    }

    #[test]
    fn from_safetensors_roundtrip() {
        let cfg = cfg_no_dropout();
        let in_feat: i64 = 8;
        let out_feat: i64 = 4;
        let name = "model.layers.0.q_proj";

        let a_data: Vec<f32> = (0..in_feat * cfg.rank).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..cfg.rank * out_feat).map(|i| i as f32 * 0.2).collect();

        let mut original =
            LoRALinear::new_zero_b(in_feat, out_feat, &cfg, name, DType::Float32).unwrap();
        original
            .set_weights(
                MxArray::from_float32(&a_data, &[in_feat, cfg.rank]).unwrap(),
                MxArray::from_float32(&b_data, &[cfg.rank, out_feat]).unwrap(),
            )
            .unwrap();

        // Dump params into HashMap
        let mut map: HashMap<String, MxArray> = HashMap::new();
        for (k, arr) in original.params() {
            map.insert(k.to_owned(), arr.clone());
        }

        // Reload
        let loaded = LoRALinear::from_safetensors(&map, name, &cfg).unwrap();

        let a_orig = to_vec(original.a());
        let a_load = to_vec(loaded.a());
        let b_orig = to_vec(original.b());
        let b_load = to_vec(loaded.b());

        assert!(
            allclose(&a_orig, &a_load, 1e-6),
            "A mismatch after roundtrip"
        );
        assert!(
            allclose(&b_orig, &b_load, 1e-6),
            "B mismatch after roundtrip"
        );
        assert!((loaded.scale - cfg.scale()).abs() < 1e-6);
        assert_eq!(loaded.name, name);
    }
}
