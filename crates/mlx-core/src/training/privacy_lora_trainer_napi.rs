//! NAPI wrapper for [`PrivacyLoraTrainer`].
//!
//! Exposes the internal Rust trainer to JavaScript / TypeScript via a thin
//! wrapper that funnels the heavy `train()` / `save_adapter()` calls through
//! `spawn_blocking` so the Node.js event loop is not stalled.
//!
//! The wrapper owns the trainer behind an `Arc<tokio::sync::Mutex<_>>` so it
//! can be cloned into the blocking task. The mutex is acquired synchronously
//! inside the blocking task via `blocking_lock()`. The underlying trainer is
//! single-threaded by nature (MLX state cannot be operated on in parallel from
//! multiple threads), so callers should not invoke `train()` / `saveAdapter()`
//! concurrently from JS either — overlapping calls block on the mutex.

use std::path::PathBuf;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::Mutex;

use super::privacy_lora_trainer::{PrivacyLoraTrainConfig, PrivacyLoraTrainer};

/// JS-shaped configuration for [`PrivacyLoraTrainerJs::create`].
///
/// Mirrors [`PrivacyLoraTrainConfig`], but uses NAPI-compatible numeric types
/// (`f64` instead of `f32`, `u32` instead of `usize`/`i64` for counts) and
/// `Option<T>` for every tunable so JavaScript callers can omit any field to
/// fall back to the documented defaults.
#[napi(object)]
#[derive(Clone)]
pub struct PrivacyLoraTrainConfigJs {
    /// Filesystem path to the base privacy-filter checkpoint directory.
    pub model_path: String,
    /// Path to the training JSONL dataset (pre-tokenized).
    pub data_path: String,
    /// Optional evaluation JSONL dataset path. Currently ignored by the
    /// trainer — reserved for future eval-loop integration.
    pub eval_path: Option<String>,
    /// Directory where `adapter.safetensors`, `adapter_config.json`,
    /// `optimizer_state.safetensors`, and `trainer_state.json` are written.
    pub output_dir: String,
    /// LoRA rank. Default: 16.
    pub rank: Option<i64>,
    /// LoRA alpha (scaling factor). Default: 32.
    pub alpha: Option<f64>,
    /// LoRA dropout probability. Default: 0.05.
    pub dropout: Option<f64>,
    /// AdamW learning rate for the LoRA A/B matrices. Default: 1e-4.
    pub lora_lr: Option<f64>,
    /// AdamW learning rate for the classifier head. Default: 5e-5.
    pub classifier_lr: Option<f64>,
    /// Examples per micro-batch. Default: 2.
    pub batch_size: Option<u32>,
    /// Maximum sequence length (longer examples are truncated). Default: 256.
    pub max_seq_len: Option<u32>,
    /// Number of epochs to iterate over the dataset. Default: 3.
    pub num_epochs: Option<u32>,
    /// Number of micro-batches accumulated per optimizer step. Default: 4.
    pub grad_accum_steps: Option<u32>,
    /// L2 norm clipping threshold across all gradients. Default: 1.0.
    pub grad_clip: Option<f64>,
    /// Save an intermediate checkpoint every N optimizer steps. Default: 500.
    /// Set to 0 to disable intermediate checkpointing.
    pub save_every: Option<u32>,
    /// Optional path to an existing `output_dir`-style checkpoint to resume
    /// from. Loads `adapter.safetensors`, `optimizer_state.safetensors`, and
    /// `trainer_state.json` if present.
    pub resume_from: Option<String>,
    /// Token id used to pad short sequences. Default: 0.
    pub pad_token_id: Option<i64>,
}

impl From<PrivacyLoraTrainConfigJs> for PrivacyLoraTrainConfig {
    fn from(js: PrivacyLoraTrainConfigJs) -> Self {
        Self {
            model_path: PathBuf::from(js.model_path),
            data_path: PathBuf::from(js.data_path),
            eval_path: js.eval_path.map(PathBuf::from),
            output_dir: PathBuf::from(js.output_dir),
            rank: js.rank,
            // Down-cast f64 → f32 for the internal config. NAPI doesn't have a
            // first-class f32 type, so we accept f64 on the boundary and
            // narrow here. Out-of-range values silently saturate, which is
            // fine for the alpha/dropout ranges users actually pick.
            alpha: js.alpha.map(|v| v as f32),
            dropout: js.dropout.map(|v| v as f32),
            lora_lr: js.lora_lr,
            classifier_lr: js.classifier_lr,
            batch_size: js.batch_size.map(|v| v as usize),
            max_seq_len: js.max_seq_len.map(|v| v as usize),
            num_epochs: js.num_epochs.map(|v| v as usize),
            grad_accum_steps: js.grad_accum_steps.map(|v| v as usize),
            grad_clip: js.grad_clip,
            save_every: js.save_every.map(|v| v as usize),
            resume_from: js.resume_from.map(PathBuf::from),
            pad_token_id: js.pad_token_id,
        }
    }
}

/// NAPI-exposed handle to a [`PrivacyLoraTrainer`].
///
/// Construct via [`PrivacyLoraTrainerJs::create`] (synchronous factory — the
/// underlying [`PrivacyLoraTrainer::create`] is itself synchronous and loads
/// the base model + dataset on the calling thread). Drive training via
/// [`PrivacyLoraTrainerJs::train`] and persist results via
/// [`PrivacyLoraTrainerJs::save_adapter`].
///
/// The trainer is held behind a `tokio::sync::Mutex<PrivacyLoraTrainer>` and
/// shared via an `Arc` so it can be cloned into a `spawn_blocking` task. The
/// blocking task acquires the mutex with `blocking_lock()` — we never hold
/// the mutex across an `.await`, so a plain `Mutex<T>` (no `Option` /
/// take-restore dance) is enough. Concurrent calls from JS serialize on the
/// mutex.
#[napi]
pub struct PrivacyLoraTrainerJs {
    inner: Arc<Mutex<PrivacyLoraTrainer>>,
}

#[napi]
impl PrivacyLoraTrainerJs {
    /// Construct a new trainer.
    ///
    /// Synchronous: the heavy lifting (loading the base checkpoint, attaching
    /// zero-B LoRA adapters, parsing the JSONL dataset) happens inline. Use
    /// from a Node.js worker if you need to keep the event loop responsive
    /// during creation.
    #[napi(factory)]
    pub fn create(config: PrivacyLoraTrainConfigJs) -> Result<Self> {
        let inner_cfg: PrivacyLoraTrainConfig = config.into();
        let trainer = PrivacyLoraTrainer::create(inner_cfg)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(trainer)),
        })
    }

    /// Run the full training loop and return the final optimizer step count.
    ///
    /// The trainer iterates over the JSONL dataset for `num_epochs` epochs,
    /// applying gradient accumulation per step. Intermediate checkpoints are
    /// written every `save_every` steps if that field is non-zero.
    ///
    /// Errors:
    /// - If another call (e.g. a previous `train()` or `saveAdapter()`) is
    ///   in flight, this method will block waiting for the mutex.
    #[napi]
    pub async fn train(&self) -> Result<u32> {
        let inner = self.inner.clone();
        let result = napi::bindgen_prelude::spawn_blocking(move || {
            // Acquire the mutex inside the blocking task. Tokio's
            // `blocking_lock` is the right tool here — we're not on a Tokio
            // worker thread, so awaiting `.lock()` would be wrong; and we
            // need to block until the (single-threaded) trainer is free.
            let mut guard = inner.blocking_lock();
            guard.train()
        })
        .await
        .map_err(|e| Error::from_reason(format!("train() task join error: {e}")))??;

        Ok(result as u32)
    }

    /// Persist adapter weights, optimizer state, and trainer progress to the
    /// configured `output_dir`. Overwrites any previous checkpoint at that
    /// path.
    #[napi]
    pub async fn save_adapter(&self) -> Result<()> {
        let inner = self.inner.clone();
        napi::bindgen_prelude::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            guard.save_adapter()
        })
        .await
        .map_err(|e| Error::from_reason(format!("saveAdapter() task join error: {e}")))??;

        Ok(())
    }
}
