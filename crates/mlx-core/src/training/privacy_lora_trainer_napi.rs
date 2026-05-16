//! NAPI wrapper for [`PrivacyLoraTrainer`].
//!
//! Exposes the internal Rust trainer to JavaScript / TypeScript via a thin
//! wrapper that funnels the heavy `train()` / `save_adapter()` calls through
//! `spawn_blocking` so the Node.js event loop is not stalled.
//!
//! ## Threading model & MLX stream registration
//!
//! MLX's metal backend keeps `CommandEncoder` instances in
//! `static thread_local` storage (see `mlx/backend/metal/device.cpp`). The
//! per-thread "default stream" returned by `mlx::core::default_stream(d)`
//! is also thread-local, and `set_default_stream(s)` does *not* register
//! the encoder for `s` on the calling thread — it just updates the
//! per-thread default index.
//!
//! Practical consequence: the model is loaded synchronously inside
//! [`PrivacyLoraTrainerJs::create`] (the NAPI factory, which runs on the
//! Node main thread). All of that model's arrays are stream-affinitised to
//! whatever the main thread's default GPU stream is at load time. Tokio's
//! `spawn_blocking` runs the training closure on a separate worker thread,
//! and unless we explicitly register the model's stream there, MLX throws
//! "There is no Stream(gpu, N) in current thread" on the first eval.
//!
//! The wrapper therefore captures the GPU stream at construction time, and
//! every `spawn_blocking` closure registers + activates that same stream on
//! its own worker thread before touching MLX state. See
//! [`Stream::register_on_current_thread`].
//!
//! The trainer itself is held behind an `Arc<tokio::sync::Mutex<_>>` so it
//! can be cloned into the blocking task; the mutex is acquired synchronously
//! inside the blocking task via `blocking_lock()`. Concurrent calls from JS
//! serialize on the mutex.

use std::path::PathBuf;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::Mutex;

use crate::stream::{DeviceType, Stream};

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
///
/// `gpu_stream` is captured on the construction thread (where the model is
/// loaded) and re-registered on every `spawn_blocking` worker so that all
/// MLX ops, regardless of which Tokio worker runs them, share one logical
/// default GPU stream. See the module-level docs for the why.
#[napi]
pub struct PrivacyLoraTrainerJs {
    inner: Arc<Mutex<PrivacyLoraTrainer>>,
    gpu_stream: Stream,
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
        // Capture the calling thread's default GPU stream BEFORE loading the
        // model. The model's arrays will end up affinitised to this stream
        // (since `PrivacyFilterModel::load_from_dir` uses MLX ops that read
        // the per-thread default), and we re-register the same `Stream`
        // value on every Tokio worker that subsequently runs training.
        let gpu_stream = Stream::default(DeviceType::Gpu);

        let inner_cfg: PrivacyLoraTrainConfig = config.into();
        let trainer = PrivacyLoraTrainer::create(inner_cfg)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(trainer)),
            gpu_stream,
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
        let stream = self.gpu_stream;
        let result = napi::bindgen_prelude::spawn_blocking(move || {
            // Adopt the construction-thread's GPU stream on this Tokio
            // worker. `register_on_current_thread` seeds the per-thread
            // CommandEncoder map for `stream.index`; `make_default` sets it
            // as the default so subsequent MLX ops use it. Without this,
            // the autograd worker fails on first eval with "no Stream(gpu,
            // N) in current thread" because the model's arrays carry an
            // index that exists only in the main thread's TLS.
            stream.register_on_current_thread();
            stream.make_default();

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
        let stream = self.gpu_stream;
        napi::bindgen_prelude::spawn_blocking(move || {
            stream.register_on_current_thread();
            stream.make_default();

            let mut guard = inner.blocking_lock();
            guard.save_adapter()
        })
        .await
        .map_err(|e| Error::from_reason(format!("saveAdapter() task join error: {e}")))??;

        Ok(())
    }
}
