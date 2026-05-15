//! Trainer-facing utilities (param registry, freezing, write-back).
//!
//! This module owns the small set of helpers that sit between the
//! model-shaped weight structures (e.g. [`crate::models::privacy_filter::ModelWeights`])
//! and the autograd / optimizer machinery in [`crate::autograd`] and
//! [`crate::optimizers`]. It is intentionally internal to the Rust
//! layer — no NAPI surface is exposed from here.

pub mod bioes_alignment;
pub mod privacy_dataset;
pub mod privacy_lora_trainer;
pub mod privacy_lora_trainer_napi;
pub mod trainable_params;

pub use bioes_alignment::{
    AlignedExample, EntitySpan, align_bioes, canonical_label_strs, default_label_to_id,
    truncate_bioes_inplace,
};
pub use privacy_dataset::{TokenClassificationExample, make_batch, read_jsonl};
pub use privacy_lora_trainer::{PrivacyLoraTrainConfig, PrivacyLoraTrainer};
pub use privacy_lora_trainer_napi::{PrivacyLoraTrainConfigJs, PrivacyLoraTrainerJs};
pub use trainable_params::{
    ParamRole, TrainableParam, TrainableParams, write_back_trainable_params,
};
