// Allow architectural patterns that would require significant refactoring
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
// Allow doc formatting variations
#![allow(clippy::doc_nested_refdefs)]

pub mod array;
pub mod autograd;
pub mod cache_limit;
pub mod convert;
pub mod dataset;
pub mod decode_profiler;
pub mod gradients;
pub mod grpo;
pub(crate) mod inference_trace;
pub mod model_thread;
pub mod models;
pub mod moe;
pub mod nn;
pub mod optimizers;
pub mod output_store;
pub mod param_manager;
pub mod profiling;
pub mod response_store;
pub mod sampling;
pub mod sft;
pub mod stream;
pub mod tensor;
pub mod tokenizer;
pub mod tools;
pub mod tracing;
pub mod training;
pub mod training_model;
pub mod training_state;
pub mod transformer;
pub mod utils;
pub mod vision;

#[cfg(not(target_family = "wasm"))]
#[global_allocator]
static GLOBAL: mimalloc_safe::MiMalloc = mimalloc_safe::MiMalloc;
