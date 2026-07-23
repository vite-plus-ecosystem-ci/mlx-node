//! Quantization graph helpers.
//!
//! Holds the FP8 (E4M3) activation fake-quant used to reproduce NVIDIA
//! modelopt activation math and the plain per-output-channel E4M3 weight
//! storage used by the Unsloth DGX artifact profile.

pub mod fp8_activation;
pub mod fp8_weight;
