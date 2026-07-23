//! Plain E4M3 FP8 weight storage with one dequant scale per output channel.
//!
//! This is deliberately distinct from MLX `mxfp8`: the stored weight is raw
//! E4M3 bytes (`Uint8`) with a floating `[... , N, 1]` dequant scale, not a
//! packed MX tensor with E8M0 block scales. Runtime support in this crate is a
//! correctness fallback: weights are reconstructed to BF16 once at load and
//! ordinary A16 matmul / gather-mm is used. This module does not claim native
//! W8A8 execution.

use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};

/// Serialized per-layer mode discriminator.
pub const FP8_E4M3_MODE: &str = "fp8_e4m3";
/// Plain E4M3 stores eight bits per weight.
pub const FP8_E4M3_BITS: i32 = 8;
/// There is no K-axis quantization group; scales are per output channel.
pub const FP8_E4M3_GROUP_SIZE: i32 = -1;
/// Largest finite value in MLX's E4M3 encoding.
const FP8_E4M3_MAX: f64 = 448.0;

fn validate_rank(shape: &[i64], expected_ndim: usize, label: &str) -> Result<()> {
    if shape.len() != expected_ndim {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': expected {expected_ndim}-D weight, got shape {shape:?}"
        )));
    }
    if shape.iter().any(|&dim| dim <= 0) {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': weight dimensions must be positive, got {shape:?}"
        )));
    }
    Ok(())
}

/// Quantize a 2-D dense or 3-D expert-stack weight to the checkpoint format.
///
/// For source `weight[..., N, K]`, emits:
///
/// - raw E4M3 `Uint8` weight with the unchanged `[..., N, K]` shape;
/// - BF16 dequant scale `[..., N, 1]`, where `scale = max(abs(row)) / 448`.
///
/// A strictly-positive floor makes all-zero rows round-trip as zero without a
/// divide-by-zero. Quantization is memoryless min/max and applies no activation
/// quantization.
pub fn quantize_per_output_channel(
    weight: &MxArray,
    key_for_error: &str,
) -> Result<(MxArray, MxArray)> {
    let source_dtype = weight.dtype()?;
    if !matches!(
        source_dtype,
        DType::Float32 | DType::Float16 | DType::BFloat16
    ) {
        return Err(Error::from_reason(format!(
            "plain FP8 quantization for '{key_for_error}' requires a floating source weight, got {source_dtype:?}"
        )));
    }
    if weight.has_nan_or_inf()? {
        return Err(Error::from_reason(format!(
            "plain FP8 quantization for '{key_for_error}' requires finite source weights"
        )));
    }
    let shape = weight.shape()?.to_vec();
    if shape.len() != 2 && shape.len() != 3 {
        return Err(Error::from_reason(format!(
            "plain FP8 quantization for '{key_for_error}' supports only 2-D [N,K] or \
             3-D [E,N,K] weights, got {shape:?}"
        )));
    }
    if shape.iter().any(|&dim| dim <= 0) {
        return Err(Error::from_reason(format!(
            "plain FP8 quantization for '{key_for_error}' requires positive dimensions, got {shape:?}"
        )));
    }

    let weight_f32 = weight.astype(DType::Float32)?;
    let dequant_scale = weight_f32
        .abs()?
        .max(Some(&[-1]), Some(true))?
        .div_scalar(FP8_E4M3_MAX)?
        .clip(Some(f32::MIN_POSITIVE as f64), None)?;
    let encoded = weight_f32
        .div(&dequant_scale)?
        .clip(Some(-FP8_E4M3_MAX), Some(FP8_E4M3_MAX))?
        .to_fp8()?;
    // Unsloth's plain FP8 recipe stores one BF16 inverse/dequant scale per
    // output channel. Keep that dtype stable regardless of source/CLI dtype.
    let dequant_scale = dequant_scale.astype(DType::BFloat16)?;
    Ok((encoded, dequant_scale))
}

/// Validate the cheap structural/plain-scale portion of plain-FP8 storage.
///
/// This is shared by the runtime builder and the converter's already-quantized
/// re-conversion guard. It intentionally does not reconstruct the full BF16
/// tensor; the runtime's [`validate_and_dequantize`] performs the additional
/// reserved-byte and BF16-finiteness checks while decoding.
pub fn validate_storage_metadata(
    weight: &MxArray,
    dequant_scale: &MxArray,
    expected_ndim: usize,
    label: &str,
) -> Result<()> {
    let weight_dtype = weight.dtype()?;
    if weight_dtype != DType::Uint8 {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': expected Uint8 E4M3 .weight, got {weight_dtype:?}"
        )));
    }
    let weight_shape = weight.shape()?.to_vec();
    validate_rank(&weight_shape, expected_ndim, label)?;

    let scale_dtype = dequant_scale.dtype()?;
    if !matches!(
        scale_dtype,
        DType::Float32 | DType::Float16 | DType::BFloat16
    ) {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': expected floating .scales, got {scale_dtype:?}"
        )));
    }
    let scale_shape = dequant_scale.shape()?.to_vec();
    let mut expected_scale_shape = weight_shape[..expected_ndim - 1].to_vec();
    expected_scale_shape.push(1);
    if scale_shape != expected_scale_shape {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': expected .scales shape {expected_scale_shape:?} \
             for weight {weight_shape:?}, got {scale_shape:?}"
        )));
    }
    if dequant_scale.has_nan_or_inf()? {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': .scales contains NaN or Inf"
        )));
    }
    let scale_f32 = dequant_scale.astype(DType::Float32)?;
    let min_scale = scale_f32.min(None, None)?;
    min_scale.eval();
    if min_scale.item_at_float32(0)? <= 0.0 {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': .scales must be strictly positive"
        )));
    }

    Ok(())
}

/// Strictly validate plain-FP8 checkpoint storage and reconstruct BF16 weight.
///
/// `expected_ndim=2` accepts `[N,K]` + `[N,1]`; `expected_ndim=3` accepts
/// `[E,N,K]` + `[E,N,1]`. The scale may use any real floating dtype but must
/// be finite and strictly positive. The returned tensor has the weight's
/// source orientation and is consumed by ordinary A16 matmul/gather-mm.
pub fn validate_and_dequantize(
    weight: &MxArray,
    dequant_scale: &MxArray,
    expected_ndim: usize,
    label: &str,
) -> Result<MxArray> {
    validate_storage_metadata(weight, dequant_scale, expected_ndim, label)?;

    let decoded = weight.from_fp8(DType::BFloat16)?;
    if decoded.has_nan_or_inf()? {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': .weight contains an invalid/non-finite E4M3 encoding"
        )));
    }
    // MLX's fast decoder maps the E4M3FN reserved 0x7f/0xff bit patterns to
    // +/-480 rather than NaN. `to_fp8` never emits them (it saturates at
    // +/-448), so reject any decoded magnitude above the format's finite max.
    let max_decoded = decoded.astype(DType::Float32)?.abs()?.max(None, None)?;
    max_decoded.eval();
    if max_decoded.item_at_float32(0)? > FP8_E4M3_MAX as f32 {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': .weight contains a reserved E4M3 encoding"
        )));
    }
    let reconstructed = decoded.mul(&dequant_scale.astype(DType::BFloat16)?)?;
    if reconstructed.has_nan_or_inf()? {
        return Err(Error::from_reason(format!(
            "plain FP8 layer '{label}': BF16 reconstruction contains NaN or Inf"
        )));
    }
    Ok(reconstructed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_fp8_roundtrip_has_expected_storage_contract() {
        let source = MxArray::from_float32(&[0.0, 1.0, -2.0, 0.5, 3.0, -1.5, 0.25, -0.75], &[2, 4])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let (weight, scales) = quantize_per_output_channel(&source, "test").unwrap();
        assert_eq!(weight.dtype().unwrap(), DType::Uint8);
        assert_eq!(weight.shape().unwrap().to_vec(), vec![2, 4]);
        assert_eq!(scales.dtype().unwrap(), DType::BFloat16);
        assert_eq!(scales.shape().unwrap().to_vec(), vec![2, 1]);

        let dequant = validate_and_dequantize(&weight, &scales, 2, "test").unwrap();
        assert_eq!(dequant.dtype().unwrap(), DType::BFloat16);
        assert_eq!(dequant.shape().unwrap().to_vec(), vec![2, 4]);
        let got = dequant
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        let want = source.astype(DType::Float32).unwrap().to_float32().unwrap();
        for (got, want) in got.iter().zip(want.iter()) {
            assert!((got - want).abs() <= 0.06, "got={got}, want={want}");
        }
    }

    #[test]
    fn plain_fp8_validation_rejects_wrong_scale_shape_and_dtype() {
        let weight = MxArray::from_uint8(&[0; 8], &[2, 4]).unwrap();
        let bad_shape = MxArray::from_float32(&[1.0, 1.0], &[2]).unwrap();
        assert!(validate_and_dequantize(&weight, &bad_shape, 2, "bad-shape").is_err());

        let bad_dtype = MxArray::from_uint8(&[1, 1], &[2, 1]).unwrap();
        assert!(validate_and_dequantize(&weight, &bad_dtype, 2, "bad-dtype").is_err());

        let zero_scale = MxArray::from_float32(&[0.0, 1.0], &[2, 1]).unwrap();
        assert!(validate_and_dequantize(&weight, &zero_scale, 2, "zero-scale").is_err());

        // E4M3 reserves 0x7f/0xff for NaN. Reject such bytes at load rather
        // than allowing a malformed checkpoint to poison every later matmul.
        let invalid_weight = MxArray::from_uint8(&[0x7f, 0, 0, 0, 0, 0, 0, 0], &[2, 4]).unwrap();
        let scales = MxArray::from_float32(&[1.0, 1.0], &[2, 1]).unwrap();
        assert!(validate_and_dequantize(&invalid_weight, &scales, 2, "invalid-byte").is_err());
    }

    #[test]
    fn plain_fp8_quantization_rejects_non_float_or_non_finite_source() {
        let integer_source = MxArray::from_uint8(&[0; 8], &[2, 4]).unwrap();
        assert!(quantize_per_output_channel(&integer_source, "integer").is_err());

        let non_finite = MxArray::from_float32(
            &[0.0, 1.0, f32::INFINITY, 0.5, 3.0, -1.5, 0.25, -0.75],
            &[2, 4],
        )
        .unwrap();
        assert!(quantize_per_output_channel(&non_finite, "non-finite").is_err());
    }
}
