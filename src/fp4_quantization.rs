//! FlashInfer FP4 quantization helpers (Blackwell).
//!
//! Exposes the `block_scale_interleave_sm100` kernel from FlashInfer's
//! `flashinfer_jit_cache/fp4_quantization_{100,103,110,120}` module,
//! used to convert modelopt-format unswizzled per-group FP8 weight scales
//! (`[E?, out, in / 16]` FP8 E4M3) into the swizzled
//! (`[E? * round_up(out, 128) * round_up(in/16, 4)]` bytes) layout the NVFP4
//! fused-MoE kernel consumes.
//!
//! Cross-reference:
//! - `flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h`
//! - `flashinfer/csrc/fp4_quantization.cu::block_scale_interleave_sm100`
//! - `flashinfer/flashinfer/fp4_quantization.py::block_scale_interleave`

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_CUDA, KDL_UINT, TVMFFIAny, any_dltensor_ptr, any_none,
};
use crate::runtime::FlashInferRuntime;

/// Backend variant of the FP4 quantization JIT-cache module to load.
///
/// The kernel is per-SM-arch shipped as `fp4_quantization_{90,100,103,110,120}.so`.
/// Only `Sm100`/`Sm103`/`Sm110`/`Sm120` are valid for the swizzle kernel
/// (`block_scale_interleave_sm100` is built only for those archs);
/// `Sm90` would be selected for Hopper-only fp4_quantize but does not
/// expose the swizzle symbol used here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp4QuantizationBackend {
    Sm100,
    Sm103,
    Sm110,
    Sm120,
}

impl Fp4QuantizationBackend {
    fn kernel_uri(self) -> &'static str {
        match self {
            Fp4QuantizationBackend::Sm100 => "fp4_quantization_100",
            Fp4QuantizationBackend::Sm103 => "fp4_quantization_103",
            Fp4QuantizationBackend::Sm110 => "fp4_quantization_110",
            Fp4QuantizationBackend::Sm120 => "fp4_quantization_120",
        }
    }
}

/// Computes the swizzled per-expert scale buffer size in bytes (== uint8
/// elements).
///
/// Mirrors `flashinfer.fp4_quantization._compute_swizzled_layout_sf_size`:
///   `padded_row = round_up(rows, 128)`
///   `padded_col = round_up(cols, 4)`
///   `expert_size = padded_row * padded_col`
///
/// `cols` is expected in *unswizzled* element count along the inner dim,
/// i.e. `in / 16` for an NVFP4 weight tile of inner dim `in`.
pub fn swizzled_layout_size(rows: i64, cols: i64) -> i64 {
    let padded_row = (rows + 127) / 128 * 128;
    let padded_col = (cols + 3) / 4 * 4;
    padded_row * padded_col
}

/// Input descriptor for `block_scale_interleave_sm100`.
///
/// The kernel accepts an `rank-2` ([rows, cols]) or `rank-3` ([E, rows, cols])
/// uint8 tensor of unswizzled FP8 E4M3 block-scale bytes. The total output
/// element count is `(num_experts or 1) * swizzled_layout_size(rows, cols)`
/// uint8 bytes laid out as a flat 1-D buffer.
#[derive(Debug, Clone, Copy)]
pub struct BlockScaleInterleaveParams {
    /// Input pointer to unswizzled FP8 E4M3 bytes (interpreted as uint8 by
    /// the kernel).
    pub input_ptr: *const c_void,
    /// Number of experts (= leading batch dim), or 1 for a rank-2 input.
    pub num_experts: i64,
    /// Inner row count `rows` (typically `2 * inter` for FC1, `hidden` for FC2).
    pub rows: i64,
    /// Inner column count `cols` (= `in / 16` for the NVFP4 weight tile).
    pub cols: i64,
    /// Output buffer pointer; caller allocates
    /// `num_experts * swizzled_layout_size(rows, cols)` uint8 bytes.
    pub output_ptr: *mut c_void,
    /// CUDA device id.
    pub device_id: i32,
    /// CUDA stream (`cudaStream_t`).
    pub stream: *mut c_void,
    /// Kernel SM backend (e.g. `Sm100` for B200).
    pub backend: Fp4QuantizationBackend,
}

/// Invoke FlashInfer's `block_scale_interleave_sm100` kernel to convert
/// modelopt-format unswizzled FP8 E4M3 per-group weight scales into the
/// CUTLASS-tile-friendly swizzled layout expected by the NVFP4 fused-MoE
/// kernel.
pub fn block_scale_interleave(params: &BlockScaleInterleaveParams) -> Result<(), FlashInferError> {
    if params.input_ptr.is_null() {
        return Err(FlashInferError::invalid_argument(
            "block_scale_interleave: input_ptr is null",
        ));
    }
    if params.output_ptr.is_null() {
        return Err(FlashInferError::invalid_argument(
            "block_scale_interleave: output_ptr is null",
        ));
    }
    if params.rows <= 0 || params.cols <= 0 || params.num_experts <= 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "block_scale_interleave: rows, cols, and num_experts must be > 0 (got rows={}, cols={}, num_experts={})",
            params.rows, params.cols, params.num_experts
        )));
    }

    let runtime = FlashInferRuntime::global()?;

    let uint8_dtype = DLDataType {
        code: KDL_UINT,
        bits: 8,
        lanes: 1,
    };

    // Input tensor: rank-3 [E, rows, cols] uint8 (kernel also accepts rank-2
    // but we always pass rank-3 for uniform handling).
    let mut in_shape = [params.num_experts, params.rows, params.cols];
    let mut in_strides = [params.rows * params.cols, params.cols, 1_i64];
    let in_tensor = DLTensor {
        data: params.input_ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.device_id,
        },
        ndim: 3,
        dtype: uint8_dtype,
        shape: in_shape.as_mut_ptr(),
        strides: in_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    // Output tensor: rank-1 flat uint8 buffer of size
    // num_experts * swizzled_layout_size(rows, cols).
    let out_len = params
        .num_experts
        .checked_mul(swizzled_layout_size(params.rows, params.cols))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "block_scale_interleave: swizzled output size overflows i64",
            )
        })?;
    let mut out_shape = [out_len];
    let mut out_strides = [1_i64];
    let out_tensor = DLTensor {
        data: params.output_ptr,
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.device_id,
        },
        ndim: 1,
        dtype: uint8_dtype,
        shape: out_shape.as_mut_ptr(),
        strides: out_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let args: [TVMFFIAny; 2] = [any_dltensor_ptr(&in_tensor), any_dltensor_ptr(&out_tensor)];
    let mut result = any_none();

    // SAFETY: stream context API contract comes from tvm ffi and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.device_id, params.stream)? };
    let mut restore_guard = StreamRestoreGuard::new(runtime, params.device_id, previous_stream);

    let call_result = (|| -> Result<(), FlashInferError> {
        // SAFETY: invoking TVM Function with ABI-packed args.
        unsafe {
            runtime.call_block_scale_interleave(
                params.backend.kernel_uri(),
                args.as_ptr(),
                args.len() as i32,
                &mut result as *mut _,
            )?;
        }
        Ok(())
    })();

    let restore_result = restore_guard.restore_now();
    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

struct StreamRestoreGuard<'a> {
    runtime: &'a FlashInferRuntime,
    device_id: i32,
    previous_stream: *mut c_void,
    active: bool,
}

impl<'a> StreamRestoreGuard<'a> {
    fn new(runtime: &'a FlashInferRuntime, device_id: i32, previous_stream: *mut c_void) -> Self {
        Self {
            runtime,
            device_id,
            previous_stream,
            active: true,
        }
    }

    fn restore_now(&mut self) -> Result<(), FlashInferError> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        // SAFETY: `previous_stream` is returned from TVMFFIEnvSetStream for this device.
        unsafe {
            self.runtime
                .restore_stream(self.device_id, self.previous_stream)
        }
    }
}

impl Drop for StreamRestoreGuard<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        // SAFETY: best-effort stream restore in drop path.
        let _ = unsafe {
            self.runtime
                .restore_stream(self.device_id, self.previous_stream)
        };
    }
}
