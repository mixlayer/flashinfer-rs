//! Dense NVFP4 quantization and SM120-family CUTLASS GEMM bindings.

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_FLOAT, KDL_UINT, KTVM_FFI_INT,
    TVMFFIAny, any_bool, any_dltensor_ptr, any_i64, any_none,
};
use crate::runtime::FlashInferRuntime;

/// Scalar storage types accepted by the dense NVFP4 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvFp4DType {
    /// IEEE half precision.
    F16,
    /// Brain floating point 16-bit precision.
    BF16,
    /// Raw bytes, including packed E2M1 values and E4M3 scale bytes.
    U8,
    /// IEEE single precision, used for global scaling tensors.
    F32,
}

impl NvFp4DType {
    fn as_dl_dtype(self) -> DLDataType {
        let (code, bits) = match self {
            Self::F16 => (KDL_FLOAT, 16),
            Self::BF16 => (KDL_BFLOAT, 16),
            Self::U8 => (KDL_UINT, 8),
            Self::F32 => (KDL_FLOAT, 32),
        };
        DLDataType {
            code,
            bits,
            lanes: 1,
        }
    }
}

/// A contiguous rank-1 CUDA tensor descriptor.
///
/// The pointed-to allocation must remain live until work on `stream` completes.
#[derive(Debug, Clone, Copy)]
pub struct NvFp4Tensor1DDesc {
    /// Device pointer at the first logical element.
    pub ptr: *const c_void,
    /// Element count.
    pub len: i64,
    /// Scalar storage type.
    pub dtype: NvFp4DType,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// A contiguous row-major rank-2 CUDA tensor descriptor.
///
/// The pointed-to allocation must remain live until work on `stream` completes.
#[derive(Debug, Clone, Copy)]
pub struct NvFp4Tensor2DDesc {
    /// Device pointer at the first logical element.
    pub ptr: *const c_void,
    /// Number of rows.
    pub rows: i64,
    /// Number of columns in storage elements, not unpacked FP4 lanes.
    pub cols: i64,
    /// Scalar storage type.
    pub dtype: NvFp4DType,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// Arguments for dynamic per-token NVFP4 activation quantization.
#[derive(Debug, Clone, Copy)]
pub struct NvFp4QuantizeParams {
    /// Contiguous F16/BF16 activations `[M,K]`, with `K` divisible by 16.
    pub input: NvFp4Tensor2DDesc,
    /// F32 quantization multiplier `[1]`, normally `(448 * 6) / absmax(input)`.
    pub global_scale: NvFp4Tensor1DDesc,
    /// Writable packed E2M1 bytes `[M,K/2]`.
    pub output: NvFp4Tensor2DDesc,
    /// Writable U8 E4M3 scales in FlashInfer's padded 128x4 layout.
    pub output_scale: NvFp4Tensor1DDesc,
    /// Whether to enable programmatic dependent launch.
    pub enable_pdl: bool,
    /// CUDA stream used for asynchronous execution.
    pub stream: *mut c_void,
}

impl NvFp4QuantizeParams {
    /// Validate pointer, shape, dtype, and device contracts without launching CUDA.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_ptrs(&[
            (self.input.ptr, "input"),
            (self.global_scale.ptr, "global_scale"),
            (self.output.ptr, "output"),
            (self.output_scale.ptr, "output_scale"),
        ])?;
        if self.input.rows <= 0 || self.input.cols <= 0 || self.input.cols % 16 != 0 {
            return invalid("input must have positive [M,K] dimensions with K divisible by 16");
        }
        if !matches!(self.input.dtype, NvFp4DType::F16 | NvFp4DType::BF16) {
            return invalid("input must be F16 or BF16");
        }
        if self.global_scale.dtype != NvFp4DType::F32 || self.global_scale.len != 1 {
            return invalid("global_scale must be F32 with one element");
        }
        if self.output.dtype != NvFp4DType::U8
            || self.output.rows != self.input.rows
            || self.output.cols != self.input.cols / 2
        {
            return invalid("output must be U8 [M,K/2]");
        }
        let expected_scale = swizzled_scale_len(self.input.rows, self.input.cols / 16)?;
        if self.output_scale.dtype != NvFp4DType::U8 || self.output_scale.len != expected_scale {
            return invalid(format!(
                "output_scale must be U8 with {expected_scale} elements"
            ));
        }
        validate_same_device(
            self.input.device_id,
            &[
                self.global_scale.device_id,
                self.output.device_id,
                self.output_scale.device_id,
            ],
        )
    }
}

/// Arguments for one-time conversion of checkpoint E4M3 scale bytes.
#[derive(Debug, Clone, Copy)]
pub struct NvFp4ScaleInterleaveParams {
    /// Contiguous unswizzled U8 scale bytes `[N,K/16]`.
    pub input: NvFp4Tensor2DDesc,
    /// Writable U8 scale bytes with padded, swizzled storage.
    pub output: NvFp4Tensor1DDesc,
    /// CUDA stream used for asynchronous execution.
    pub stream: *mut c_void,
}

impl NvFp4ScaleInterleaveParams {
    /// Validate pointer, shape, dtype, and device contracts without launching CUDA.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_ptrs(&[(self.input.ptr, "input"), (self.output.ptr, "output")])?;
        if self.input.rows <= 0 || self.input.cols <= 0 {
            return invalid("scale input dimensions must be positive");
        }
        if self.input.dtype != NvFp4DType::U8 || self.output.dtype != NvFp4DType::U8 {
            return invalid("scale input and output must use U8 raw storage");
        }
        let expected = swizzled_scale_len(self.input.rows, self.input.cols)?;
        if self.output.len != expected {
            return invalid(format!("interleaved output must have {expected} elements"));
        }
        validate_same_device(self.input.device_id, &[self.output.device_id])
    }
}

/// Arguments for FlashInfer's dense W4A4 NVFP4 CUTLASS GEMM on SM120 GPUs.
#[derive(Debug, Clone, Copy)]
pub struct NvFp4GemmParams {
    /// Packed activation bytes `[M,K/2]`.
    pub input: NvFp4Tensor2DDesc,
    /// Packed checkpoint weight bytes `[N,K/2]`.
    pub weight: NvFp4Tensor2DDesc,
    /// Swizzled activation scale bytes.
    pub input_scale: NvFp4Tensor1DDesc,
    /// Swizzled checkpoint weight scale bytes.
    pub weight_scale: NvFp4Tensor1DDesc,
    /// F32 dequantization multiplier `[1]`.
    pub global_scale: NvFp4Tensor1DDesc,
    /// Writable F16/BF16 output `[M,N]`.
    pub output: NvFp4Tensor2DDesc,
    /// Writable U8 scratch allocation.
    pub workspace: NvFp4Tensor1DDesc,
    /// CUTLASS tactic index, or `-1` for FlashInfer's default tactic.
    pub tactic: i64,
    /// CUDA stream used for asynchronous execution.
    pub stream: *mut c_void,
}

impl NvFp4GemmParams {
    /// Validate pointer, shape, dtype, and device contracts without launching CUDA.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_ptrs(&[
            (self.input.ptr, "input"),
            (self.weight.ptr, "weight"),
            (self.input_scale.ptr, "input_scale"),
            (self.weight_scale.ptr, "weight_scale"),
            (self.global_scale.ptr, "global_scale"),
            (self.output.ptr, "output"),
            (self.workspace.ptr, "workspace"),
        ])?;
        if self.input.rows <= 0 || self.input.cols <= 0 {
            return invalid("packed input dimensions must be positive");
        }
        if self.input.dtype != NvFp4DType::U8
            || self.weight.dtype != NvFp4DType::U8
            || self.input_scale.dtype != NvFp4DType::U8
            || self.weight_scale.dtype != NvFp4DType::U8
            || self.workspace.dtype != NvFp4DType::U8
        {
            return invalid("packed values, scales, and workspace must use U8 storage");
        }
        if self.weight.rows <= 0 || self.weight.cols != self.input.cols {
            return invalid("weight must be [N,K/2] with the same packed K as input");
        }
        if !matches!(self.output.dtype, NvFp4DType::F16 | NvFp4DType::BF16)
            || self.output.rows != self.input.rows
            || self.output.cols != self.weight.rows
        {
            return invalid("output must be F16/BF16 [M,N]");
        }
        if self.global_scale.dtype != NvFp4DType::F32 || self.global_scale.len != 1 {
            return invalid("global_scale must be F32 with one element");
        }
        if self.workspace.len <= 0 {
            return invalid("workspace must not be empty");
        }
        let scale_cols = self
            .input
            .cols
            .checked_mul(2)
            .ok_or_else(|| FlashInferError::invalid_argument("packed K overflow"))?
            / 16;
        let input_scale_len = swizzled_scale_len(self.input.rows, scale_cols)?;
        let weight_scale_len = swizzled_scale_len(self.weight.rows, scale_cols)?;
        if self.input_scale.len < input_scale_len || self.weight_scale.len < weight_scale_len {
            return invalid("scale buffer is smaller than its padded 128x4 layout");
        }
        if self.tactic < -1 {
            return invalid("tactic must be -1 or a non-negative index");
        }
        validate_same_device(
            self.input.device_id,
            &[
                self.weight.device_id,
                self.input_scale.device_id,
                self.weight_scale.device_id,
                self.global_scale.device_id,
                self.output.device_id,
                self.workspace.device_id,
            ],
        )
    }
}

/// Return the padded element count for FlashInfer's 128x4 scale layout.
pub fn swizzled_scale_len(rows: i64, scale_cols: i64) -> Result<i64, FlashInferError> {
    if rows <= 0 || scale_cols <= 0 {
        return invalid("scale dimensions must be positive");
    }
    let padded_rows = rows
        .checked_add(127)
        .ok_or_else(|| FlashInferError::invalid_argument("scale row padding overflow"))?
        / 128
        * 128;
    let padded_cols = scale_cols
        .checked_add(3)
        .ok_or_else(|| FlashInferError::invalid_argument("scale column padding overflow"))?
        / 4
        * 4;
    padded_rows
        .checked_mul(padded_cols)
        .ok_or_else(|| FlashInferError::invalid_argument("scale element count overflow"))
}

/// Quantize contiguous F16/BF16 activations into NVFP4 and swizzled E4M3 scales.
pub fn nvfp4_quantize(params: &NvFp4QuantizeParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: validation establishes the typed FlashInfer ABI contract.
    unsafe { quantize_with_runtime(runtime, params) }
}

/// Interleave raw checkpoint E4M3 scale bytes into FlashInfer's 128x4 layout.
pub fn nvfp4_scale_interleave(params: &NvFp4ScaleInterleaveParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: validation establishes the typed FlashInfer ABI contract.
    unsafe { interleave_with_runtime(runtime, params) }
}

/// Multiply packed NVFP4 activations and weights with FlashInfer's SM120 CUTLASS kernel.
pub fn nvfp4_gemm(params: &NvFp4GemmParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: validation establishes the typed FlashInfer ABI contract.
    unsafe { gemm_with_runtime(runtime, params) }
}

/// Return the number of explicit CUTLASS tactics exposed by the loaded FP4 GEMM module.
///
/// This is a candidate count, not a shape-specific validity query or an
/// autotuning result. Valid tactic indices are `0..count`; `-1` remains the
/// module's default configuration.
pub fn nvfp4_gemm_tactic_num() -> Result<i64, FlashInferError> {
    let runtime = FlashInferRuntime::global()?;
    let mut result = any_none();
    // SAFETY: the loaded TVM-FFI function accepts no arguments and writes one
    // integer result into the caller-owned result slot.
    unsafe { runtime.call_fp4_gemm_tactic_num(&mut result)? };
    if result.type_index != KTVM_FFI_INT {
        return invalid("fp4_gemm_tactic_num returned a non-integer result");
    }
    // SAFETY: the type index above identifies the active union field.
    let count = unsafe { result.value.v_int64 };
    if count <= 0 {
        return invalid("fp4_gemm_tactic_num returned a non-positive count");
    }
    Ok(count)
}

/// Quantize flat cudarc buffers representing `[rows,cols]` activations.
///
/// The output and scale buffers must have exact lengths derived from the input.
/// Work is enqueued asynchronously on `stream`.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_quantize_cudarc<T, I, G, O, S>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    global_scale: &G,
    output: &mut O,
    output_scale: &mut S,
    rows: usize,
    cols: usize,
    input_dtype: NvFp4DType,
    enable_pdl: bool,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    G: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
    S: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| FlashInferError::invalid_argument("rows * cols overflow"))?;
    let packed_len = rows
        .checked_mul(cols / 2)
        .ok_or_else(|| FlashInferError::invalid_argument("packed output size overflow"))?;
    let rows_i64 = to_i64(rows, "rows")?;
    let cols_i64 = to_i64(cols, "cols")?;
    let scale_len = usize::try_from(swizzled_scale_len(rows_i64, cols_i64 / 16)?)
        .map_err(|_| FlashInferError::invalid_argument("scale length does not fit usize"))?;
    if input.len() != input_len
        || global_scale.len() != 1
        || output.len() != packed_len
        || output_scale.len() != scale_len
    {
        return invalid("cudarc NVFP4 quantize buffer length mismatch");
    }
    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (global_ptr, _global_sync) = global_scale.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let (scale_ptr, _scale_sync) = output_scale.device_ptr_mut(stream);
    let device_id = device_id(stream)?;
    nvfp4_quantize(&NvFp4QuantizeParams {
        input: NvFp4Tensor2DDesc {
            ptr: input_ptr as usize as *const c_void,
            rows: rows_i64,
            cols: cols_i64,
            dtype: input_dtype,
            device_id,
        },
        global_scale: NvFp4Tensor1DDesc {
            ptr: global_ptr as usize as *const c_void,
            len: 1,
            dtype: NvFp4DType::F32,
            device_id,
        },
        output: NvFp4Tensor2DDesc {
            ptr: output_ptr as usize as *const c_void,
            rows: rows_i64,
            cols: cols_i64 / 2,
            dtype: NvFp4DType::U8,
            device_id,
        },
        output_scale: NvFp4Tensor1DDesc {
            ptr: scale_ptr as usize as *const c_void,
            len: to_i64(scale_len, "scale_len")?,
            dtype: NvFp4DType::U8,
            device_id,
        },
        enable_pdl,
        stream: stream.cu_stream().cast(),
    })
}

/// Interleave a flat cudarc `[rows,cols]` raw E4M3 scale buffer.
///
/// Work is enqueued asynchronously on `stream`.
#[cfg(feature = "cudarc")]
pub fn nvfp4_scale_interleave_cudarc<I, O>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    output: &mut O,
    rows: usize,
    cols: usize,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    O: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| FlashInferError::invalid_argument("rows * cols overflow"))?;
    let rows_i64 = to_i64(rows, "rows")?;
    let cols_i64 = to_i64(cols, "cols")?;
    let output_len = usize::try_from(swizzled_scale_len(rows_i64, cols_i64)?)
        .map_err(|_| FlashInferError::invalid_argument("scale length does not fit usize"))?;
    if input.len() != input_len || output.len() != output_len {
        return invalid("cudarc NVFP4 interleave buffer length mismatch");
    }
    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let device_id = device_id(stream)?;
    nvfp4_scale_interleave(&NvFp4ScaleInterleaveParams {
        input: NvFp4Tensor2DDesc {
            ptr: input_ptr as usize as *const c_void,
            rows: rows_i64,
            cols: cols_i64,
            dtype: NvFp4DType::U8,
            device_id,
        },
        output: NvFp4Tensor1DDesc {
            ptr: output_ptr as usize as *const c_void,
            len: to_i64(output_len, "output_len")?,
            dtype: NvFp4DType::U8,
            device_id,
        },
        stream: stream.cu_stream().cast(),
    })
}

/// Execute dense NVFP4 GEMM from flat cudarc buffers.
///
/// Packed input is `[m,k/2]`, packed weight is `[n,k/2]`, and output is
/// `[m,n]`. Work is enqueued asynchronously on `stream`.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_cudarc<T, I, W, IS, WS, G, O, X>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    weight: &W,
    input_scale: &IS,
    weight_scale: &WS,
    global_scale: &G,
    output: &mut O,
    workspace: &mut X,
    m: usize,
    n: usize,
    k: usize,
    output_dtype: NvFp4DType,
    tactic: i64,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    IS: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    WS: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    G: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    X: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    let packed_input_len = m
        .checked_mul(k / 2)
        .ok_or_else(|| FlashInferError::invalid_argument("packed input size overflow"))?;
    let packed_weight_len = n
        .checked_mul(k / 2)
        .ok_or_else(|| FlashInferError::invalid_argument("packed weight size overflow"))?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| FlashInferError::invalid_argument("output size overflow"))?;
    if input.len() != packed_input_len
        || weight.len() != packed_weight_len
        || global_scale.len() != 1
        || output.len() != output_len
        || workspace.len() == 0
    {
        return invalid("cudarc NVFP4 GEMM buffer length mismatch");
    }
    let workspace_len = workspace.len();
    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (weight_ptr, _weight_sync) = weight.device_ptr(stream);
    let (input_scale_ptr, _input_scale_sync) = input_scale.device_ptr(stream);
    let (weight_scale_ptr, _weight_scale_sync) = weight_scale.device_ptr(stream);
    let (global_ptr, _global_sync) = global_scale.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let (workspace_ptr, _workspace_sync) = workspace.device_ptr_mut(stream);
    let device_id = device_id(stream)?;
    let m = to_i64(m, "m")?;
    let n = to_i64(n, "n")?;
    let packed_k = to_i64(k / 2, "packed_k")?;
    nvfp4_gemm(&NvFp4GemmParams {
        input: desc2(input_ptr, m, packed_k, NvFp4DType::U8, device_id),
        weight: desc2(weight_ptr, n, packed_k, NvFp4DType::U8, device_id),
        input_scale: desc1(
            input_scale_ptr,
            input_scale.len(),
            NvFp4DType::U8,
            device_id,
        )?,
        weight_scale: desc1(
            weight_scale_ptr,
            weight_scale.len(),
            NvFp4DType::U8,
            device_id,
        )?,
        global_scale: desc1(global_ptr, 1, NvFp4DType::F32, device_id)?,
        output: desc2(output_ptr, m, n, output_dtype, device_id),
        workspace: desc1(workspace_ptr, workspace_len, NvFp4DType::U8, device_id)?,
        tactic,
        stream: stream.cu_stream().cast(),
    })
}

#[cfg(feature = "cudarc")]
fn device_id(stream: &cudarc::driver::CudaStream) -> Result<i32, FlashInferError> {
    i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit i32"))
}

#[cfg(feature = "cudarc")]
fn to_i64(value: usize, name: &str) -> Result<i64, FlashInferError> {
    i64::try_from(value)
        .map_err(|_| FlashInferError::invalid_argument(format!("{name} does not fit i64")))
}

#[cfg(feature = "cudarc")]
fn desc1(
    ptr: u64,
    len: usize,
    dtype: NvFp4DType,
    device_id: i32,
) -> Result<NvFp4Tensor1DDesc, FlashInferError> {
    Ok(NvFp4Tensor1DDesc {
        ptr: ptr as usize as *const c_void,
        len: to_i64(len, "descriptor length")?,
        dtype,
        device_id,
    })
}

#[cfg(feature = "cudarc")]
fn desc2(ptr: u64, rows: i64, cols: i64, dtype: NvFp4DType, device_id: i32) -> NvFp4Tensor2DDesc {
    NvFp4Tensor2DDesc {
        ptr: ptr as usize as *const c_void,
        rows,
        cols,
        dtype,
        device_id,
    }
}

// Match fp4Quantize.cpp's host ABI, preserving multiplier and PDL semantics.
fn quantize_args(
    input: &DLTensor,
    global_scale: &DLTensor,
    output: &DLTensor,
    output_scale: &DLTensor,
    enable_pdl: bool,
) -> [TVMFFIAny; 10] {
    [
        any_dltensor_ptr(input),
        any_dltensor_ptr(global_scale),
        any_dltensor_ptr(output),
        any_dltensor_ptr(output_scale),
        any_i64(16),
        any_bool(false),
        any_bool(true),
        any_bool(false),
        // fp4Quantize.cpp: isGlobalScaleInversed precedes enable_pdl.
        // Our global_scale contract supplies the quantization multiplier.
        any_bool(false),
        any_bool(enable_pdl),
    ]
}

unsafe fn quantize_with_runtime(
    runtime: &FlashInferRuntime,
    params: &NvFp4QuantizeParams,
) -> Result<(), FlashInferError> {
    let (input, mut input_shape, mut input_strides) = tensor2d(params.input);
    let (global_scale, mut global_shape, mut global_strides) = tensor1d(params.global_scale);
    let (output, mut output_shape, mut output_strides) = tensor2d(params.output);
    let (output_scale, mut scale_shape, mut scale_strides) = tensor1d(params.output_scale);
    let input = bind_tensor(input, &mut input_shape, &mut input_strides);
    let global_scale = bind_tensor(global_scale, &mut global_shape, &mut global_strides);
    let output = bind_tensor(output, &mut output_shape, &mut output_strides);
    let output_scale = bind_tensor(output_scale, &mut scale_shape, &mut scale_strides);
    let args = quantize_args(
        &input,
        &global_scale,
        &output,
        &output_scale,
        params.enable_pdl,
    );
    unsafe {
        call_on_stream(runtime, params.input.device_id, params.stream, |result| {
            runtime.call_fp4_quantize(args.as_ptr(), args.len() as i32, result)
        })
    }
}

unsafe fn interleave_with_runtime(
    runtime: &FlashInferRuntime,
    params: &NvFp4ScaleInterleaveParams,
) -> Result<(), FlashInferError> {
    let (input, mut input_shape, mut input_strides) = tensor2d(params.input);
    let (output, mut output_shape, mut output_strides) = tensor1d(params.output);
    let input = bind_tensor(input, &mut input_shape, &mut input_strides);
    let output = bind_tensor(output, &mut output_shape, &mut output_strides);
    let args = [any_dltensor_ptr(&input), any_dltensor_ptr(&output)];
    unsafe {
        call_on_stream(runtime, params.input.device_id, params.stream, |result| {
            runtime.call_block_scale_interleave_sm100(args.as_ptr(), args.len() as i32, result)
        })
    }
}

unsafe fn gemm_with_runtime(
    runtime: &FlashInferRuntime,
    params: &NvFp4GemmParams,
) -> Result<(), FlashInferError> {
    let (input, mut input_shape, mut input_strides) = tensor2d(params.input);
    let (weight, mut weight_shape, mut weight_strides) = tensor2d(params.weight);
    let (input_scale, mut input_scale_shape, mut input_scale_strides) =
        tensor1d(params.input_scale);
    let (weight_scale, mut weight_scale_shape, mut weight_scale_strides) =
        tensor1d(params.weight_scale);
    let (global_scale, mut global_shape, mut global_strides) = tensor1d(params.global_scale);
    let (output, mut output_shape, mut output_strides) = tensor2d(params.output);
    let (workspace, mut workspace_shape, mut workspace_strides) = tensor1d(params.workspace);
    let input = bind_tensor(input, &mut input_shape, &mut input_strides);
    let weight = bind_tensor(weight, &mut weight_shape, &mut weight_strides);
    let input_scale = bind_tensor(
        input_scale,
        &mut input_scale_shape,
        &mut input_scale_strides,
    );
    let weight_scale = bind_tensor(
        weight_scale,
        &mut weight_scale_shape,
        &mut weight_scale_strides,
    );
    let global_scale = bind_tensor(global_scale, &mut global_shape, &mut global_strides);
    let output = bind_tensor(output, &mut output_shape, &mut output_strides);
    let workspace = bind_tensor(workspace, &mut workspace_shape, &mut workspace_strides);
    let args = [
        any_dltensor_ptr(&input),
        any_dltensor_ptr(&weight),
        any_dltensor_ptr(&input_scale),
        any_dltensor_ptr(&weight_scale),
        any_dltensor_ptr(&global_scale),
        any_dltensor_ptr(&output),
        any_dltensor_ptr(&workspace),
        any_i64(params.tactic),
    ];
    unsafe {
        call_on_stream(runtime, params.input.device_id, params.stream, |result| {
            runtime.call_fp4_gemm(args.as_ptr(), args.len() as i32, result)
        })
    }
}

unsafe fn call_on_stream(
    runtime: &FlashInferRuntime,
    device_id: i32,
    stream: *mut c_void,
    call: impl FnOnce(*mut TVMFFIAny) -> Result<(), FlashInferError>,
) -> Result<(), FlashInferError> {
    let previous = unsafe { runtime.set_stream(device_id, stream)? };
    let mut result = any_none();
    let call_result = call(&mut result);
    let restore_result = unsafe { runtime.restore_stream(device_id, previous) };
    match (call_result, restore_result) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn tensor1d(desc: NvFp4Tensor1DDesc) -> (DLTensor, [i64; 1], [i64; 1]) {
    (
        base_tensor(desc.ptr, desc.dtype, desc.device_id, 1),
        [desc.len],
        [1],
    )
}

fn tensor2d(desc: NvFp4Tensor2DDesc) -> (DLTensor, [i64; 2], [i64; 2]) {
    (
        base_tensor(desc.ptr, desc.dtype, desc.device_id, 2),
        [desc.rows, desc.cols],
        [desc.cols, 1],
    )
}

fn base_tensor(ptr: *const c_void, dtype: NvFp4DType, device_id: i32, ndim: i32) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim,
        dtype: dtype.as_dl_dtype(),
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    }
}

fn bind_tensor<const N: usize>(
    mut tensor: DLTensor,
    shape: &mut [i64; N],
    strides: &mut [i64; N],
) -> DLTensor {
    tensor.shape = shape.as_mut_ptr();
    tensor.strides = strides.as_mut_ptr();
    tensor
}

fn validate_ptrs(ptrs: &[(*const c_void, &str)]) -> Result<(), FlashInferError> {
    for (ptr, name) in ptrs {
        if ptr.is_null() {
            return invalid(format!("{name} pointer is null"));
        }
    }
    Ok(())
}

fn validate_same_device(reference: i32, others: &[i32]) -> Result<(), FlashInferError> {
    if reference < 0 || others.iter().any(|device| *device != reference) {
        return invalid("all tensors must use the same non-negative CUDA device");
    }
    Ok(())
}

fn invalid<T>(message: impl Into<String>) -> Result<T, FlashInferError> {
    Err(FlashInferError::invalid_argument(message))
}

#[cfg(test)]
mod tests {
    use super::{NvFp4DType, NvFp4Tensor2DDesc, quantize_args, swizzled_scale_len, tensor2d};

    #[test]
    fn quantize_abi_preserves_global_scale_and_pdl_positions() {
        let (tensor, _, _) = tensor2d(NvFp4Tensor2DDesc {
            ptr: std::ptr::null(),
            rows: 1,
            cols: 16,
            dtype: NvFp4DType::BF16,
            device_id: 0,
        });
        for pdl in [false, true] {
            let args = quantize_args(&tensor, &tensor, &tensor, &tensor, pdl);
            assert_eq!(args.len(), 10);
            // These slots are constructed by any_i64/any_bool, so reading the
            // integer union member is valid and requires no tensor access.
            unsafe {
                assert_eq!(args[4].value.v_int64, 16);
                assert_eq!(args[8].value.v_int64, 0);
                assert_eq!(args[9].value.v_int64, i64::from(pdl));
            }
        }
    }

    #[test]
    fn scale_layout_pads_rows_and_columns() {
        assert_eq!(swizzled_scale_len(1, 5).unwrap(), 128 * 8);
        assert_eq!(swizzled_scale_len(128, 320).unwrap(), 128 * 320);
        assert_eq!(swizzled_scale_len(129, 3).unwrap(), 256 * 4);
    }
}
