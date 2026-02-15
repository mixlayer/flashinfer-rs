use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    any_bool, any_dltensor_ptr, any_f64, any_none, DLDataType, DLDevice, DLTensor, TVMFFIAny,
    KDL_BFLOAT, KDL_CUDA, KDL_FLOAT,
};
use crate::runtime::FlashInferRuntime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    BF16,
}

impl DType {
    fn as_dl_dtype(self) -> DLDataType {
        match self {
            DType::F16 => DLDataType {
                code: KDL_FLOAT,
                bits: 16,
                lanes: 1,
            },
            DType::BF16 => DLDataType {
                code: KDL_BFLOAT,
                bits: 16,
                lanes: 1,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Tensor1DDesc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub dtype: DType,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Tensor2DDesc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub dtype: DType,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Tensor3DDesc {
    pub ptr: *const c_void,
    pub dim0: i64,
    pub dim1: i64,
    pub dim2: i64,
    pub stride0: i64,
    pub stride1: i64,
    pub stride2: i64,
    pub dtype: DType,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct RmsNormParams {
    /// Input activations, rank-2: `[batch_size, hidden_size]`.
    pub input: Tensor2DDesc,
    /// RMSNorm weights, rank-1: `[hidden_size]`.
    pub weight: Tensor1DDesc,
    /// Output activations, rank-2: `[batch_size, hidden_size]`.
    pub out: Tensor2DDesc,
    /// Numerical stability epsilon used in `rsqrt(mean(x^2) + eps)`.
    pub eps: f64,
    /// Whether to enable PDL mode in the FlashInfer kernel.
    pub enable_pdl: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl RmsNormParams {
    pub fn new(
        input: Tensor2DDesc,
        weight: Tensor1DDesc,
        out: Tensor2DDesc,
        eps: f64,
        stream: *mut c_void,
    ) -> Self {
        Self {
            input,
            weight,
            out,
            eps,
            enable_pdl: false,
            stream,
        }
    }

    pub fn with_enable_pdl(mut self, enable_pdl: bool) -> Self {
        self.enable_pdl = enable_pdl;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_rmsnorm_2d(self.input, self.weight, self.out, self.eps, self.enable_pdl)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FusedQkRmsNormParams {
    /// Input activations, rank-3: `[batch_size, num_heads, head_dim]`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/norm.cu::rmsnorm` 3D branch (`QKRMSNorm` path).
    pub input: Tensor3DDesc,
    /// RMSNorm weights, rank-1: `[head_dim]`.
    pub weight: Tensor1DDesc,
    /// Output activations, rank-3: `[batch_size, num_heads, head_dim]`.
    pub out: Tensor3DDesc,
    /// Numerical stability epsilon used in `rsqrt(mean(x^2) + eps)`.
    pub eps: f64,
    /// Whether to enable PDL mode in the FlashInfer kernel.
    pub enable_pdl: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl FusedQkRmsNormParams {
    pub fn new(
        input: Tensor3DDesc,
        weight: Tensor1DDesc,
        out: Tensor3DDesc,
        eps: f64,
        stream: *mut c_void,
    ) -> Self {
        Self {
            input,
            weight,
            out,
            eps,
            enable_pdl: false,
            stream,
        }
    }

    pub fn with_enable_pdl(mut self, enable_pdl: bool) -> Self {
        self.enable_pdl = enable_pdl;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_rmsnorm_3d(self.input, self.weight, self.out, self.eps, self.enable_pdl)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GemmaRmsNormParams {
    /// Input activations, rank-2: `[batch_size, hidden_size]`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/norm.cu::gemma_rmsnorm` (`CHECK_DIM(2, input)`).
    pub input: Tensor2DDesc,
    /// RMSNorm weights, rank-1: `[hidden_size]`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/norm.cu::gemma_rmsnorm` (`CHECK_DIM(1, weight)`).
    pub weight: Tensor1DDesc,
    /// Output activations, rank-2: `[batch_size, hidden_size]`, same shape as `input`.
    pub out: Tensor2DDesc,
    /// Numerical stability epsilon used in `rsqrt(mean(x^2) + eps)`.
    pub eps: f64,
    /// Whether to enable PDL mode in the FlashInfer kernel.
    pub enable_pdl: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl GemmaRmsNormParams {
    pub fn new(
        input: Tensor2DDesc,
        weight: Tensor1DDesc,
        out: Tensor2DDesc,
        eps: f64,
        stream: *mut c_void,
    ) -> Self {
        Self {
            input,
            weight,
            out,
            eps,
            enable_pdl: false,
            stream,
        }
    }

    pub fn with_enable_pdl(mut self, enable_pdl: bool) -> Self {
        self.enable_pdl = enable_pdl;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_rmsnorm_2d(self.input, self.weight, self.out, self.eps, self.enable_pdl)
    }
}

pub fn rmsnorm(params: &RmsNormParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: all FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { rmsnorm_2d_with_runtime(runtime, params) }
}

pub fn fused_qk_rmsnorm(params: &FusedQkRmsNormParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: all FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { rmsnorm_3d_with_runtime(runtime, params) }
}

pub fn qk_rmsnorm(params: &FusedQkRmsNormParams) -> Result<(), FlashInferError> {
    fused_qk_rmsnorm(params)
}

pub fn gemma_rmsnorm(params: &GemmaRmsNormParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: all FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { gemma_rmsnorm_with_runtime(runtime, params) }
}

unsafe fn rmsnorm_2d_with_runtime(
    runtime: &FlashInferRuntime,
    params: &RmsNormParams,
) -> Result<(), FlashInferError> {
    let mut input_shape = [params.input.rows, params.input.cols];
    let mut input_strides = [params.input.stride_row, params.input.stride_col];
    let input_tensor = DLTensor {
        data: params.input.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.input.device_id,
        },
        ndim: 2,
        dtype: params.input.dtype.as_dl_dtype(),
        shape: input_shape.as_mut_ptr(),
        strides: input_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut out_shape = [params.out.rows, params.out.cols];
    let mut out_strides = [params.out.stride_row, params.out.stride_col];
    let out_tensor = DLTensor {
        data: params.out.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.out.device_id,
        },
        ndim: 2,
        dtype: params.out.dtype.as_dl_dtype(),
        shape: out_shape.as_mut_ptr(),
        strides: out_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut weight_shape = [params.weight.len];
    let mut weight_strides = [params.weight.stride];
    let weight_tensor = DLTensor {
        data: params.weight.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.weight.device_id,
        },
        ndim: 1,
        dtype: params.weight.dtype.as_dl_dtype(),
        shape: weight_shape.as_mut_ptr(),
        strides: weight_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let args: [TVMFFIAny; 5] = [
        any_dltensor_ptr(&out_tensor),
        any_dltensor_ptr(&input_tensor),
        any_dltensor_ptr(&weight_tensor),
        any_f64(params.eps),
        any_bool(params.enable_pdl),
    ];
    let mut result = any_none();

    // SAFETY: stream context API contract comes from tvm ffi and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.input.device_id, params.stream)? };
    let mut restore_guard =
        StreamRestoreGuard::new(runtime, params.input.device_id, previous_stream);
    // SAFETY: argument packing follows TVMFFIAny ABI.
    let call_result =
        unsafe { runtime.call_rmsnorm(args.as_ptr(), args.len() as i32, &mut result as *mut _) };
    let restore_result = restore_guard.restore_now();

    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

unsafe fn rmsnorm_3d_with_runtime(
    runtime: &FlashInferRuntime,
    params: &FusedQkRmsNormParams,
) -> Result<(), FlashInferError> {
    let mut input_shape = [params.input.dim0, params.input.dim1, params.input.dim2];
    let mut input_strides = [
        params.input.stride0,
        params.input.stride1,
        params.input.stride2,
    ];
    let input_tensor = DLTensor {
        data: params.input.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.input.device_id,
        },
        ndim: 3,
        dtype: params.input.dtype.as_dl_dtype(),
        shape: input_shape.as_mut_ptr(),
        strides: input_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut out_shape = [params.out.dim0, params.out.dim1, params.out.dim2];
    let mut out_strides = [params.out.stride0, params.out.stride1, params.out.stride2];
    let out_tensor = DLTensor {
        data: params.out.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.out.device_id,
        },
        ndim: 3,
        dtype: params.out.dtype.as_dl_dtype(),
        shape: out_shape.as_mut_ptr(),
        strides: out_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut weight_shape = [params.weight.len];
    let mut weight_strides = [params.weight.stride];
    let weight_tensor = DLTensor {
        data: params.weight.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.weight.device_id,
        },
        ndim: 1,
        dtype: params.weight.dtype.as_dl_dtype(),
        shape: weight_shape.as_mut_ptr(),
        strides: weight_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let args: [TVMFFIAny; 5] = [
        any_dltensor_ptr(&out_tensor),
        any_dltensor_ptr(&input_tensor),
        any_dltensor_ptr(&weight_tensor),
        any_f64(params.eps),
        any_bool(params.enable_pdl),
    ];
    let mut result = any_none();

    // SAFETY: stream context API contract comes from tvm ffi and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.input.device_id, params.stream)? };
    let mut restore_guard =
        StreamRestoreGuard::new(runtime, params.input.device_id, previous_stream);
    // SAFETY: argument packing follows TVMFFIAny ABI.
    let call_result =
        unsafe { runtime.call_rmsnorm(args.as_ptr(), args.len() as i32, &mut result as *mut _) };
    let restore_result = restore_guard.restore_now();

    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

unsafe fn gemma_rmsnorm_with_runtime(
    runtime: &FlashInferRuntime,
    params: &GemmaRmsNormParams,
) -> Result<(), FlashInferError> {
    let mut input_shape = [params.input.rows, params.input.cols];
    let mut input_strides = [params.input.stride_row, params.input.stride_col];
    let input_tensor = DLTensor {
        data: params.input.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.input.device_id,
        },
        ndim: 2,
        dtype: params.input.dtype.as_dl_dtype(),
        shape: input_shape.as_mut_ptr(),
        strides: input_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut out_shape = [params.out.rows, params.out.cols];
    let mut out_strides = [params.out.stride_row, params.out.stride_col];
    let out_tensor = DLTensor {
        data: params.out.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.out.device_id,
        },
        ndim: 2,
        dtype: params.out.dtype.as_dl_dtype(),
        shape: out_shape.as_mut_ptr(),
        strides: out_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut weight_shape = [params.weight.len];
    let mut weight_strides = [params.weight.stride];
    let weight_tensor = DLTensor {
        data: params.weight.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.weight.device_id,
        },
        ndim: 1,
        dtype: params.weight.dtype.as_dl_dtype(),
        shape: weight_shape.as_mut_ptr(),
        strides: weight_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let args: [TVMFFIAny; 5] = [
        any_dltensor_ptr(&out_tensor),
        any_dltensor_ptr(&input_tensor),
        any_dltensor_ptr(&weight_tensor),
        any_f64(params.eps),
        any_bool(params.enable_pdl),
    ];
    let mut result = any_none();

    // SAFETY: stream context API contract comes from tvm ffi and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.input.device_id, params.stream)? };
    let mut restore_guard =
        StreamRestoreGuard::new(runtime, params.input.device_id, previous_stream);
    // SAFETY: argument packing follows TVMFFIAny ABI.
    let call_result = unsafe {
        runtime.call_gemma_rmsnorm(args.as_ptr(), args.len() as i32, &mut result as *mut _)
    };
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

fn validate_rmsnorm_2d(
    input: Tensor2DDesc,
    weight: Tensor1DDesc,
    out: Tensor2DDesc,
    eps: f64,
    _enable_pdl: bool,
) -> Result<(), FlashInferError> {
    if input.ptr.is_null() {
        return Err(FlashInferError::invalid_argument("input pointer is null"));
    }
    if weight.ptr.is_null() {
        return Err(FlashInferError::invalid_argument("weight pointer is null"));
    }
    if out.ptr.is_null() {
        return Err(FlashInferError::invalid_argument("out pointer is null"));
    }
    if input.rows <= 0 || input.cols <= 0 {
        return Err(FlashInferError::invalid_argument(
            "input shape must be positive",
        ));
    }
    if weight.len <= 0 {
        return Err(FlashInferError::invalid_argument(
            "weight length must be positive",
        ));
    }
    if out.rows <= 0 || out.cols <= 0 {
        return Err(FlashInferError::invalid_argument(
            "out shape must be positive",
        ));
    }
    if input.cols != weight.len {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: input.cols ({}) must equal weight.len ({})",
            input.cols, weight.len
        )));
    }
    if out.rows != input.rows || out.cols != input.cols {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: out ({}, {}) must match input ({}, {})",
            out.rows, out.cols, input.rows, input.cols
        )));
    }
    if input.dtype != weight.dtype || input.dtype != out.dtype {
        return Err(FlashInferError::invalid_argument(
            "dtype mismatch across input/weight/out",
        ));
    }
    if input.stride_col != 1 {
        return Err(FlashInferError::invalid_argument(
            "input last-dimension stride must be 1",
        ));
    }
    if out.stride_col != 1 {
        return Err(FlashInferError::invalid_argument(
            "out last-dimension stride must be 1",
        ));
    }
    if weight.stride != 1 {
        return Err(FlashInferError::invalid_argument("weight stride must be 1"));
    }
    if input.device_id != out.device_id || input.device_id != weight.device_id {
        return Err(FlashInferError::invalid_argument(
            "device mismatch across input/weight/out",
        ));
    }
    if !eps.is_finite() {
        return Err(FlashInferError::invalid_argument(
            "eps must be a finite value",
        ));
    }
    Ok(())
}

fn validate_rmsnorm_3d(
    input: Tensor3DDesc,
    weight: Tensor1DDesc,
    out: Tensor3DDesc,
    eps: f64,
    _enable_pdl: bool,
) -> Result<(), FlashInferError> {
    if input.ptr.is_null() {
        return Err(FlashInferError::invalid_argument("input pointer is null"));
    }
    if weight.ptr.is_null() {
        return Err(FlashInferError::invalid_argument("weight pointer is null"));
    }
    if out.ptr.is_null() {
        return Err(FlashInferError::invalid_argument("out pointer is null"));
    }
    if input.dim0 <= 0 || input.dim1 <= 0 || input.dim2 <= 0 {
        return Err(FlashInferError::invalid_argument(
            "input shape must be positive",
        ));
    }
    if weight.len <= 0 {
        return Err(FlashInferError::invalid_argument(
            "weight length must be positive",
        ));
    }
    if out.dim0 <= 0 || out.dim1 <= 0 || out.dim2 <= 0 {
        return Err(FlashInferError::invalid_argument(
            "out shape must be positive",
        ));
    }
    if input.dim2 != weight.len {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: input.dim2 ({}) must equal weight.len ({})",
            input.dim2, weight.len
        )));
    }
    if out.dim0 != input.dim0 || out.dim1 != input.dim1 || out.dim2 != input.dim2 {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: out ({}, {}, {}) must match input ({}, {}, {})",
            out.dim0, out.dim1, out.dim2, input.dim0, input.dim1, input.dim2
        )));
    }
    if input.dtype != weight.dtype || input.dtype != out.dtype {
        return Err(FlashInferError::invalid_argument(
            "dtype mismatch across input/weight/out",
        ));
    }
    if input.stride2 != 1 {
        return Err(FlashInferError::invalid_argument(
            "input last-dimension stride must be 1",
        ));
    }
    if out.stride2 != 1 {
        return Err(FlashInferError::invalid_argument(
            "out last-dimension stride must be 1",
        ));
    }
    if weight.stride != 1 {
        return Err(FlashInferError::invalid_argument("weight stride must be 1"));
    }
    if input.device_id != out.device_id || input.device_id != weight.device_id {
        return Err(FlashInferError::invalid_argument(
            "device mismatch across input/weight/out",
        ));
    }
    if !eps.is_finite() {
        return Err(FlashInferError::invalid_argument(
            "eps must be a finite value",
        ));
    }
    Ok(())
}

#[cfg(feature = "cudarc")]
pub fn rmsnorm_cudarc<T, I, W, O>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    weight: &W,
    out: &mut O,
    rows: usize,
    cols: usize,
    dtype: DType,
    eps: f64,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    W: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    rmsnorm_cudarc_with_options(stream, input, weight, out, rows, cols, dtype, eps, false)
}

#[cfg(feature = "cudarc")]
pub fn rmsnorm_cudarc_with_options<T, I, W, O>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    weight: &W,
    out: &mut O,
    rows: usize,
    cols: usize,
    dtype: DType,
    eps: f64,
    enable_pdl: bool,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    W: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    let expected_input = rows
        .checked_mul(cols)
        .ok_or_else(|| FlashInferError::invalid_argument("rows * cols overflow"))?;

    if input.len() != expected_input {
        return Err(FlashInferError::invalid_argument(format!(
            "input length ({}) must equal rows * cols ({expected_input})",
            input.len()
        )));
    }
    if out.len() != expected_input {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal rows * cols ({expected_input})",
            out.len()
        )));
    }
    if weight.len() != cols {
        return Err(FlashInferError::invalid_argument(format!(
            "weight length ({}) must equal cols ({cols})",
            weight.len()
        )));
    }

    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (weight_ptr, _weight_sync) = weight.device_ptr(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);

    let rows_i64 = i64::try_from(rows)
        .map_err(|_| FlashInferError::invalid_argument("rows does not fit in i64"))?;
    let cols_i64 = i64::try_from(cols)
        .map_err(|_| FlashInferError::invalid_argument("cols does not fit in i64"))?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let params = RmsNormParams::new(
        Tensor2DDesc {
            ptr: input_ptr as usize as *const c_void,
            rows: rows_i64,
            cols: cols_i64,
            stride_row: cols_i64,
            stride_col: 1,
            dtype,
            device_id,
        },
        Tensor1DDesc {
            ptr: weight_ptr as usize as *const c_void,
            len: cols_i64,
            stride: 1,
            dtype,
            device_id,
        },
        Tensor2DDesc {
            ptr: out_ptr as usize as *const c_void,
            rows: rows_i64,
            cols: cols_i64,
            stride_row: cols_i64,
            stride_col: 1,
            dtype,
            device_id,
        },
        eps,
        stream.cu_stream().cast(),
    )
    .with_enable_pdl(enable_pdl);

    rmsnorm(&params)
}

#[cfg(feature = "cudarc")]
pub fn fused_qk_rmsnorm_cudarc<T, I, W, O>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    weight: &W,
    out: &mut O,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    dtype: DType,
    eps: f64,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    W: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    fused_qk_rmsnorm_cudarc_with_options(
        stream, input, weight, out, batch_size, num_heads, head_dim, dtype, eps, false,
    )
}

#[cfg(feature = "cudarc")]
pub fn qk_rmsnorm_cudarc<T, I, W, O>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    weight: &W,
    out: &mut O,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    dtype: DType,
    eps: f64,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    W: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    fused_qk_rmsnorm_cudarc(
        stream, input, weight, out, batch_size, num_heads, head_dim, dtype, eps,
    )
}

#[cfg(feature = "cudarc")]
pub fn fused_qk_rmsnorm_cudarc_with_options<T, I, W, O>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    weight: &W,
    out: &mut O,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    dtype: DType,
    eps: f64,
    enable_pdl: bool,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    W: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    let expected = batch_size
        .checked_mul(num_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| {
            FlashInferError::invalid_argument("batch_size * num_heads * head_dim overflow")
        })?;

    if input.len() != expected {
        return Err(FlashInferError::invalid_argument(format!(
            "input length ({}) must equal batch_size * num_heads * head_dim ({expected})",
            input.len()
        )));
    }
    if out.len() != expected {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal batch_size * num_heads * head_dim ({expected})",
            out.len()
        )));
    }
    if weight.len() != head_dim {
        return Err(FlashInferError::invalid_argument(format!(
            "weight length ({}) must equal head_dim ({head_dim})",
            weight.len()
        )));
    }

    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (weight_ptr, _weight_sync) = weight.device_ptr(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);

    let batch_size_i64 = i64::try_from(batch_size)
        .map_err(|_| FlashInferError::invalid_argument("batch_size does not fit in i64"))?;
    let num_heads_i64 = i64::try_from(num_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_heads does not fit in i64"))?;
    let head_dim_i64 = i64::try_from(head_dim)
        .map_err(|_| FlashInferError::invalid_argument("head_dim does not fit in i64"))?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let stride1 = head_dim_i64;
    let stride0 = num_heads_i64
        .checked_mul(head_dim_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("stride overflow"))?;

    let params = FusedQkRmsNormParams::new(
        Tensor3DDesc {
            ptr: input_ptr as usize as *const c_void,
            dim0: batch_size_i64,
            dim1: num_heads_i64,
            dim2: head_dim_i64,
            stride0,
            stride1,
            stride2: 1,
            dtype,
            device_id,
        },
        Tensor1DDesc {
            ptr: weight_ptr as usize as *const c_void,
            len: head_dim_i64,
            stride: 1,
            dtype,
            device_id,
        },
        Tensor3DDesc {
            ptr: out_ptr as usize as *const c_void,
            dim0: batch_size_i64,
            dim1: num_heads_i64,
            dim2: head_dim_i64,
            stride0,
            stride1,
            stride2: 1,
            dtype,
            device_id,
        },
        eps,
        stream.cu_stream().cast(),
    )
    .with_enable_pdl(enable_pdl);

    fused_qk_rmsnorm(&params)
}

#[cfg(feature = "cudarc")]
pub fn gemma_rmsnorm_cudarc<T, I, W, O>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    weight: &W,
    out: &mut O,
    rows: usize,
    cols: usize,
    dtype: DType,
    eps: f64,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    W: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    gemma_rmsnorm_cudarc_with_options(stream, input, weight, out, rows, cols, dtype, eps, false)
}

#[cfg(feature = "cudarc")]
pub fn gemma_rmsnorm_cudarc_with_options<T, I, W, O>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    weight: &W,
    out: &mut O,
    rows: usize,
    cols: usize,
    dtype: DType,
    eps: f64,
    enable_pdl: bool,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    W: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    let expected_input = rows
        .checked_mul(cols)
        .ok_or_else(|| FlashInferError::invalid_argument("rows * cols overflow"))?;

    if input.len() != expected_input {
        return Err(FlashInferError::invalid_argument(format!(
            "input length ({}) must equal rows * cols ({expected_input})",
            input.len()
        )));
    }
    if out.len() != expected_input {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal rows * cols ({expected_input})",
            out.len()
        )));
    }
    if weight.len() != cols {
        return Err(FlashInferError::invalid_argument(format!(
            "weight length ({}) must equal cols ({cols})",
            weight.len()
        )));
    }

    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (weight_ptr, _weight_sync) = weight.device_ptr(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);

    let rows_i64 = i64::try_from(rows)
        .map_err(|_| FlashInferError::invalid_argument("rows does not fit in i64"))?;
    let cols_i64 = i64::try_from(cols)
        .map_err(|_| FlashInferError::invalid_argument("cols does not fit in i64"))?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let params = GemmaRmsNormParams::new(
        Tensor2DDesc {
            ptr: input_ptr as usize as *const c_void,
            rows: rows_i64,
            cols: cols_i64,
            stride_row: cols_i64,
            stride_col: 1,
            dtype,
            device_id,
        },
        Tensor1DDesc {
            ptr: weight_ptr as usize as *const c_void,
            len: cols_i64,
            stride: 1,
            dtype,
            device_id,
        },
        Tensor2DDesc {
            ptr: out_ptr as usize as *const c_void,
            rows: rows_i64,
            cols: cols_i64,
            stride_row: cols_i64,
            stride_col: 1,
            dtype,
            device_id,
        },
        eps,
        stream.cu_stream().cast(),
    )
    .with_enable_pdl(enable_pdl);

    gemma_rmsnorm(&params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_null() -> *const c_void {
        std::ptr::NonNull::<u8>::dangling().as_ptr().cast()
    }

    fn valid_rms_params() -> RmsNormParams {
        RmsNormParams::new(
            Tensor2DDesc {
                ptr: non_null(),
                rows: 2,
                cols: 4,
                stride_row: 4,
                stride_col: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor1DDesc {
                ptr: non_null(),
                len: 4,
                stride: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor2DDesc {
                ptr: non_null(),
                rows: 2,
                cols: 4,
                stride_row: 4,
                stride_col: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            1e-6,
            std::ptr::null_mut(),
        )
    }

    fn valid_qk_params() -> FusedQkRmsNormParams {
        FusedQkRmsNormParams::new(
            Tensor3DDesc {
                ptr: non_null(),
                dim0: 2,
                dim1: 8,
                dim2: 4,
                stride0: 32,
                stride1: 4,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor1DDesc {
                ptr: non_null(),
                len: 4,
                stride: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor3DDesc {
                ptr: non_null(),
                dim0: 2,
                dim1: 8,
                dim2: 4,
                stride0: 32,
                stride1: 4,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            1e-6,
            std::ptr::null_mut(),
        )
    }

    fn valid_gemma_params() -> GemmaRmsNormParams {
        GemmaRmsNormParams::new(
            Tensor2DDesc {
                ptr: non_null(),
                rows: 2,
                cols: 4,
                stride_row: 4,
                stride_col: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor1DDesc {
                ptr: non_null(),
                len: 4,
                stride: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor2DDesc {
                ptr: non_null(),
                rows: 2,
                cols: 4,
                stride_row: 4,
                stride_col: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            1e-6,
            std::ptr::null_mut(),
        )
    }

    #[test]
    fn rms_validate_rejects_dtype_mismatch() {
        let mut params = valid_rms_params();
        params.weight.dtype = DType::BF16;
        assert!(params.validate().is_err());
    }

    #[test]
    fn rms_validate_rejects_shape_mismatch() {
        let mut params = valid_rms_params();
        params.weight.len = 5;
        assert!(params.validate().is_err());
    }

    #[test]
    fn rms_validate_rejects_last_dim_stride_mismatch() {
        let mut params = valid_rms_params();
        params.input.stride_col = 2;
        assert!(params.validate().is_err());
    }

    #[test]
    fn qk_validate_rejects_last_dim_stride_mismatch() {
        let mut params = valid_qk_params();
        params.input.stride2 = 2;
        assert!(params.validate().is_err());
    }

    #[test]
    fn qk_validate_rejects_shape_mismatch() {
        let mut params = valid_qk_params();
        params.weight.len = 8;
        assert!(params.validate().is_err());
    }

    #[test]
    fn gemma_validate_rejects_device_mismatch() {
        let mut params = valid_gemma_params();
        params.out.device_id = 1;
        assert!(params.validate().is_err());
    }
}
