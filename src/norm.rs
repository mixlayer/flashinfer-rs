use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_FLOAT, TVMFFIAny, any_bool,
    any_dltensor_ptr, any_f64, any_none,
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
        if self.input.ptr.is_null() {
            return Err(FlashInferError::invalid_argument("input pointer is null"));
        }
        if self.weight.ptr.is_null() {
            return Err(FlashInferError::invalid_argument("weight pointer is null"));
        }
        if self.out.ptr.is_null() {
            return Err(FlashInferError::invalid_argument("out pointer is null"));
        }
        if self.input.rows <= 0 || self.input.cols <= 0 {
            return Err(FlashInferError::invalid_argument(
                "input shape must be positive",
            ));
        }
        if self.weight.len <= 0 {
            return Err(FlashInferError::invalid_argument(
                "weight length must be positive",
            ));
        }
        if self.out.rows <= 0 || self.out.cols <= 0 {
            return Err(FlashInferError::invalid_argument(
                "out shape must be positive",
            ));
        }
        if self.input.cols != self.weight.len {
            return Err(FlashInferError::invalid_argument(format!(
                "shape mismatch: input.cols ({}) must equal weight.len ({})",
                self.input.cols, self.weight.len
            )));
        }
        if self.out.rows != self.input.rows || self.out.cols != self.input.cols {
            return Err(FlashInferError::invalid_argument(format!(
                "shape mismatch: out ({}, {}) must match input ({}, {})",
                self.out.rows, self.out.cols, self.input.rows, self.input.cols
            )));
        }
        if self.input.dtype != self.weight.dtype || self.input.dtype != self.out.dtype {
            return Err(FlashInferError::invalid_argument(
                "dtype mismatch across input/weight/out",
            ));
        }
        if self.input.stride_col != 1 {
            return Err(FlashInferError::invalid_argument(
                "input last-dimension stride must be 1",
            ));
        }
        if self.out.stride_col != 1 {
            return Err(FlashInferError::invalid_argument(
                "out last-dimension stride must be 1",
            ));
        }
        if self.weight.stride != 1 {
            return Err(FlashInferError::invalid_argument("weight stride must be 1"));
        }
        if self.input.device_id != self.out.device_id
            || self.input.device_id != self.weight.device_id
        {
            return Err(FlashInferError::invalid_argument(
                "device mismatch across input/weight/out",
            ));
        }
        if !self.eps.is_finite() {
            return Err(FlashInferError::invalid_argument(
                "eps must be a finite value",
            ));
        }
        Ok(())
    }
}

pub fn gemma_rmsnorm(params: &GemmaRmsNormParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: all FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { gemma_rmsnorm_with_runtime(runtime, params) }
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

    fn valid_params() -> GemmaRmsNormParams {
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
    fn validate_rejects_dtype_mismatch() {
        let mut params = valid_params();
        params.weight.dtype = DType::BF16;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_shape_mismatch() {
        let mut params = valid_params();
        params.weight.len = 5;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_last_dim_stride_mismatch() {
        let mut params = valid_params();
        params.input.stride_col = 2;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_device_mismatch() {
        let mut params = valid_params();
        params.out.device_id = 1;
        assert!(params.validate().is_err());
    }
}
