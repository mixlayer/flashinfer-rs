use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_FLOAT, KTVM_FFI_INT, TVMFFIAny,
    any_bool, any_dltensor_ptr, any_i64, any_none,
};
use crate::norm::DType;
use crate::runtime::FlashInferRuntime;

const TGV_GEMM_FP16_KERNEL_URI: &str = "tgv_gemm_fp16";
const TGV_GEMM_BF16_KERNEL_URI: &str = "tgv_gemm_bf16";
const TRTLLM_GEMM_KERNEL_URI: &str = "trtllm_gemm";
const TRTLLM_LOW_LATENCY_GEMM_KERNEL_URI: &str = "trtllm_low_latency_gemm";

#[derive(Debug, Clone, Copy)]
pub struct GemmTensor1DDesc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub dtype: DType,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct GemmTensor2DDesc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub dtype: DType,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct TgvGemmParams {
    /// Left input matrix A in MM semantics, shape `[m, k]`, row-major (`stride_col == 1`).
    ///
    /// Cross-reference:
    /// `flashinfer/flashinfer/gemm/gemm_base.py::tgv_gemm_runner` (`module.tgv_gemm(b.t(), a.t(), ...)`).
    pub a: GemmTensor2DDesc,
    /// Right input matrix B in MM semantics, shape `[k, n]`, column-major (`stride_row == 1`).
    ///
    /// Cross-reference:
    /// `flashinfer/flashinfer/gemm/gemm_base.py::mm_bf16` docs and TGV call mapping.
    pub b: GemmTensor2DDesc,
    /// Optional bias vector added on output columns, shape `[n]`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/tgv_gemm.cu::tgv_gemm` (`bias` is 1D and size must match output columns).
    pub bias: Option<GemmTensor1DDesc>,
    /// Output matrix in MM semantics, shape `[m, n]`, row-major (`stride_col == 1`).
    pub out: GemmTensor2DDesc,
    /// Kernel tactic id (`-1` lets kernel pick its default).
    pub tactic: i64,
    /// Whether to enable PDL mode.
    pub enable_pdl: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl TgvGemmParams {
    pub fn new(
        a: GemmTensor2DDesc,
        b: GemmTensor2DDesc,
        out: GemmTensor2DDesc,
        stream: *mut c_void,
    ) -> Self {
        Self {
            a,
            b,
            bias: None,
            out,
            tactic: -1,
            enable_pdl: false,
            stream,
        }
    }

    pub fn with_bias(mut self, bias: GemmTensor1DDesc) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn with_tactic(mut self, tactic: i64) -> Self {
        self.tactic = tactic;
        self
    }

    pub fn with_enable_pdl(mut self, enable_pdl: bool) -> Self {
        self.enable_pdl = enable_pdl;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_tgv_gemm(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum TrtllmInputDType {
    E2m1 = 0,
    E4m3 = 1,
}

impl TrtllmInputDType {
    fn as_i64(self) -> i64 {
        self as i64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum TrtllmOutputDType {
    Bf16 = 2,
}

impl TrtllmOutputDType {
    fn as_i64(self) -> i64 {
        self as i64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrtllmGemmTacticsQuery {
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub input_dtype: TrtllmInputDType,
    pub output_dtype: TrtllmOutputDType,
    pub use_8x4_sf_layout: bool,
}

impl TrtllmGemmTacticsQuery {
    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_positive("m", self.m)?;
        check_positive("n", self.n)?;
        check_positive("k", self.k)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrtllmLowLatencyGemmTacticsQuery {
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub input_dtype: TrtllmInputDType,
    pub output_dtype: TrtllmOutputDType,
}

impl TrtllmLowLatencyGemmTacticsQuery {
    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_positive("m", self.m)?;
        check_positive("n", self.n)?;
        check_positive("k", self.k)?;
        Ok(())
    }
}

pub fn tgv_gemm(params: &TgvGemmParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: all FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { tgv_gemm_with_runtime(runtime, params) }
}

pub fn tgv_gemm_tactic_num(dtype: DType) -> Result<i64, FlashInferError> {
    let runtime = FlashInferRuntime::global()?;
    let kernel_uri = tgv_kernel_uri(dtype);
    let mut result = any_none();
    // SAFETY: runtime and symbol loading contracts are validated on initialization/resolution.
    unsafe {
        runtime.call_tgv_gemm_tactic_num(
            kernel_uri,
            std::ptr::null(),
            0,
            &mut result as *mut TVMFFIAny,
        )?;
    }
    unpack_i64(&result, "tgv_gemm_tactic_num")
}

pub fn trtllm_gemm_tactics(query: &TrtllmGemmTacticsQuery) -> Result<Vec<i64>, FlashInferError> {
    query.validate()?;
    let runtime = FlashInferRuntime::global()?;

    let args: [TVMFFIAny; 6] = [
        any_i64(query.m),
        any_i64(query.n),
        any_i64(query.k),
        any_i64(query.input_dtype.as_i64()),
        any_i64(query.output_dtype.as_i64()),
        any_bool(query.use_8x4_sf_layout),
    ];
    let mut result = any_none();
    // SAFETY: argument packing follows TVMFFIAny ABI.
    unsafe {
        runtime.call_trtllm_gemm_tactics(
            TRTLLM_GEMM_KERNEL_URI,
            args.as_ptr(),
            args.len() as i32,
            &mut result as *mut _,
        )?;
        runtime.decode_i64_array(&result)
    }
}

pub fn trtllm_low_latency_gemm_tactics(
    query: &TrtllmLowLatencyGemmTacticsQuery,
) -> Result<Vec<i64>, FlashInferError> {
    query.validate()?;
    let runtime = FlashInferRuntime::global()?;

    let args: [TVMFFIAny; 5] = [
        any_i64(query.m),
        any_i64(query.n),
        any_i64(query.k),
        any_i64(query.input_dtype.as_i64()),
        any_i64(query.output_dtype.as_i64()),
    ];
    let mut result = any_none();
    // SAFETY: argument packing follows TVMFFIAny ABI.
    unsafe {
        runtime.call_trtllm_low_latency_gemm_tactics(
            TRTLLM_LOW_LATENCY_GEMM_KERNEL_URI,
            args.as_ptr(),
            args.len() as i32,
            &mut result as *mut _,
        )?;
        runtime.decode_i64_array(&result)
    }
}

pub fn trtllm_low_latency_workspace_size_in_bytes(
    m: i64,
    n: i64,
    k: i64,
    tactic: i64,
) -> Result<i64, FlashInferError> {
    check_positive("m", m)?;
    check_positive("n", n)?;
    check_positive("k", k)?;
    if tactic < 0 {
        return Err(FlashInferError::invalid_argument(
            "tactic must be greater than or equal to 0",
        ));
    }

    let runtime = FlashInferRuntime::global()?;
    let args: [TVMFFIAny; 4] = [any_i64(m), any_i64(n), any_i64(k), any_i64(tactic)];
    let mut result = any_none();
    // SAFETY: argument packing follows TVMFFIAny ABI.
    unsafe {
        runtime.call_trtllm_low_latency_workspace_size(
            TRTLLM_LOW_LATENCY_GEMM_KERNEL_URI,
            args.as_ptr(),
            args.len() as i32,
            &mut result as *mut _,
        )?;
    }
    unpack_i64(&result, "get_workspace_size_in_bytes")
}

unsafe fn tgv_gemm_with_runtime(
    runtime: &FlashInferRuntime,
    params: &TgvGemmParams,
) -> Result<(), FlashInferError> {
    // Upstream calls `tgv_gemm(b.t(), a.t(), ...)`; we pass transposed views without copies.
    let mut mat1_shape = [params.b.cols, params.b.rows];
    let mut mat1_strides = [params.b.stride_col, params.b.stride_row];
    let mat1_tensor = tensor_2d(
        params.b.ptr,
        params.b.dtype,
        params.b.device_id,
        &mut mat1_shape,
        &mut mat1_strides,
    );

    let mut mat2_shape = [params.a.cols, params.a.rows];
    let mut mat2_strides = [params.a.stride_col, params.a.stride_row];
    let mat2_tensor = tensor_2d(
        params.a.ptr,
        params.a.dtype,
        params.a.device_id,
        &mut mat2_shape,
        &mut mat2_strides,
    );

    let mut out_shape = [params.out.rows, params.out.cols];
    let mut out_strides = [params.out.stride_row, params.out.stride_col];
    let out_tensor = tensor_2d(
        params.out.ptr,
        params.out.dtype,
        params.out.device_id,
        &mut out_shape,
        &mut out_strides,
    );

    let mut bias_shape = [0_i64; 1];
    let mut bias_strides = [0_i64; 1];
    let bias_tensor = params.bias.map(|bias| {
        tensor_1d(
            bias.ptr,
            bias.dtype,
            bias.device_id,
            &mut bias_shape,
            &mut bias_strides,
            bias.len,
            bias.stride,
        )
    });

    let args: [TVMFFIAny; 6] = [
        any_dltensor_ptr(&mat1_tensor),
        any_dltensor_ptr(&mat2_tensor),
        optional_dltensor_any(bias_tensor.as_ref()),
        any_i64(params.tactic),
        any_dltensor_ptr(&out_tensor),
        any_bool(params.enable_pdl),
    ];
    let mut result = any_none();

    let kernel_uri = tgv_kernel_uri(params.a.dtype);

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.a.device_id, params.stream)? };
    let mut restore_guard = StreamRestoreGuard::new(runtime, params.a.device_id, previous_stream);

    // SAFETY: symbol signature follows TVMFFISafeCallType.
    let call_result = unsafe {
        runtime.call_tgv_gemm(
            kernel_uri,
            args.as_ptr(),
            args.len() as i32,
            &mut result as *mut _,
        )
    };
    let restore_result = restore_guard.restore_now();

    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn validate_tgv_gemm(params: &TgvGemmParams) -> Result<(), FlashInferError> {
    check_non_null(params.a.ptr, "a")?;
    check_non_null(params.b.ptr, "b")?;
    check_non_null(params.out.ptr, "out")?;
    if let Some(bias) = params.bias {
        check_non_null(bias.ptr, "bias")?;
    }

    check_positive("a.rows", params.a.rows)?;
    check_positive("a.cols", params.a.cols)?;
    check_positive("b.rows", params.b.rows)?;
    check_positive("b.cols", params.b.cols)?;
    check_positive("out.rows", params.out.rows)?;
    check_positive("out.cols", params.out.cols)?;

    if params.tactic < -1 {
        return Err(FlashInferError::invalid_argument(
            "tactic must be -1 or a non-negative value",
        ));
    }

    if params.a.dtype != params.b.dtype || params.a.dtype != params.out.dtype {
        return Err(FlashInferError::invalid_argument(
            "dtype mismatch across a/b/out",
        ));
    }

    if params.a.cols != params.b.rows {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: a.cols ({}) must equal b.rows ({})",
            params.a.cols, params.b.rows
        )));
    }

    if params.out.rows != params.a.rows || params.out.cols != params.b.cols {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: out ({}, {}) must equal ({}, {})",
            params.out.rows, params.out.cols, params.a.rows, params.b.cols
        )));
    }

    if params.a.device_id != params.b.device_id || params.a.device_id != params.out.device_id {
        return Err(FlashInferError::invalid_argument(
            "device mismatch across a/b/out",
        ));
    }

    if params.a.stride_col != 1 {
        return Err(FlashInferError::invalid_argument(
            "a must be row-major with last-dimension stride 1",
        ));
    }
    if params.b.stride_row != 1 {
        return Err(FlashInferError::invalid_argument(
            "b must be column-major with leading-dimension stride 1",
        ));
    }
    if params.out.stride_col != 1 {
        return Err(FlashInferError::invalid_argument(
            "out must be row-major with last-dimension stride 1",
        ));
    }

    if let Some(bias) = params.bias {
        check_positive("bias.len", bias.len)?;
        if bias.len != params.b.cols {
            return Err(FlashInferError::invalid_argument(format!(
                "shape mismatch: bias.len ({}) must equal output columns ({})",
                bias.len, params.b.cols
            )));
        }
        if bias.dtype != params.a.dtype {
            return Err(FlashInferError::invalid_argument(
                "dtype mismatch: bias must match input dtype",
            ));
        }
        if bias.device_id != params.a.device_id {
            return Err(FlashInferError::invalid_argument(
                "device mismatch: bias must be on same device as a/b/out",
            ));
        }
        if bias.stride != 1 {
            return Err(FlashInferError::invalid_argument(
                "bias must be contiguous with stride 1",
            ));
        }
    }

    Ok(())
}

fn check_non_null(ptr: *const c_void, name: &str) -> Result<(), FlashInferError> {
    if ptr.is_null() {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} pointer is null"
        )));
    }
    Ok(())
}

fn check_positive(name: &str, value: i64) -> Result<(), FlashInferError> {
    if value <= 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be positive"
        )));
    }
    Ok(())
}

fn tgv_kernel_uri(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => TGV_GEMM_FP16_KERNEL_URI,
        DType::BF16 => TGV_GEMM_BF16_KERNEL_URI,
    }
}

fn dl_dtype_from_dtype(dtype: DType) -> DLDataType {
    match dtype {
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

fn tensor_2d(
    ptr: *const c_void,
    dtype: DType,
    device_id: i32,
    shape: &mut [i64; 2],
    strides: &mut [i64; 2],
) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim: 2,
        dtype: dl_dtype_from_dtype(dtype),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn tensor_1d(
    ptr: *const c_void,
    dtype: DType,
    device_id: i32,
    shape: &mut [i64; 1],
    strides: &mut [i64; 1],
    len: i64,
    stride: i64,
) -> DLTensor {
    shape[0] = len;
    strides[0] = stride;
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim: 1,
        dtype: dl_dtype_from_dtype(dtype),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn optional_dltensor_any(tensor: Option<&DLTensor>) -> TVMFFIAny {
    match tensor {
        Some(tensor) => any_dltensor_ptr(tensor),
        None => any_none(),
    }
}

fn unpack_i64(any: &TVMFFIAny, context: &str) -> Result<i64, FlashInferError> {
    if any.type_index != KTVM_FFI_INT {
        return Err(FlashInferError::invalid_argument(format!(
            "{context} returned non-int type index {}",
            any.type_index
        )));
    }
    // SAFETY: type_index is checked to be KTVM_FFI_INT.
    Ok(unsafe { any.value.v_int64 })
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
#[derive(Debug, Clone, Copy)]
pub struct TgvGemmCudarcOptions {
    pub tactic: i64,
    pub enable_pdl: bool,
}

#[cfg(feature = "cudarc")]
impl Default for TgvGemmCudarcOptions {
    fn default() -> Self {
        Self {
            tactic: -1,
            enable_pdl: false,
        }
    }
}

#[cfg(feature = "cudarc")]
pub fn tgv_gemm_cudarc<T, A, B, O>(
    stream: &cudarc::driver::CudaStream,
    a: &A,
    b: &B,
    out: &mut O,
    m: usize,
    k: usize,
    n: usize,
    dtype: DType,
    options: TgvGemmCudarcOptions,
) -> Result<(), FlashInferError>
where
    A: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    B: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    tgv_gemm_cudarc_with_optional_bias::<T, A, B, O, A>(
        stream, a, b, None, out, m, k, n, dtype, options,
    )
}

#[cfg(feature = "cudarc")]
pub fn tgv_gemm_cudarc_with_bias<T, A, B, Bias, O>(
    stream: &cudarc::driver::CudaStream,
    a: &A,
    b: &B,
    bias: &Bias,
    out: &mut O,
    m: usize,
    k: usize,
    n: usize,
    dtype: DType,
    options: TgvGemmCudarcOptions,
) -> Result<(), FlashInferError>
where
    A: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    B: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    Bias: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    tgv_gemm_cudarc_with_optional_bias(stream, a, b, Some(bias), out, m, k, n, dtype, options)
}

#[cfg(feature = "cudarc")]
fn tgv_gemm_cudarc_with_optional_bias<T, A, B, O, Bias>(
    stream: &cudarc::driver::CudaStream,
    a: &A,
    b: &B,
    bias: Option<&Bias>,
    out: &mut O,
    m: usize,
    k: usize,
    n: usize,
    dtype: DType,
    options: TgvGemmCudarcOptions,
) -> Result<(), FlashInferError>
where
    A: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    B: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    Bias: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
{
    let a_expected = m
        .checked_mul(k)
        .ok_or_else(|| FlashInferError::invalid_argument("m * k overflow"))?;
    let b_expected = k
        .checked_mul(n)
        .ok_or_else(|| FlashInferError::invalid_argument("k * n overflow"))?;
    let out_expected = m
        .checked_mul(n)
        .ok_or_else(|| FlashInferError::invalid_argument("m * n overflow"))?;

    if a.len() != a_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "a length ({}) must equal m * k ({a_expected})",
            a.len()
        )));
    }
    if b.len() != b_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "b length ({}) must equal k * n ({b_expected})",
            b.len()
        )));
    }
    if out.len() != out_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal m * n ({out_expected})",
            out.len()
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != n {
            return Err(FlashInferError::invalid_argument(format!(
                "bias length ({}) must equal n ({n})",
                bias.len()
            )));
        }
    }

    let (a_ptr, _a_sync) = a.device_ptr(stream);
    let (b_ptr, _b_sync) = b.device_ptr(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);
    let (bias_ptr, _bias_sync) = if let Some(bias) = bias {
        let (ptr, sync) = bias.device_ptr(stream);
        (Some(ptr), Some(sync))
    } else {
        (None, None)
    };

    let m_i64 =
        i64::try_from(m).map_err(|_| FlashInferError::invalid_argument("m does not fit in i64"))?;
    let k_i64 =
        i64::try_from(k).map_err(|_| FlashInferError::invalid_argument("k does not fit in i64"))?;
    let n_i64 =
        i64::try_from(n).map_err(|_| FlashInferError::invalid_argument("n does not fit in i64"))?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let params = TgvGemmParams::new(
        GemmTensor2DDesc {
            ptr: a_ptr as usize as *const c_void,
            rows: m_i64,
            cols: k_i64,
            stride_row: k_i64,
            stride_col: 1,
            dtype,
            device_id,
        },
        GemmTensor2DDesc {
            ptr: b_ptr as usize as *const c_void,
            rows: k_i64,
            cols: n_i64,
            stride_row: 1,
            stride_col: k_i64,
            dtype,
            device_id,
        },
        GemmTensor2DDesc {
            ptr: out_ptr as usize as *const c_void,
            rows: m_i64,
            cols: n_i64,
            stride_row: n_i64,
            stride_col: 1,
            dtype,
            device_id,
        },
        stream.cu_stream().cast(),
    )
    .with_tactic(options.tactic)
    .with_enable_pdl(options.enable_pdl);

    let params = if let Some(bias_ptr) = bias_ptr {
        params.with_bias(GemmTensor1DDesc {
            ptr: bias_ptr as usize as *const c_void,
            len: n_i64,
            stride: 1,
            dtype,
            device_id,
        })
    } else {
        params
    };

    let _keep_bias_sync_alive = _bias_sync;
    tgv_gemm(&params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dangling_ptr() -> *const c_void {
        std::ptr::NonNull::<u8>::dangling()
            .as_ptr()
            .cast::<c_void>()
    }

    fn valid_tgv_params(dtype: DType) -> TgvGemmParams {
        let ptr = dangling_ptr();
        TgvGemmParams::new(
            GemmTensor2DDesc {
                ptr,
                rows: 8,
                cols: 16,
                stride_row: 16,
                stride_col: 1,
                dtype,
                device_id: 0,
            },
            GemmTensor2DDesc {
                ptr,
                rows: 16,
                cols: 32,
                stride_row: 1,
                stride_col: 16,
                dtype,
                device_id: 0,
            },
            GemmTensor2DDesc {
                ptr,
                rows: 8,
                cols: 32,
                stride_row: 32,
                stride_col: 1,
                dtype,
                device_id: 0,
            },
            std::ptr::null_mut(),
        )
    }

    #[test]
    fn tgv_validate_accepts_base_case_fp16() {
        let params = valid_tgv_params(DType::F16);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn tgv_validate_accepts_base_case_bf16() {
        let params = valid_tgv_params(DType::BF16);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn tgv_validate_rejects_null_pointer() {
        let mut params = valid_tgv_params(DType::F16);
        params.a.ptr = std::ptr::null();
        let err = params.validate().expect_err("expected null ptr rejection");
        assert!(err.to_string().contains("a pointer is null"));
    }

    #[test]
    fn tgv_validate_rejects_shape_mismatch() {
        let mut params = valid_tgv_params(DType::F16);
        params.b.rows = 15;
        let err = params.validate().expect_err("expected shape mismatch");
        assert!(err.to_string().contains("a.cols"));
    }

    #[test]
    fn tgv_validate_rejects_dtype_mismatch() {
        let mut params = valid_tgv_params(DType::F16);
        params.out.dtype = DType::BF16;
        let err = params.validate().expect_err("expected dtype mismatch");
        assert!(err.to_string().contains("dtype mismatch"));
    }

    #[test]
    fn tgv_validate_rejects_stride_layout_mismatch() {
        let mut params = valid_tgv_params(DType::F16);
        params.b.stride_row = 2;
        let err = params.validate().expect_err("expected stride mismatch");
        assert!(err.to_string().contains("column-major"));
    }

    #[test]
    fn tgv_validate_rejects_device_mismatch() {
        let mut params = valid_tgv_params(DType::F16);
        params.out.device_id = 1;
        let err = params.validate().expect_err("expected device mismatch");
        assert!(err.to_string().contains("device mismatch"));
    }

    #[test]
    fn tgv_validate_rejects_bias_shape_mismatch() {
        let mut params = valid_tgv_params(DType::F16);
        params.bias = Some(GemmTensor1DDesc {
            ptr: dangling_ptr(),
            len: 31,
            stride: 1,
            dtype: DType::F16,
            device_id: 0,
        });
        let err = params.validate().expect_err("expected bias shape mismatch");
        assert!(err.to_string().contains("bias.len"));
    }

    #[test]
    fn trtllm_gemm_query_validate_rejects_non_positive_dims() {
        let query = TrtllmGemmTacticsQuery {
            m: 0,
            n: 64,
            k: 128,
            input_dtype: TrtllmInputDType::E4m3,
            output_dtype: TrtllmOutputDType::Bf16,
            use_8x4_sf_layout: false,
        };
        let err = query.validate().expect_err("expected invalid query");
        assert!(err.to_string().contains("m must be positive"));
    }

    #[test]
    fn trtllm_low_latency_query_validate_rejects_non_positive_dims() {
        let query = TrtllmLowLatencyGemmTacticsQuery {
            m: 16,
            n: 0,
            k: 128,
            input_dtype: TrtllmInputDType::E4m3,
            output_dtype: TrtllmOutputDType::Bf16,
        };
        let err = query.validate().expect_err("expected invalid query");
        assert!(err.to_string().contains("n must be positive"));
    }

    #[test]
    fn workspace_size_validate_rejects_invalid_tactic() {
        let err = trtllm_low_latency_workspace_size_in_bytes(16, 32, 64, -1)
            .expect_err("expected invalid tactic");
        assert!(err.to_string().contains("tactic"));
    }
}
