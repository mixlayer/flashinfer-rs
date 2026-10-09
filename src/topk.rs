//! Bindings for the fixed `topk/topk.so` module in the pinned JIT-cache wheel.
//!
//! The tensor and argument contracts follow FlashInfer v0.6.12's
//! `csrc/flashinfer_topk_binding.cu`, `csrc/topk.cu`, and `flashinfer/topk.py`.

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_FLOAT, KDL_INT, KDL_UINT, TVMFFIAny,
    any_bool, any_dltensor_ptr, any_i64, any_none,
};
use crate::runtime::FlashInferRuntime;

#[cfg(feature = "cudarc")]
use cudarc::driver::{DevicePtr, DevicePtrMut};

/// Workspace size used by the upstream deterministic radix top-k wrapper.
pub const TOPK_ROW_STATES_BYTES: usize = 1024 * 1024;

/// Element type accepted by FlashInfer's radix top-k kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopKDType {
    /// IEEE half precision.
    F16,
    /// Brain floating point with 16 bits.
    BF16,
    /// IEEE single precision.
    F32,
}

impl TopKDType {
    fn as_dl_dtype(self) -> DLDataType {
        match self {
            Self::F16 => DLDataType {
                code: KDL_FLOAT,
                bits: 16,
                lanes: 1,
            },
            Self::BF16 => DLDataType {
                code: KDL_BFLOAT,
                bits: 16,
                lanes: 1,
            },
            Self::F32 => DLDataType {
                code: KDL_FLOAT,
                bits: 32,
                lanes: 1,
            },
        }
    }
}

/// Index preference when equal values straddle the top-k boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(i64)]
pub enum TopKTieBreak {
    /// Preserve FlashInfer's legacy boundary selection.
    #[default]
    None = 0,
    /// Prefer the smaller vocabulary index.
    Small = 1,
    /// Prefer the larger vocabulary index.
    Large = 2,
}

/// Caller-owned CUDA tensor with shape `[rows, cols]`.
#[derive(Debug, Clone, Copy)]
pub struct TopKTensor2DDesc {
    /// CUDA device pointer to the first element.
    pub ptr: *const c_void,
    /// Number of rows.
    pub rows: i64,
    /// Number of columns.
    pub cols: i64,
    /// Stride between rows, in elements.
    pub stride_row: i64,
    /// Stride between adjacent columns, in elements.
    pub stride_col: i64,
    /// Tensor element type.
    pub dtype: TopKDType,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// Caller-owned contiguous I32 CUDA tensor with shape `[rows, cols]`.
#[derive(Debug, Clone, Copy)]
pub struct TopKTensor2DI32Desc {
    /// CUDA device pointer to the first element.
    pub ptr: *const c_void,
    /// Number of rows.
    pub rows: i64,
    /// Number of columns.
    pub cols: i64,
    /// Stride between rows, in elements.
    pub stride_row: i64,
    /// Stride between adjacent columns, in elements.
    pub stride_col: i64,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// Caller-owned contiguous U8 CUDA workspace for radix row state.
#[derive(Debug, Clone, Copy)]
pub struct TopKWorkspaceDesc {
    /// CUDA device pointer to zero-initialized storage.
    pub ptr: *const c_void,
    /// Workspace size in bytes; at least [`TOPK_ROW_STATES_BYTES`].
    pub len_bytes: i64,
    /// Element stride, which must be one.
    pub stride: i64,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// Launch options for FlashInfer radix top-k.
#[derive(Debug, Clone, Copy, Default)]
pub struct TopKOptions {
    /// Sort each output row by descending value in the CUDA kernel.
    pub sorted_output: bool,
    /// Select FlashInfer's repeatable multi-CTA collection path.
    pub deterministic: bool,
    /// Select which index wins when equal values cross the top-k boundary.
    pub tie_break: TopKTieBreak,
    /// Force the graph-safe filtered implementation with scalar vectorization.
    pub dsa_graph_safe: bool,
}

/// Parameters for radix top-k over contiguous CUDA rows.
#[derive(Debug, Clone, Copy)]
pub struct TopKParams {
    /// Input values, contiguous `[rows, vocab_size]`, in F16, BF16, or F32.
    pub input: TopKTensor2DDesc,
    /// Output token indices, contiguous I32 `[rows, top_k]`.
    pub output_indices: TopKTensor2DI32Desc,
    /// Output values, contiguous `[rows, top_k]`, with the input dtype.
    pub output_values: TopKTensor2DDesc,
    /// Zero-initialized U8 CUDA row-state workspace of at least 1 MiB.
    pub row_states: TopKWorkspaceDesc,
    /// Number of values selected from each input row.
    pub top_k: i64,
    /// Determinism, sorting, tie-break, and graph-safety controls.
    pub options: TopKOptions,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    pub stream: *mut c_void,
}

impl TopKParams {
    /// Creates radix top-k launch parameters.
    pub fn new(
        input: TopKTensor2DDesc,
        output_indices: TopKTensor2DI32Desc,
        output_values: TopKTensor2DDesc,
        row_states: TopKWorkspaceDesc,
        top_k: i64,
        stream: *mut c_void,
    ) -> Self {
        Self {
            input,
            output_indices,
            output_values,
            row_states,
            top_k,
            options: TopKOptions::default(),
            stream,
        }
    }

    /// Applies launch options.
    pub fn with_options(mut self, options: TopKOptions) -> Self {
        self.options = options;
        self
    }

    /// Validates shape, dtype, layout, device, and option constraints.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_params(self)
    }
}

/// Selects the largest `top_k` values and their indices from every input row.
///
/// The launch is asynchronous on `params.stream`. The row-state workspace must
/// remain live until the stream has completed the kernel.
pub fn top_k(params: &TopKParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;

    let mut input_shape = [params.input.rows, params.input.cols];
    let mut input_strides = [params.input.stride_row, params.input.stride_col];
    let input = tensor_2d(
        params.input.ptr,
        params.input.device_id,
        params.input.dtype.as_dl_dtype(),
        &mut input_shape,
        &mut input_strides,
    );

    let mut indices_shape = [params.output_indices.rows, params.output_indices.cols];
    let mut indices_strides = [
        params.output_indices.stride_row,
        params.output_indices.stride_col,
    ];
    let output_indices = tensor_2d(
        params.output_indices.ptr,
        params.output_indices.device_id,
        dl_i32(),
        &mut indices_shape,
        &mut indices_strides,
    );

    let mut values_shape = [params.output_values.rows, params.output_values.cols];
    let mut values_strides = [
        params.output_values.stride_row,
        params.output_values.stride_col,
    ];
    let output_values = tensor_2d(
        params.output_values.ptr,
        params.output_values.device_id,
        params.output_values.dtype.as_dl_dtype(),
        &mut values_shape,
        &mut values_strides,
    );

    let mut row_states_shape = [params.row_states.len_bytes];
    let mut row_states_strides = [params.row_states.stride];
    let row_states = tensor_1d_u8(
        params.row_states.ptr,
        params.row_states.device_id,
        &mut row_states_shape,
        &mut row_states_strides,
    );

    let args: [TVMFFIAny; 9] = [
        any_dltensor_ptr(&input),
        any_dltensor_ptr(&output_indices),
        any_dltensor_ptr(&output_values),
        any_dltensor_ptr(&row_states),
        any_i64(params.top_k),
        any_bool(params.options.sorted_output),
        any_bool(params.options.deterministic),
        any_i64(params.options.tie_break as i64),
        any_bool(params.options.dsa_graph_safe),
    ];
    let mut result = any_none();

    // SAFETY: stream context API is resolved from the pinned TVM-FFI runtime.
    let previous_stream = unsafe { runtime.set_stream(params.input.device_id, params.stream)? };
    let mut restore_guard =
        StreamRestoreGuard::new(runtime, params.input.device_id, previous_stream);
    // SAFETY: argument order exactly matches FlashInfer v0.6.12's typed export.
    let call_result =
        unsafe { runtime.call_radix_topk(args.as_ptr(), args.len() as i32, &mut result as *mut _) };
    let restore_result = restore_guard.restore_now();

    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

/// Cudarc convenience wrapper for [`top_k`].
///
/// All buffers are flat contiguous allocations. `input` has `rows * cols`
/// elements; both outputs have `rows * top_k` elements; `row_states` contains
/// at least [`TOPK_ROW_STATES_BYTES`] zero-initialized bytes.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_k_cudarc<T, I, OI, OV, W>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    output_indices: &mut OI,
    output_values: &mut OV,
    row_states: &mut W,
    rows: usize,
    cols: usize,
    top_k_value: usize,
    dtype: TopKDType,
    options: TopKOptions,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + DevicePtr<T>,
    OI: cudarc::driver::DeviceSlice<i32> + DevicePtrMut<i32>,
    OV: cudarc::driver::DeviceSlice<T> + DevicePtrMut<T>,
    W: cudarc::driver::DeviceSlice<u8> + DevicePtrMut<u8>,
{
    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| FlashInferError::invalid_argument("rows * cols overflow"))?;
    let output_len = rows
        .checked_mul(top_k_value)
        .ok_or_else(|| FlashInferError::invalid_argument("rows * top_k overflow"))?;
    if input.len() != input_len {
        return Err(FlashInferError::invalid_argument(format!(
            "input length ({}) must equal rows * cols ({input_len})",
            input.len()
        )));
    }
    if output_indices.len() != output_len {
        return Err(FlashInferError::invalid_argument(format!(
            "output_indices length ({}) must equal rows * top_k ({output_len})",
            output_indices.len()
        )));
    }
    if output_values.len() != output_len {
        return Err(FlashInferError::invalid_argument(format!(
            "output_values length ({}) must equal rows * top_k ({output_len})",
            output_values.len()
        )));
    }
    if row_states.len() < TOPK_ROW_STATES_BYTES {
        return Err(FlashInferError::invalid_argument(format!(
            "row_states length ({}) must be at least {TOPK_ROW_STATES_BYTES}",
            row_states.len()
        )));
    }

    let rows = to_i64(rows, "rows")?;
    let cols = to_i64(cols, "cols")?;
    let top_k_value = to_i64(top_k_value, "top_k")?;
    let row_states_len = to_i64(row_states.len(), "row_states.len")?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;
    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (indices_ptr, _indices_sync) = output_indices.device_ptr_mut(stream);
    let (values_ptr, _values_sync) = output_values.device_ptr_mut(stream);
    let (row_states_ptr, _row_states_sync) = row_states.device_ptr_mut(stream);

    let params = TopKParams::new(
        TopKTensor2DDesc {
            ptr: input_ptr as usize as *const c_void,
            rows,
            cols,
            stride_row: cols,
            stride_col: 1,
            dtype,
            device_id,
        },
        TopKTensor2DI32Desc {
            ptr: indices_ptr as usize as *const c_void,
            rows,
            cols: top_k_value,
            stride_row: top_k_value,
            stride_col: 1,
            device_id,
        },
        TopKTensor2DDesc {
            ptr: values_ptr as usize as *const c_void,
            rows,
            cols: top_k_value,
            stride_row: top_k_value,
            stride_col: 1,
            dtype,
            device_id,
        },
        TopKWorkspaceDesc {
            ptr: row_states_ptr as usize as *const c_void,
            len_bytes: row_states_len,
            stride: 1,
            device_id,
        },
        top_k_value,
        stream.cu_stream().cast(),
    )
    .with_options(options);
    top_k(&params)
}

fn validate_params(params: &TopKParams) -> Result<(), FlashInferError> {
    validate_value_matrix("input", params.input)?;
    validate_i32_matrix("output_indices", params.output_indices)?;
    validate_value_matrix("output_values", params.output_values)?;
    validate_workspace(params.row_states)?;

    if params.top_k <= 0 || params.top_k > params.input.cols {
        return Err(FlashInferError::invalid_argument(format!(
            "top_k must be in 1..={}",
            params.input.cols
        )));
    }
    if params.input.rows > u32::MAX as i64
        || params.input.cols > u32::MAX as i64
        || params.top_k > u32::MAX as i64
    {
        return Err(FlashInferError::invalid_argument(
            "rows, cols, and top_k must fit in u32",
        ));
    }
    if params.output_indices.rows != params.input.rows || params.output_indices.cols != params.top_k
    {
        return Err(FlashInferError::invalid_argument(
            "output_indices shape must be [input.rows, top_k]",
        ));
    }
    if params.output_values.rows != params.input.rows || params.output_values.cols != params.top_k {
        return Err(FlashInferError::invalid_argument(
            "output_values shape must be [input.rows, top_k]",
        ));
    }
    if params.output_values.dtype != params.input.dtype {
        return Err(FlashInferError::invalid_argument(
            "output_values dtype must match input dtype",
        ));
    }
    for (name, device_id) in [
        ("output_indices", params.output_indices.device_id),
        ("output_values", params.output_values.device_id),
        ("row_states", params.row_states.device_id),
    ] {
        if device_id != params.input.device_id {
            return Err(FlashInferError::invalid_argument(format!(
                "{name} must be on the input CUDA device"
            )));
        }
    }
    if params.options.tie_break != TopKTieBreak::None && !params.options.deterministic {
        return Err(FlashInferError::invalid_argument(
            "tie_break requires deterministic=true",
        ));
    }
    Ok(())
}

fn validate_value_matrix(name: &str, tensor: TopKTensor2DDesc) -> Result<(), FlashInferError> {
    validate_matrix(
        name,
        tensor.ptr,
        tensor.rows,
        tensor.cols,
        tensor.stride_row,
        tensor.stride_col,
        tensor.device_id,
    )
}

fn validate_i32_matrix(name: &str, tensor: TopKTensor2DI32Desc) -> Result<(), FlashInferError> {
    validate_matrix(
        name,
        tensor.ptr,
        tensor.rows,
        tensor.cols,
        tensor.stride_row,
        tensor.stride_col,
        tensor.device_id,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_matrix(
    name: &str,
    ptr: *const c_void,
    rows: i64,
    cols: i64,
    stride_row: i64,
    stride_col: i64,
    device_id: i32,
) -> Result<(), FlashInferError> {
    if ptr.is_null() {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} pointer must be non-null"
        )));
    }
    if rows <= 0 || cols <= 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} dimensions must be positive"
        )));
    }
    // A singleton row never advances by `stride_row`, and DLPack/Candle may
    // preserve the parent row stride for such an otherwise contiguous view.
    if stride_col != 1 || (rows > 1 && stride_row != cols) {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be contiguous row-major"
        )));
    }
    if device_id < 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} device_id must be non-negative"
        )));
    }
    Ok(())
}

fn validate_workspace(workspace: TopKWorkspaceDesc) -> Result<(), FlashInferError> {
    if workspace.ptr.is_null() {
        return Err(FlashInferError::invalid_argument(
            "row_states pointer must be non-null",
        ));
    }
    if workspace.len_bytes < TOPK_ROW_STATES_BYTES as i64 {
        return Err(FlashInferError::invalid_argument(format!(
            "row_states must contain at least {TOPK_ROW_STATES_BYTES} bytes"
        )));
    }
    if workspace.stride != 1 {
        return Err(FlashInferError::invalid_argument(
            "row_states must be contiguous",
        ));
    }
    if workspace.device_id < 0 {
        return Err(FlashInferError::invalid_argument(
            "row_states device_id must be non-negative",
        ));
    }
    Ok(())
}

fn tensor_2d(
    ptr: *const c_void,
    device_id: i32,
    dtype: DLDataType,
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
        dtype,
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn tensor_1d_u8(
    ptr: *const c_void,
    device_id: i32,
    shape: &mut [i64; 1],
    strides: &mut [i64; 1],
) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim: 1,
        dtype: DLDataType {
            code: KDL_UINT,
            bits: 8,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn dl_i32() -> DLDataType {
    DLDataType {
        code: KDL_INT,
        bits: 32,
        lanes: 1,
    }
}

fn to_i64(value: usize, name: &str) -> Result<i64, FlashInferError> {
    i64::try_from(value)
        .map_err(|_| FlashInferError::invalid_argument(format!("{name} does not fit in i64")))
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
        // SAFETY: `previous_stream` came from TVMFFIEnvSetStream for this device.
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
        // SAFETY: best-effort restoration of the prior stream.
        let _ = unsafe {
            self.runtime
                .restore_stream(self.device_id, self.previous_stream)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_null() -> *const c_void {
        std::ptr::dangling::<u8>().cast()
    }

    fn valid_params() -> TopKParams {
        TopKParams::new(
            TopKTensor2DDesc {
                ptr: non_null(),
                rows: 2,
                cols: 32,
                stride_row: 32,
                stride_col: 1,
                dtype: TopKDType::F32,
                device_id: 0,
            },
            TopKTensor2DI32Desc {
                ptr: non_null(),
                rows: 2,
                cols: 4,
                stride_row: 4,
                stride_col: 1,
                device_id: 0,
            },
            TopKTensor2DDesc {
                ptr: non_null(),
                rows: 2,
                cols: 4,
                stride_row: 4,
                stride_col: 1,
                dtype: TopKDType::F32,
                device_id: 0,
            },
            TopKWorkspaceDesc {
                ptr: non_null(),
                len_bytes: TOPK_ROW_STATES_BYTES as i64,
                stride: 1,
                device_id: 0,
            },
            4,
            std::ptr::null_mut(),
        )
    }

    #[test]
    fn validates_complete_topk_contract() {
        valid_params().validate().expect("valid top-k params");
    }

    #[test]
    fn rejects_output_shape_mismatch() {
        let mut params = valid_params();
        params.output_indices.cols = 3;
        assert!(params.validate().is_err());
    }

    #[test]
    fn rejects_non_contiguous_input() {
        let mut params = valid_params();
        params.input.stride_row = 33;
        assert!(params.validate().is_err());
    }

    #[test]
    fn accepts_parent_stride_for_singleton_input_row() {
        let mut params = valid_params();
        params.input.rows = 1;
        params.input.stride_row = params.input.cols + 32;
        params.output_indices.rows = 1;
        params.output_values.rows = 1;
        params.validate().expect("singleton row stride is unused");
    }

    #[test]
    fn rejects_small_row_state_workspace() {
        let mut params = valid_params();
        params.row_states.len_bytes = TOPK_ROW_STATES_BYTES as i64 - 1;
        assert!(params.validate().is_err());
    }

    #[test]
    fn explicit_tie_break_requires_deterministic_mode() {
        let mut params = valid_params();
        params.options.tie_break = TopKTieBreak::Small;
        assert!(params.validate().is_err());
        params.options.deterministic = true;
        params.validate().expect("deterministic tie-break");
    }
}
