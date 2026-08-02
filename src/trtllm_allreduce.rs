//! TensorRT-LLM standalone all-reduce bindings from FlashInfer's prebuilt communication module.

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_INT, KTVM_FFI_NONE, TVMFFIAny,
    any_bool, any_dltensor_ptr, any_dtype, any_i64, any_none, any_object_handle,
};
use crate::runtime::FlashInferRuntime;

const ALLREDUCE_PATTERN: i64 = 0;
const BF16_ELEMENTS_PER_ACCESS: i64 = 8;
const LAMPORT_BF16_ELEMENTS_PER_ACCESS: i64 = 8;
const MAX_COMM_SIZE_BYTES: i64 = i32::MAX as i64 & !((1_i64 << 21) - 1);

/// Description of a contiguous rank-2 BF16 CUDA tensor reduced in place.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmAllReduceBf16TensorDesc {
    /// Mutable device pointer to the first BF16 element.
    pub ptr: *mut c_void,
    /// Number of tokens, shape `[tokens, hidden_size]`.
    pub tokens: i64,
    /// Hidden dimension, shape `[tokens, hidden_size]`; must be divisible by 8 BF16 elements.
    pub hidden_size: i64,
    /// Row stride in elements; this binding requires `hidden_size`.
    pub stride_tokens: i64,
    /// Hidden-dimension stride in elements; this binding requires `1`.
    pub stride_hidden: i64,
    /// CUDA device ordinal containing the tensor and workspace.
    pub device_id: i32,
}

/// Description and allocation metadata for the TensorRT-LLM all-reduce workspace pointer table.
///
/// `ptr` addresses a contiguous CUDA I64 tensor with shape `[3 * world_size + 1]`. Its entries
/// are peer communication pointers, peer barrier pointers, peer Lamport pointers, and the local
/// five-I32 metadata allocation pointer, in that order. The pointed-to IPC allocations remain
/// owned by the caller and must outlive every asynchronous launch using this descriptor.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmAllReduceWorkspaceDesc {
    /// Device pointer to the first I64 workspace-table entry.
    pub ptr: *const c_void,
    /// Number of I64 entries; must equal `3 * world_size + 1`.
    pub len: i64,
    /// Tensor-parallel world size used when the IPC workspace was created.
    pub world_size: i64,
    /// Maximum token count used to size the peer communication and Lamport allocations.
    pub max_tokens: i64,
    /// Hidden size used to size the peer communication and Lamport allocations.
    pub hidden_size: i64,
    /// Whether the workspace Lamport allocation was sized for FP32 elements.
    ///
    /// Standalone BF16 all-reduce requires this to be `false`.
    pub use_fp32_lamport: bool,
    /// CUDA device ordinal containing the workspace table.
    pub device_id: i32,
}

/// Parameters for a standalone in-place BF16 TensorRT-LLM all-reduce.
///
/// The operation replaces every `[tokens, hidden_size]` element in `buffer` with the sum across
/// all participating ranks. `buffer` and `workspace` must be contiguous CUDA allocations on the
/// same device. The call only enqueues work on `stream`; the caller owns synchronization and must
/// keep the tensor, pointer table, and every IPC allocation referenced by the table alive until
/// the stream completes.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmAllReduceBf16Params {
    /// BF16 input/output buffer `[tokens, hidden_size]`, mutated in place.
    pub buffer: TrtllmAllReduceBf16TensorDesc,
    /// CUDA I64 workspace pointer table and the metadata describing its allocation capacity.
    pub workspace: TrtllmAllReduceWorkspaceDesc,
    /// Rank of this process in `workspace.world_size`.
    pub rank: i64,
    /// Whether to enable CUDA programmatic dependent launch.
    pub launch_with_pdl: bool,
    /// `true` selects the one-shot Lamport algorithm; `false` selects two-shot synchronization.
    ///
    /// This value is never changed implicitly. Invalid choices return an error before launch.
    pub use_oneshot: bool,
    /// Whether the one-shot kernel triggers PDL completion after its final peer write.
    pub trigger_completion_at_end: bool,
    /// Whether BF16 values are accumulated in FP32.
    pub fp32_acc: bool,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    pub stream: *mut c_void,
}

impl TrtllmAllReduceBf16Params {
    /// Creates standalone BF16 all-reduce parameters with PDL and FP32 accumulation disabled.
    pub fn new(
        buffer: TrtllmAllReduceBf16TensorDesc,
        workspace: TrtllmAllReduceWorkspaceDesc,
        rank: i64,
        use_oneshot: bool,
        stream: *mut c_void,
    ) -> Self {
        Self {
            buffer,
            workspace,
            rank,
            launch_with_pdl: false,
            use_oneshot,
            trigger_completion_at_end: false,
            fp32_acc: false,
            stream,
        }
    }

    /// Validates tensor shape, layout, rank, workspace capacity, and algorithm constraints.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_allreduce(self)
    }
}

/// Parameters for asynchronously filling a local Lamport allocation with BF16 negative zero.
///
/// This initialization must finish on every rank before the workspace is used by all-reduce.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmLamportInitializeBf16Params {
    /// Mutable CUDA pointer to a local BF16 Lamport allocation.
    pub ptr: *mut c_void,
    /// Allocation length in BF16 elements; must be positive and divisible by 8.
    pub len: i64,
    /// CUDA device ordinal containing `ptr`.
    pub device_id: i32,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous initialization.
    pub stream: *mut c_void,
}

impl TrtllmLamportInitializeBf16Params {
    /// Creates BF16 Lamport initialization parameters.
    pub fn new(ptr: *mut c_void, len: i64, device_id: i32, stream: *mut c_void) -> Self {
        Self {
            ptr,
            len,
            device_id,
            stream,
        }
    }

    /// Validates the Lamport allocation pointer, length, alignment, and device ordinal.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_lamport_initialize(self)
    }
}

/// Enqueues a standalone in-place BF16 TensorRT-LLM all-reduce.
pub fn trtllm_allreduce_bf16_in_place(
    params: &TrtllmAllReduceBf16Params,
) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: validation establishes the tensor, workspace, scalar, and stream ABI contracts.
    unsafe { allreduce_with_runtime(runtime, params) }
}

/// Enqueues BF16 negative-zero initialization of a local Lamport workspace allocation.
pub fn trtllm_lamport_initialize_bf16(
    params: &TrtllmLamportInitializeBf16Params,
) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: validation establishes the pointer, scalar, and stream ABI contracts.
    unsafe { lamport_initialize_with_runtime(runtime, params) }
}

unsafe fn allreduce_with_runtime(
    runtime: &FlashInferRuntime,
    params: &TrtllmAllReduceBf16Params,
) -> Result<(), FlashInferError> {
    let mut buffer_shape = [params.buffer.tokens, params.buffer.hidden_size];
    let mut buffer_strides = [params.buffer.stride_tokens, params.buffer.stride_hidden];
    let buffer = DLTensor {
        data: params.buffer.ptr,
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.buffer.device_id,
        },
        ndim: 2,
        dtype: bf16_dtype(),
        shape: buffer_shape.as_mut_ptr(),
        strides: buffer_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut workspace_shape = [params.workspace.len];
    let mut workspace_strides = [1_i64];
    let workspace = DLTensor {
        data: params.workspace.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.workspace.device_id,
        },
        ndim: 1,
        dtype: i64_dtype(),
        shape: workspace_shape.as_mut_ptr(),
        strides: workspace_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let args = pack_allreduce_args(&buffer, &workspace, params);
    let mut result = any_none();

    // SAFETY: TVM-FFI stream context is set for the validated CUDA device.
    let previous_stream = unsafe { runtime.set_stream(params.buffer.device_id, params.stream)? };
    let mut stream_guard =
        StreamRestoreGuard::new(runtime, params.buffer.device_id, previous_stream);
    // SAFETY: argument order and values match FlashInfer v0.6.4's exported typed function.
    let call_result = unsafe {
        runtime.call_trtllm_allreduce_fusion(
            args.as_ptr(),
            args.len() as i32,
            &mut result as *mut _,
        )
    };
    let call_result = call_result.and_then(|()| validate_void_result(runtime, &result));
    let restore_result = stream_guard.restore_now();

    merge_call_and_restore(call_result, restore_result)
}

unsafe fn lamport_initialize_with_runtime(
    runtime: &FlashInferRuntime,
    params: &TrtllmLamportInitializeBf16Params,
) -> Result<(), FlashInferError> {
    let args = pack_lamport_initialize_args(params.ptr, params.len)?;
    let mut result = any_none();

    // SAFETY: TVM-FFI stream context is set for the validated CUDA device.
    let previous_stream = unsafe { runtime.set_stream(params.device_id, params.stream)? };
    let mut stream_guard = StreamRestoreGuard::new(runtime, params.device_id, previous_stream);
    // SAFETY: argument order and values match FlashInfer v0.6.4's exported typed function.
    let call_result = unsafe {
        runtime.call_trtllm_lamport_initialize(
            args.as_ptr(),
            args.len() as i32,
            &mut result as *mut _,
        )
    };
    let call_result = call_result.and_then(|()| validate_void_result(runtime, &result));
    let restore_result = stream_guard.restore_now();

    merge_call_and_restore(call_result, restore_result)
}

fn pack_allreduce_args(
    buffer: &DLTensor,
    workspace: &DLTensor,
    params: &TrtllmAllReduceBf16Params,
) -> [TVMFFIAny; 21] {
    [
        any_dltensor_ptr(buffer),
        any_i64(params.workspace.world_size),
        any_i64(params.rank),
        any_i64(params.buffer.tokens),
        any_i64(params.buffer.hidden_size),
        any_dltensor_ptr(workspace),
        any_bool(params.launch_with_pdl),
        any_bool(params.use_oneshot),
        any_bool(params.trigger_completion_at_end),
        any_bool(params.fp32_acc),
        any_i64(ALLREDUCE_PATTERN),
        any_dltensor_ptr(buffer),
        any_none(),
        any_none(),
        any_none(),
        any_none(),
        any_none(),
        any_none(),
        any_none(),
        any_none(),
        any_none(),
    ]
}

fn pack_lamport_initialize_args(
    ptr: *mut c_void,
    len: i64,
) -> Result<[TVMFFIAny; 3], FlashInferError> {
    let address = i64::try_from(ptr as usize).map_err(|_| {
        FlashInferError::invalid_argument("Lamport buffer address does not fit in i64")
    })?;
    Ok([any_i64(address), any_i64(len), any_dtype(bf16_dtype())])
}

fn validate_allreduce(params: &TrtllmAllReduceBf16Params) -> Result<(), FlashInferError> {
    let buffer = params.buffer;
    let workspace = params.workspace;

    if buffer.ptr.is_null() {
        return Err(FlashInferError::invalid_argument(
            "all-reduce buffer pointer is null",
        ));
    }
    if workspace.ptr.is_null() {
        return Err(FlashInferError::invalid_argument(
            "all-reduce workspace pointer is null",
        ));
    }
    if (buffer.ptr as usize) % 16 != 0 {
        return Err(FlashInferError::invalid_argument(
            "all-reduce buffer pointer must be 16-byte aligned",
        ));
    }
    if (workspace.ptr as usize) % std::mem::align_of::<i64>() != 0 {
        return Err(FlashInferError::invalid_argument(
            "all-reduce workspace pointer must be aligned for i64",
        ));
    }
    if buffer.tokens <= 0 || buffer.hidden_size <= 0 {
        return Err(FlashInferError::invalid_argument(
            "all-reduce shape must be positive",
        ));
    }
    if buffer.tokens > i32::MAX as i64 || buffer.hidden_size > i32::MAX as i64 {
        return Err(FlashInferError::invalid_argument(
            "all-reduce tokens and hidden_size must fit in i32",
        ));
    }
    if buffer.hidden_size % BF16_ELEMENTS_PER_ACCESS != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "BF16 hidden_size must be divisible by {BF16_ELEMENTS_PER_ACCESS}"
        )));
    }
    if buffer.stride_tokens != buffer.hidden_size || buffer.stride_hidden != 1 {
        return Err(FlashInferError::invalid_argument(
            "all-reduce buffer must be contiguous with strides [hidden_size, 1]",
        ));
    }
    if buffer.device_id < 0 || workspace.device_id < 0 {
        return Err(FlashInferError::invalid_argument(
            "CUDA device ordinals must be non-negative",
        ));
    }
    if buffer.device_id != workspace.device_id {
        return Err(FlashInferError::invalid_argument(
            "all-reduce buffer and workspace must be on the same CUDA device",
        ));
    }
    if !matches!(workspace.world_size, 2 | 4 | 8 | 16) {
        return Err(FlashInferError::invalid_argument(
            "TensorRT-LLM all-reduce supports world sizes 2, 4, 8, and 16",
        ));
    }
    if params.rank < 0 || params.rank >= workspace.world_size {
        return Err(FlashInferError::invalid_argument(format!(
            "rank {} is outside world size {}",
            params.rank, workspace.world_size
        )));
    }
    let expected_workspace_len = workspace
        .world_size
        .checked_mul(3)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| FlashInferError::invalid_argument("workspace length overflow"))?;
    if workspace.len != expected_workspace_len {
        return Err(FlashInferError::invalid_argument(format!(
            "workspace pointer table must contain {expected_workspace_len} i64 entries, got {}",
            workspace.len
        )));
    }
    if workspace.max_tokens <= 0 || workspace.hidden_size <= 0 {
        return Err(FlashInferError::invalid_argument(
            "workspace max_tokens and hidden_size must be positive",
        ));
    }
    if workspace.use_fp32_lamport {
        return Err(FlashInferError::invalid_argument(
            "BF16 all-reduce requires a non-FP32 Lamport workspace",
        ));
    }

    let elements = buffer
        .tokens
        .checked_mul(buffer.hidden_size)
        .ok_or_else(|| FlashInferError::invalid_argument("all-reduce element count overflow"))?;
    let workspace_elements = workspace
        .max_tokens
        .checked_mul(workspace.hidden_size)
        .ok_or_else(|| FlashInferError::invalid_argument("workspace element capacity overflow"))?;
    if elements > workspace_elements {
        return Err(FlashInferError::invalid_argument(format!(
            "all-reduce requires {elements} elements but workspace capacity is {workspace_elements}"
        )));
    }

    if params.use_oneshot {
        let lamport_bytes = elements
            .checked_mul(2)
            .and_then(|value| value.checked_mul(workspace.world_size))
            .ok_or_else(|| FlashInferError::invalid_argument("Lamport byte count overflow"))?;
        if lamport_bytes > MAX_COMM_SIZE_BYTES {
            return Err(FlashInferError::invalid_argument(format!(
                "one-shot Lamport communication requires {lamport_bytes} bytes, exceeding the {MAX_COMM_SIZE_BYTES}-byte limit"
            )));
        }
    } else if buffer.tokens <= workspace.world_size {
        return Err(FlashInferError::invalid_argument(format!(
            "two-shot all-reduce requires tokens ({}) greater than world size ({})",
            buffer.tokens, workspace.world_size
        )));
    }

    Ok(())
}

fn validate_lamport_initialize(
    params: &TrtllmLamportInitializeBf16Params,
) -> Result<(), FlashInferError> {
    if params.ptr.is_null() {
        return Err(FlashInferError::invalid_argument(
            "Lamport buffer pointer is null",
        ));
    }
    if (params.ptr as usize) % 16 != 0 {
        return Err(FlashInferError::invalid_argument(
            "Lamport buffer pointer must be 16-byte aligned",
        ));
    }
    if params.len <= 0 || params.len % LAMPORT_BF16_ELEMENTS_PER_ACCESS != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "Lamport BF16 length must be positive and divisible by {LAMPORT_BF16_ELEMENTS_PER_ACCESS}"
        )));
    }
    if params.device_id < 0 {
        return Err(FlashInferError::invalid_argument(
            "CUDA device ordinal must be non-negative",
        ));
    }
    Ok(())
}

fn validate_void_result(
    runtime: &FlashInferRuntime,
    result: &TVMFFIAny,
) -> Result<(), FlashInferError> {
    if result.type_index == KTVM_FFI_NONE {
        return Ok(());
    }
    if let Some(object) = any_object_handle(result) {
        // SAFETY: a successful TVM safe call transfers ownership of object results to the caller.
        unsafe { runtime.object_dec_ref(object) };
    }
    Err(FlashInferError::invalid_argument(
        "TensorRT-LLM communication function unexpectedly returned a value",
    ))
}

fn merge_call_and_restore(
    call_result: Result<(), FlashInferError>,
    restore_result: Result<(), FlashInferError>,
) -> Result<(), FlashInferError> {
    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

const fn bf16_dtype() -> DLDataType {
    DLDataType {
        code: KDL_BFLOAT,
        bits: 16,
        lanes: 1,
    }
}

const fn i64_dtype() -> DLDataType {
    DLDataType {
        code: KDL_INT,
        bits: 64,
        lanes: 1,
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
        // SAFETY: previous_stream was returned by TVMFFIEnvSetStream for this device.
        unsafe {
            self.runtime
                .restore_stream(self.device_id, self.previous_stream)
        }
    }
}

impl Drop for StreamRestoreGuard<'_> {
    fn drop(&mut self) {
        let _ = self.restore_now();
    }
}

/// `cudarc` launch options for standalone in-place BF16 TensorRT-LLM all-reduce.
#[cfg(feature = "cudarc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrtllmAllReduceBf16CudarcOptions {
    /// Whether to enable CUDA programmatic dependent launch.
    pub launch_with_pdl: bool,
    /// Explicitly select one-shot (`true`) or two-shot (`false`) operation.
    pub use_oneshot: bool,
    /// Whether the one-shot kernel triggers PDL completion after its final peer write.
    pub trigger_completion_at_end: bool,
    /// Whether BF16 values are accumulated in FP32.
    pub fp32_acc: bool,
}

#[cfg(feature = "cudarc")]
impl Default for TrtllmAllReduceBf16CudarcOptions {
    fn default() -> Self {
        Self {
            launch_with_pdl: false,
            use_oneshot: true,
            trigger_completion_at_end: false,
            fp32_acc: false,
        }
    }
}

/// `cudarc` convenience wrapper for standalone in-place BF16 TensorRT-LLM all-reduce.
///
/// `buffer` stores BF16 bit patterns in `u16` elements with shape `[tokens, hidden_size]`.
/// `workspace_ptrs` is a device I64 table with shape `[3 * world_size + 1]`. The caller owns the
/// IPC allocations referenced by that table. The launch is asynchronous on `stream` and never
/// changes the requested one-shot/two-shot algorithm.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn trtllm_allreduce_bf16_in_place_cudarc<B, W>(
    stream: &cudarc::driver::CudaStream,
    buffer: &mut B,
    workspace_ptrs: &W,
    tokens: usize,
    hidden_size: usize,
    world_size: usize,
    rank: usize,
    workspace_max_tokens: usize,
    workspace_hidden_size: usize,
    options: TrtllmAllReduceBf16CudarcOptions,
) -> Result<(), FlashInferError>
where
    B: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtrMut<u16>,
    W: cudarc::driver::DeviceSlice<i64> + cudarc::driver::DevicePtr<i64>,
{
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| FlashInferError::invalid_argument(error.to_string()))?;

    let expected_buffer_len = tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| FlashInferError::invalid_argument("buffer element count overflow"))?;
    check_len("buffer", buffer.len(), expected_buffer_len)?;
    let expected_workspace_len = world_size
        .checked_mul(3)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| FlashInferError::invalid_argument("workspace length overflow"))?;
    check_len(
        "workspace_ptrs",
        workspace_ptrs.len(),
        expected_workspace_len,
    )?;

    let (buffer_ptr, _buffer_sync) = buffer.device_ptr_mut(stream);
    let (workspace_ptr, _workspace_sync) = workspace_ptrs.device_ptr(stream);
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("CUDA device id does not fit in i32"))?;
    let params = TrtllmAllReduceBf16Params {
        buffer: TrtllmAllReduceBf16TensorDesc {
            ptr: buffer_ptr as usize as *mut c_void,
            tokens: to_i64(tokens, "tokens")?,
            hidden_size: to_i64(hidden_size, "hidden_size")?,
            stride_tokens: to_i64(hidden_size, "hidden_size")?,
            stride_hidden: 1,
            device_id,
        },
        workspace: TrtllmAllReduceWorkspaceDesc {
            ptr: workspace_ptr as usize as *const c_void,
            len: to_i64(workspace_ptrs.len(), "workspace_ptrs length")?,
            world_size: to_i64(world_size, "world_size")?,
            max_tokens: to_i64(workspace_max_tokens, "workspace_max_tokens")?,
            hidden_size: to_i64(workspace_hidden_size, "workspace_hidden_size")?,
            use_fp32_lamport: false,
            device_id,
        },
        rank: to_i64(rank, "rank")?,
        launch_with_pdl: options.launch_with_pdl,
        use_oneshot: options.use_oneshot,
        trigger_completion_at_end: options.trigger_completion_at_end,
        fp32_acc: options.fp32_acc,
        stream: stream.cu_stream().cast(),
    };
    trtllm_allreduce_bf16_in_place(&params)
}

/// `cudarc` convenience wrapper for asynchronous BF16 Lamport workspace initialization.
///
/// `buffer` is a local Lamport allocation represented as BF16 bit patterns in `u16` elements.
#[cfg(feature = "cudarc")]
pub fn trtllm_lamport_initialize_bf16_cudarc<B>(
    stream: &cudarc::driver::CudaStream,
    buffer: &mut B,
) -> Result<(), FlashInferError>
where
    B: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtrMut<u16>,
{
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| FlashInferError::invalid_argument(error.to_string()))?;
    let buffer_len = buffer.len();
    let (ptr, _sync) = buffer.device_ptr_mut(stream);
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("CUDA device id does not fit in i32"))?;
    trtllm_lamport_initialize_bf16(&TrtllmLamportInitializeBf16Params {
        ptr: ptr as usize as *mut c_void,
        len: to_i64(buffer_len, "Lamport buffer length")?,
        device_id,
        stream: stream.cu_stream().cast(),
    })
}

#[cfg(feature = "cudarc")]
fn check_len(name: &str, actual: usize, expected: usize) -> Result<(), FlashInferError> {
    if actual != expected {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

#[cfg(feature = "cudarc")]
fn to_i64(value: usize, name: &str) -> Result<i64, FlashInferError> {
    i64::try_from(value)
        .map_err(|_| FlashInferError::invalid_argument(format!("{name} does not fit in i64")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{KTVM_FFI_BOOL, KTVM_FFI_DATA_TYPE, KTVM_FFI_DL_TENSOR_PTR, KTVM_FFI_INT};

    fn valid_params() -> TrtllmAllReduceBf16Params {
        TrtllmAllReduceBf16Params {
            buffer: TrtllmAllReduceBf16TensorDesc {
                ptr: 0x1000_usize as *mut c_void,
                tokens: 4,
                hidden_size: 6144,
                stride_tokens: 6144,
                stride_hidden: 1,
                device_id: 0,
            },
            workspace: TrtllmAllReduceWorkspaceDesc {
                ptr: 0x2000_usize as *const c_void,
                len: 13,
                world_size: 4,
                max_tokens: 128,
                hidden_size: 6144,
                use_fp32_lamport: false,
                device_id: 0,
            },
            rank: 1,
            launch_with_pdl: false,
            use_oneshot: true,
            trigger_completion_at_end: false,
            fp32_acc: false,
            stream: std::ptr::null_mut(),
        }
    }

    #[test]
    fn validates_supported_decode_shape_and_workspace() {
        valid_params().validate().expect("valid all-reduce params");
    }

    #[test]
    fn rejects_unsupported_world_size() {
        let mut params = valid_params();
        params.workspace.world_size = 6;
        params.workspace.len = 19;
        let error = params.validate().expect_err("world size 6 must fail");
        assert!(error.to_string().contains("world sizes 2, 4, 8, and 16"));
    }

    #[test]
    fn rejects_insufficient_workspace_capacity() {
        let mut params = valid_params();
        params.buffer.tokens = 129;
        let error = params.validate().expect_err("workspace capacity must fail");
        assert!(error.to_string().contains("workspace capacity"));
    }

    #[test]
    fn rejects_two_shot_when_tokens_do_not_exceed_world_size() {
        let mut params = valid_params();
        params.use_oneshot = false;
        let error = params.validate().expect_err("two-shot shape must fail");
        assert!(error.to_string().contains("two-shot all-reduce requires"));
    }

    #[test]
    fn rejects_oversized_one_shot_without_switching_algorithms() {
        let mut params = valid_params();
        params.buffer.tokens = 50_000;
        params.workspace.max_tokens = 50_000;
        params.workspace.world_size = 16;
        params.workspace.len = 49;
        let error = params.validate().expect_err("oversized one-shot must fail");
        assert!(error.to_string().contains("one-shot Lamport communication"));
    }

    #[test]
    fn packs_standalone_in_place_allreduce_abi() {
        let params = valid_params();
        let mut buffer_shape = [params.buffer.tokens, params.buffer.hidden_size];
        let mut buffer_strides = [params.buffer.stride_tokens, params.buffer.stride_hidden];
        let buffer = DLTensor {
            data: params.buffer.ptr,
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: 0,
            },
            ndim: 2,
            dtype: bf16_dtype(),
            shape: buffer_shape.as_mut_ptr(),
            strides: buffer_strides.as_mut_ptr(),
            byte_offset: 0,
        };
        let mut workspace_shape = [params.workspace.len];
        let mut workspace_strides = [1_i64];
        let workspace = DLTensor {
            data: params.workspace.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: 0,
            },
            ndim: 1,
            dtype: i64_dtype(),
            shape: workspace_shape.as_mut_ptr(),
            strides: workspace_strides.as_mut_ptr(),
            byte_offset: 0,
        };
        let args = pack_allreduce_args(&buffer, &workspace, &params);

        assert_eq!(args.len(), 21);
        assert_eq!(args[0].type_index, KTVM_FFI_DL_TENSOR_PTR);
        assert_eq!(args[1].type_index, KTVM_FFI_INT);
        assert_eq!(args[5].type_index, KTVM_FFI_DL_TENSOR_PTR);
        assert_eq!(args[6].type_index, KTVM_FFI_BOOL);
        assert_eq!(args[11].type_index, KTVM_FFI_DL_TENSOR_PTR);
        assert_eq!(args[12].type_index, KTVM_FFI_NONE);
        // SAFETY: fields match the value constructors used by pack_allreduce_args.
        unsafe {
            assert_eq!(args[1].value.v_int64, 4);
            assert_eq!(args[2].value.v_int64, 1);
            assert_eq!(args[3].value.v_int64, 4);
            assert_eq!(args[4].value.v_int64, 6144);
            assert_eq!(args[10].value.v_int64, ALLREDUCE_PATTERN);
            assert_eq!(args[0].value.v_ptr, args[11].value.v_ptr);
        }
    }

    #[test]
    fn packs_bf16_lamport_initialize_abi() {
        let ptr = 0x4000_usize as *mut c_void;
        let args = pack_lamport_initialize_args(ptr, 1024).expect("pack Lamport args");
        assert_eq!(args[0].type_index, KTVM_FFI_INT);
        assert_eq!(args[1].type_index, KTVM_FFI_INT);
        assert_eq!(args[2].type_index, KTVM_FFI_DATA_TYPE);
        // SAFETY: fields match the value constructors used by pack_lamport_initialize_args.
        unsafe {
            assert_eq!(args[0].value.v_int64, 0x4000);
            assert_eq!(args[1].value.v_int64, 1024);
            assert_eq!(args[2].value.v_dtype, bf16_dtype());
        }
    }

    #[test]
    fn lamport_validation_rejects_partial_vector() {
        let params = TrtllmLamportInitializeBf16Params::new(
            0x4000_usize as *mut c_void,
            9,
            0,
            std::ptr::null_mut(),
        );
        let error = params.validate().expect_err("partial vector must fail");
        assert!(error.to_string().contains("divisible by 8"));
    }
}
