//! TensorRT-LLM standalone all-reduce bindings from FlashInfer's prebuilt communication module.

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_INT, KTVM_FFI_NONE, TVMFFIAny,
    any_bool, any_dltensor_ptr, any_dtype, any_f64, any_i64, any_none, any_object_handle,
};
use crate::runtime::FlashInferRuntime;

const ALLREDUCE_ONLY_PATTERN: i64 = 0;
const ALLREDUCE_RESIDUAL_RMSNORM_PATTERN: i64 = 1;
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

/// Description of a contiguous rank-1 BF16 CUDA tensor used by all-reduce fusion.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmAllReduceBf16VectorDesc {
    /// Device pointer to the first BF16 element.
    pub ptr: *mut c_void,
    /// Number of elements, shape `[len]`.
    pub len: i64,
    /// Element stride; fused RMSNorm requires `1`.
    pub stride: i64,
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

/// Parameters for fused BF16 all-reduce, residual addition, and RMSNorm.
///
/// For every token, the operation computes
/// `residual_output = residual_input + sum_across_ranks(allreduce_input)` and
/// `norm_output = rms_norm(residual_output, rms_gamma, rms_epsilon)`. All four
/// rank-2 tensors are contiguous CUDA BF16 `[tokens, hidden_size]`; `rms_gamma`
/// is contiguous CUDA BF16 `[hidden_size]`. Inputs, outputs, and gamma must not
/// overlap. The call only enqueues work on `stream`, so all tensors and every
/// allocation referenced by `workspace` must remain live until it completes.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmAllReduceResidualRmsNormBf16Params {
    /// Rank-local BF16 contribution `[tokens, hidden_size]` to reduce.
    pub allreduce_input: TrtllmAllReduceBf16TensorDesc,
    /// Replicated BF16 residual input `[tokens, hidden_size]`.
    pub residual_input: TrtllmAllReduceBf16TensorDesc,
    /// BF16 pre-normalization result `[tokens, hidden_size]`, fully overwritten.
    pub residual_output: TrtllmAllReduceBf16TensorDesc,
    /// BF16 normalized result `[tokens, hidden_size]`, fully overwritten.
    pub norm_output: TrtllmAllReduceBf16TensorDesc,
    /// BF16 RMSNorm affine weight `[hidden_size]`.
    pub rms_gamma: TrtllmAllReduceBf16VectorDesc,
    /// Positive finite RMSNorm epsilon, representable as a positive finite F32.
    pub rms_epsilon: f64,
    /// CUDA I64 workspace pointer table and its allocation capacity.
    pub workspace: TrtllmAllReduceWorkspaceDesc,
    /// Rank of this process in `workspace.world_size`.
    pub rank: i64,
    /// Whether to enable CUDA programmatic dependent launch.
    pub launch_with_pdl: bool,
    /// `true` selects one-shot Lamport; `false` selects two-shot synchronization.
    pub use_oneshot: bool,
    /// Whether the one-shot kernel triggers PDL completion after its final peer write.
    pub trigger_completion_at_end: bool,
    /// Whether BF16 values are accumulated in FP32 during all-reduce.
    pub fp32_acc: bool,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    pub stream: *mut c_void,
}

impl TrtllmAllReduceResidualRmsNormBf16Params {
    /// Creates fused parameters with PDL and FP32 all-reduce accumulation disabled.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        allreduce_input: TrtllmAllReduceBf16TensorDesc,
        residual_input: TrtllmAllReduceBf16TensorDesc,
        residual_output: TrtllmAllReduceBf16TensorDesc,
        norm_output: TrtllmAllReduceBf16TensorDesc,
        rms_gamma: TrtllmAllReduceBf16VectorDesc,
        rms_epsilon: f64,
        workspace: TrtllmAllReduceWorkspaceDesc,
        rank: i64,
        use_oneshot: bool,
        stream: *mut c_void,
    ) -> Self {
        Self {
            allreduce_input,
            residual_input,
            residual_output,
            norm_output,
            rms_gamma,
            rms_epsilon,
            workspace,
            rank,
            launch_with_pdl: false,
            use_oneshot,
            trigger_completion_at_end: false,
            fp32_acc: false,
            stream,
        }
    }

    /// Validates tensor shapes, layouts, aliasing, devices, scalars, and workspace capacity.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_allreduce_residual_rmsnorm(self)
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

/// Enqueues fused BF16 all-reduce, residual addition, and RMSNorm.
pub fn trtllm_allreduce_residual_rmsnorm_bf16(
    params: &TrtllmAllReduceResidualRmsNormBf16Params,
) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: validation establishes every tensor, workspace, scalar, and stream ABI contract.
    unsafe { allreduce_residual_rmsnorm_with_runtime(runtime, params) }
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
    // SAFETY: argument order and values match FlashInfer v0.6.12's exported typed function.
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

unsafe fn allreduce_residual_rmsnorm_with_runtime(
    runtime: &FlashInferRuntime,
    params: &TrtllmAllReduceResidualRmsNormBf16Params,
) -> Result<(), FlashInferError> {
    let mut allreduce_input_shape = [
        params.allreduce_input.tokens,
        params.allreduce_input.hidden_size,
    ];
    let mut allreduce_input_strides = [
        params.allreduce_input.stride_tokens,
        params.allreduce_input.stride_hidden,
    ];
    let allreduce_input = bf16_tensor_2d(
        params.allreduce_input,
        &mut allreduce_input_shape,
        &mut allreduce_input_strides,
    );

    let mut residual_input_shape = [
        params.residual_input.tokens,
        params.residual_input.hidden_size,
    ];
    let mut residual_input_strides = [
        params.residual_input.stride_tokens,
        params.residual_input.stride_hidden,
    ];
    let residual_input = bf16_tensor_2d(
        params.residual_input,
        &mut residual_input_shape,
        &mut residual_input_strides,
    );

    let mut residual_output_shape = [
        params.residual_output.tokens,
        params.residual_output.hidden_size,
    ];
    let mut residual_output_strides = [
        params.residual_output.stride_tokens,
        params.residual_output.stride_hidden,
    ];
    let residual_output = bf16_tensor_2d(
        params.residual_output,
        &mut residual_output_shape,
        &mut residual_output_strides,
    );

    let mut norm_output_shape = [params.norm_output.tokens, params.norm_output.hidden_size];
    let mut norm_output_strides = [
        params.norm_output.stride_tokens,
        params.norm_output.stride_hidden,
    ];
    let norm_output = bf16_tensor_2d(
        params.norm_output,
        &mut norm_output_shape,
        &mut norm_output_strides,
    );

    let mut gamma_shape = [params.rms_gamma.len];
    let mut gamma_strides = [params.rms_gamma.stride];
    let rms_gamma = DLTensor {
        data: params.rms_gamma.ptr,
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.rms_gamma.device_id,
        },
        ndim: 1,
        dtype: bf16_dtype(),
        shape: gamma_shape.as_mut_ptr(),
        strides: gamma_strides.as_mut_ptr(),
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

    let args = pack_allreduce_residual_rmsnorm_args(
        &allreduce_input,
        &workspace,
        &residual_input,
        &residual_output,
        &norm_output,
        &rms_gamma,
        params,
    );
    let mut result = any_none();

    // SAFETY: TVM-FFI stream context is set for the validated CUDA device.
    let previous_stream =
        unsafe { runtime.set_stream(params.allreduce_input.device_id, params.stream)? };
    let mut stream_guard =
        StreamRestoreGuard::new(runtime, params.allreduce_input.device_id, previous_stream);
    // SAFETY: argument order and values match FlashInfer v0.6.12's exported typed function.
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

fn bf16_tensor_2d(
    desc: TrtllmAllReduceBf16TensorDesc,
    shape: &mut [i64; 2],
    strides: &mut [i64; 2],
) -> DLTensor {
    DLTensor {
        data: desc.ptr,
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: desc.device_id,
        },
        ndim: 2,
        dtype: bf16_dtype(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
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
    // SAFETY: argument order and values match FlashInfer v0.6.12's exported typed function.
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
        any_i64(ALLREDUCE_ONLY_PATTERN),
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

#[allow(clippy::too_many_arguments)]
fn pack_allreduce_residual_rmsnorm_args(
    allreduce_input: &DLTensor,
    workspace: &DLTensor,
    residual_input: &DLTensor,
    residual_output: &DLTensor,
    norm_output: &DLTensor,
    rms_gamma: &DLTensor,
    params: &TrtllmAllReduceResidualRmsNormBf16Params,
) -> [TVMFFIAny; 21] {
    [
        any_dltensor_ptr(allreduce_input),
        any_i64(params.workspace.world_size),
        any_i64(params.rank),
        any_i64(params.allreduce_input.tokens),
        any_i64(params.allreduce_input.hidden_size),
        any_dltensor_ptr(workspace),
        any_bool(params.launch_with_pdl),
        any_bool(params.use_oneshot),
        any_bool(params.trigger_completion_at_end),
        any_bool(params.fp32_acc),
        any_i64(ALLREDUCE_RESIDUAL_RMSNORM_PATTERN),
        any_none(),
        any_dltensor_ptr(residual_input),
        any_dltensor_ptr(residual_output),
        any_dltensor_ptr(norm_output),
        any_none(),
        any_none(),
        any_dltensor_ptr(rms_gamma),
        any_f64(params.rms_epsilon),
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
    if buffer.tokens > workspace.max_tokens {
        return Err(FlashInferError::invalid_argument(format!(
            "all-reduce tokens ({}) exceed workspace capacity max_tokens ({})",
            buffer.tokens, workspace.max_tokens
        )));
    }
    if buffer.hidden_size != workspace.hidden_size {
        return Err(FlashInferError::invalid_argument(format!(
            "all-reduce hidden_size ({}) does not match workspace hidden_size ({})",
            buffer.hidden_size, workspace.hidden_size
        )));
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

fn validate_allreduce_residual_rmsnorm(
    params: &TrtllmAllReduceResidualRmsNormBf16Params,
) -> Result<(), FlashInferError> {
    validate_allreduce(&TrtllmAllReduceBf16Params {
        buffer: params.allreduce_input,
        workspace: params.workspace,
        rank: params.rank,
        launch_with_pdl: params.launch_with_pdl,
        use_oneshot: params.use_oneshot,
        trigger_completion_at_end: params.trigger_completion_at_end,
        fp32_acc: params.fp32_acc,
        stream: params.stream,
    })?;

    validate_matching_fused_tensor(
        params.residual_input,
        params.allreduce_input,
        "residual input",
    )?;
    validate_matching_fused_tensor(
        params.residual_output,
        params.allreduce_input,
        "residual output",
    )?;
    validate_matching_fused_tensor(params.norm_output, params.allreduce_input, "norm output")?;

    let gamma = params.rms_gamma;
    if gamma.ptr.is_null() {
        return Err(FlashInferError::invalid_argument(
            "RMSNorm gamma pointer is null",
        ));
    }
    if (gamma.ptr as usize) % 16 != 0 {
        return Err(FlashInferError::invalid_argument(
            "RMSNorm gamma pointer must be 16-byte aligned",
        ));
    }
    if gamma.len != params.allreduce_input.hidden_size || gamma.stride != 1 {
        return Err(FlashInferError::invalid_argument(format!(
            "RMSNorm gamma must be contiguous [{}], got len {} stride {}",
            params.allreduce_input.hidden_size, gamma.len, gamma.stride
        )));
    }
    if gamma.device_id != params.allreduce_input.device_id {
        return Err(FlashInferError::invalid_argument(
            "RMSNorm gamma and fused all-reduce tensors must be on the same CUDA device",
        ));
    }

    let epsilon_f32 = params.rms_epsilon as f32;
    if !params.rms_epsilon.is_finite()
        || params.rms_epsilon <= 0.0
        || !epsilon_f32.is_finite()
        || epsilon_f32 <= 0.0
    {
        return Err(FlashInferError::invalid_argument(
            "RMSNorm epsilon must be positive, finite, and representable as positive finite F32",
        ));
    }

    validate_fused_buffers_do_not_overlap(params)
}

fn validate_matching_fused_tensor(
    tensor: TrtllmAllReduceBf16TensorDesc,
    reference: TrtllmAllReduceBf16TensorDesc,
    name: &str,
) -> Result<(), FlashInferError> {
    if tensor.ptr.is_null() {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} pointer is null"
        )));
    }
    if (tensor.ptr as usize) % 16 != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} pointer must be 16-byte aligned"
        )));
    }
    if tensor.tokens != reference.tokens || tensor.hidden_size != reference.hidden_size {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} shape must match [{}, {}], got [{}, {}]",
            reference.tokens, reference.hidden_size, tensor.tokens, tensor.hidden_size
        )));
    }
    if tensor.stride_tokens != tensor.hidden_size || tensor.stride_hidden != 1 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be contiguous with strides [hidden_size, 1]"
        )));
    }
    if tensor.device_id != reference.device_id {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} and all-reduce input must be on the same CUDA device"
        )));
    }
    Ok(())
}

fn validate_fused_buffers_do_not_overlap(
    params: &TrtllmAllReduceResidualRmsNormBf16Params,
) -> Result<(), FlashInferError> {
    let tensor_bytes = params
        .allreduce_input
        .tokens
        .checked_mul(params.allreduce_input.hidden_size)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| FlashInferError::invalid_argument("fused tensor byte size overflow"))?;
    let gamma_bytes = params
        .rms_gamma
        .len
        .checked_mul(2)
        .ok_or_else(|| FlashInferError::invalid_argument("RMSNorm gamma byte size overflow"))?;
    let buffers = [
        (
            "all-reduce input",
            params.allreduce_input.ptr as usize,
            tensor_bytes,
        ),
        (
            "residual input",
            params.residual_input.ptr as usize,
            tensor_bytes,
        ),
        (
            "residual output",
            params.residual_output.ptr as usize,
            tensor_bytes,
        ),
        ("norm output", params.norm_output.ptr as usize, tensor_bytes),
        ("RMSNorm gamma", params.rms_gamma.ptr as usize, gamma_bytes),
    ];

    for (index, (left_name, left_start, left_bytes)) in buffers.iter().enumerate() {
        let left_bytes = usize::try_from(*left_bytes)
            .map_err(|_| FlashInferError::invalid_argument("fused tensor byte size overflow"))?;
        let left_end = left_start
            .checked_add(left_bytes)
            .ok_or_else(|| FlashInferError::invalid_argument("fused tensor address overflow"))?;
        for (right_name, right_start, right_bytes) in &buffers[index + 1..] {
            let right_bytes = usize::try_from(*right_bytes).map_err(|_| {
                FlashInferError::invalid_argument("fused tensor byte size overflow")
            })?;
            let right_end = right_start.checked_add(right_bytes).ok_or_else(|| {
                FlashInferError::invalid_argument("fused tensor address overflow")
            })?;
            if *left_start < right_end && *right_start < left_end {
                return Err(FlashInferError::invalid_argument(format!(
                    "{left_name} and {right_name} must not overlap"
                )));
            }
        }
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

/// `cudarc` launch options for BF16 TensorRT-LLM all-reduce operations.
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

/// `cudarc` wrapper for fused BF16 all-reduce, residual addition, and RMSNorm.
///
/// `allreduce_input`, `residual_input`, `residual_output`, and `norm_output`
/// contain exactly `tokens * hidden_size` BF16 bit patterns in `u16` elements.
/// `rms_gamma` contains `hidden_size` BF16 elements. The two output buffers are
/// fully overwritten. All buffers must be distinct, contiguous allocations on
/// `stream`'s device. The launch is asynchronous and never changes the selected
/// one-shot/two-shot algorithm.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn trtllm_allreduce_residual_rmsnorm_bf16_cudarc<I, R, RO, NO, G, W>(
    stream: &cudarc::driver::CudaStream,
    allreduce_input: &I,
    residual_input: &R,
    residual_output: &mut RO,
    norm_output: &mut NO,
    rms_gamma: &G,
    rms_epsilon: f64,
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
    I: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtr<u16>,
    R: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtr<u16>,
    RO: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtrMut<u16>,
    NO: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtrMut<u16>,
    G: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtr<u16>,
    W: cudarc::driver::DeviceSlice<i64> + cudarc::driver::DevicePtr<i64>,
{
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| FlashInferError::invalid_argument(error.to_string()))?;

    let tensor_elements = tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| FlashInferError::invalid_argument("fused tensor element count overflow"))?;
    check_len("allreduce_input", allreduce_input.len(), tensor_elements)?;
    check_len("residual_input", residual_input.len(), tensor_elements)?;
    check_len("residual_output", residual_output.len(), tensor_elements)?;
    check_len("norm_output", norm_output.len(), tensor_elements)?;
    check_len("rms_gamma", rms_gamma.len(), hidden_size)?;
    let expected_workspace_len = world_size
        .checked_mul(3)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| FlashInferError::invalid_argument("workspace length overflow"))?;
    check_len(
        "workspace_ptrs",
        workspace_ptrs.len(),
        expected_workspace_len,
    )?;

    let (allreduce_input_ptr, _allreduce_input_sync) = allreduce_input.device_ptr(stream);
    let (residual_input_ptr, _residual_input_sync) = residual_input.device_ptr(stream);
    let (residual_output_ptr, _residual_output_sync) = residual_output.device_ptr_mut(stream);
    let (norm_output_ptr, _norm_output_sync) = norm_output.device_ptr_mut(stream);
    let (rms_gamma_ptr, _rms_gamma_sync) = rms_gamma.device_ptr(stream);
    let (workspace_ptr, _workspace_sync) = workspace_ptrs.device_ptr(stream);
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("CUDA device id does not fit in i32"))?;
    let tokens = to_i64(tokens, "tokens")?;
    let hidden_size = to_i64(hidden_size, "hidden_size")?;
    let tensor_desc = |ptr| TrtllmAllReduceBf16TensorDesc {
        ptr: ptr as usize as *mut c_void,
        tokens,
        hidden_size,
        stride_tokens: hidden_size,
        stride_hidden: 1,
        device_id,
    };
    let params = TrtllmAllReduceResidualRmsNormBf16Params {
        allreduce_input: tensor_desc(allreduce_input_ptr),
        residual_input: tensor_desc(residual_input_ptr),
        residual_output: tensor_desc(residual_output_ptr),
        norm_output: tensor_desc(norm_output_ptr),
        rms_gamma: TrtllmAllReduceBf16VectorDesc {
            ptr: rms_gamma_ptr as usize as *mut c_void,
            len: hidden_size,
            stride: 1,
            device_id,
        },
        rms_epsilon,
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
    trtllm_allreduce_residual_rmsnorm_bf16(&params)
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
    use crate::ffi::{
        KTVM_FFI_BOOL, KTVM_FFI_DATA_TYPE, KTVM_FFI_DL_TENSOR_PTR, KTVM_FFI_FLOAT, KTVM_FFI_INT,
    };

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

    fn valid_fused_params() -> TrtllmAllReduceResidualRmsNormBf16Params {
        let tensor = |address| TrtllmAllReduceBf16TensorDesc {
            ptr: address as *mut c_void,
            tokens: 4,
            hidden_size: 6144,
            stride_tokens: 6144,
            stride_hidden: 1,
            device_id: 0,
        };
        TrtllmAllReduceResidualRmsNormBf16Params {
            allreduce_input: tensor(0x0100_0000),
            residual_input: tensor(0x0200_0000),
            residual_output: tensor(0x0300_0000),
            norm_output: tensor(0x0400_0000),
            rms_gamma: TrtllmAllReduceBf16VectorDesc {
                ptr: 0x0500_0000_usize as *mut c_void,
                len: 6144,
                stride: 1,
                device_id: 0,
            },
            rms_epsilon: 1e-5,
            workspace: TrtllmAllReduceWorkspaceDesc {
                ptr: 0x0600_0000_usize as *const c_void,
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
            assert_eq!(args[10].value.v_int64, ALLREDUCE_ONLY_PATTERN);
            assert_eq!(args[0].value.v_ptr, args[11].value.v_ptr);
        }
    }

    #[test]
    fn validates_fused_residual_rmsnorm_contract() {
        valid_fused_params()
            .validate()
            .expect("valid fused all-reduce params");
    }

    #[test]
    fn fused_validation_rejects_shape_device_epsilon_and_aliasing_errors() {
        let mut params = valid_fused_params();
        params.residual_output.tokens = 3;
        assert!(
            params
                .validate()
                .expect_err("shape mismatch must fail")
                .to_string()
                .contains("shape must match")
        );

        let mut params = valid_fused_params();
        params.rms_gamma.device_id = 1;
        assert!(
            params
                .validate()
                .expect_err("device mismatch must fail")
                .to_string()
                .contains("same CUDA device")
        );

        let mut params = valid_fused_params();
        params.rms_epsilon = f64::NAN;
        assert!(
            params
                .validate()
                .expect_err("NaN epsilon must fail")
                .to_string()
                .contains("epsilon")
        );

        let mut params = valid_fused_params();
        params.norm_output.ptr = params.residual_output.ptr;
        assert!(
            params
                .validate()
                .expect_err("output aliasing must fail")
                .to_string()
                .contains("must not overlap")
        );
    }

    #[test]
    fn packs_fused_residual_rmsnorm_abi() {
        let params = valid_fused_params();
        let mut shapes = [[4_i64, 6144_i64]; 4];
        let mut strides = [[6144_i64, 1_i64]; 4];
        let allreduce_input =
            bf16_tensor_2d(params.allreduce_input, &mut shapes[0], &mut strides[0]);
        let residual_input = bf16_tensor_2d(params.residual_input, &mut shapes[1], &mut strides[1]);
        let residual_output =
            bf16_tensor_2d(params.residual_output, &mut shapes[2], &mut strides[2]);
        let norm_output = bf16_tensor_2d(params.norm_output, &mut shapes[3], &mut strides[3]);
        let mut gamma_shape = [params.rms_gamma.len];
        let mut gamma_strides = [params.rms_gamma.stride];
        let gamma = DLTensor {
            data: params.rms_gamma.ptr,
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: 0,
            },
            ndim: 1,
            dtype: bf16_dtype(),
            shape: gamma_shape.as_mut_ptr(),
            strides: gamma_strides.as_mut_ptr(),
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
        let args = pack_allreduce_residual_rmsnorm_args(
            &allreduce_input,
            &workspace,
            &residual_input,
            &residual_output,
            &norm_output,
            &gamma,
            &params,
        );

        assert_eq!(args.len(), 21);
        assert_eq!(args[0].type_index, KTVM_FFI_DL_TENSOR_PTR);
        assert_eq!(args[5].type_index, KTVM_FFI_DL_TENSOR_PTR);
        assert_eq!(args[10].type_index, KTVM_FFI_INT);
        assert_eq!(args[11].type_index, KTVM_FFI_NONE);
        assert_eq!(args[12].type_index, KTVM_FFI_DL_TENSOR_PTR);
        assert_eq!(args[13].type_index, KTVM_FFI_DL_TENSOR_PTR);
        assert_eq!(args[14].type_index, KTVM_FFI_DL_TENSOR_PTR);
        assert_eq!(args[15].type_index, KTVM_FFI_NONE);
        assert_eq!(args[16].type_index, KTVM_FFI_NONE);
        assert_eq!(args[17].type_index, KTVM_FFI_DL_TENSOR_PTR);
        assert_eq!(args[18].type_index, KTVM_FFI_FLOAT);
        assert_eq!(args[19].type_index, KTVM_FFI_NONE);
        assert_eq!(args[20].type_index, KTVM_FFI_NONE);
        // SAFETY: fields match the value constructors used by the fused ABI packer.
        unsafe {
            assert_eq!(args[10].value.v_int64, ALLREDUCE_RESIDUAL_RMSNORM_PATTERN);
            assert_eq!(args[18].value.v_float64, params.rms_epsilon);
            assert_eq!(
                args[12].value.v_ptr,
                (&residual_input as *const DLTensor).cast_mut().cast()
            );
            assert_eq!(
                args[13].value.v_ptr,
                (&residual_output as *const DLTensor).cast_mut().cast()
            );
            assert_eq!(
                args[14].value.v_ptr,
                (&norm_output as *const DLTensor).cast_mut().cast()
            );
            assert_eq!(
                args[17].value.v_ptr,
                (&gamma as *const DLTensor).cast_mut().cast()
            );
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
