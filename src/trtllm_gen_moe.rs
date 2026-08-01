//! TensorRT-LLM Gen fused MoE bindings for Blackwell GPUs.

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLManagedTensorVersioned, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_FLOAT,
    KDL_FLOAT8_E4M3FN, KDL_INT, TVMFFIAny, TVMFFIObjectHandle, any_bool, any_dltensor_ptr, any_f64,
    any_i64, any_none, any_object_handle,
};
use crate::runtime::FlashInferRuntime;

const ARRAY_GLOBAL: &str = "ffi.Array";
const DEEPSEEK_V3_ROUTING_METHOD: i64 = 2;
const MAJOR_K_WEIGHT_LAYOUT: i64 = 0;
const DEEPSEEK_FP8_QUANTIZATION: i64 = 1;

/// Dtype supported by the SM100 TensorRT-LLM Gen FP8 block-scale MoE binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrtllmGenMoeDType {
    /// IEEE FP32.
    F32,
    /// Brain floating point with 16 bits.
    BF16,
    /// FP8 E4M3FN.
    F8E4M3FN,
}

impl TrtllmGenMoeDType {
    fn as_dl_dtype(self) -> DLDataType {
        match self {
            Self::F32 => DLDataType {
                code: KDL_FLOAT,
                bits: 32,
                lanes: 1,
            },
            Self::BF16 => DLDataType {
                code: KDL_BFLOAT,
                bits: 16,
                lanes: 1,
            },
            Self::F8E4M3FN => DLDataType {
                code: KDL_FLOAT8_E4M3FN,
                bits: 8,
                lanes: 1,
            },
        }
    }
}

/// Description of a rank-1 CUDA tensor used by TensorRT-LLM Gen MoE.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmGenMoeTensor1DDesc {
    /// Device pointer to the first tensor element.
    pub ptr: *const c_void,
    /// Number of elements, shape `[len]`.
    pub len: i64,
    /// Element stride; this binding requires `1`.
    pub stride: i64,
    /// Tensor element dtype.
    pub dtype: TrtllmGenMoeDType,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// Description of a rank-2 CUDA tensor used by TensorRT-LLM Gen MoE.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmGenMoeTensor2DDesc {
    /// Device pointer to the first tensor element.
    pub ptr: *const c_void,
    /// First dimension, shape `[rows, cols]`.
    pub rows: i64,
    /// Second dimension, shape `[rows, cols]`.
    pub cols: i64,
    /// Row stride in elements; this binding requires `cols`.
    pub stride_row: i64,
    /// Column stride in elements; this binding requires `1`.
    pub stride_col: i64,
    /// Tensor element dtype.
    pub dtype: TrtllmGenMoeDType,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// Description of a rank-3 CUDA tensor used by TensorRT-LLM Gen MoE.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmGenMoeTensor3DDesc {
    /// Device pointer to the first tensor element.
    pub ptr: *const c_void,
    /// First dimension, shape `[dim0, dim1, dim2]`.
    pub dim0: i64,
    /// Second dimension, shape `[dim0, dim1, dim2]`.
    pub dim1: i64,
    /// Third dimension, shape `[dim0, dim1, dim2]`.
    pub dim2: i64,
    /// First-dimension stride in elements; this binding requires `dim1 * dim2`.
    pub stride0: i64,
    /// Second-dimension stride in elements; this binding requires `dim2`.
    pub stride1: i64,
    /// Third-dimension stride in elements; this binding requires `1`.
    pub stride2: i64,
    /// Tensor element dtype.
    pub dtype: TrtllmGenMoeDType,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// Parameters for SM100 TensorRT-LLM Gen DeepSeek FP8 block-scale MoE.
///
/// The operation consumes already-quantized activations and performs DeepSeekV3 grouped routing,
/// both expert GEMMs, SwiGLU, and routed-output finalization. All tensors must be contiguous CUDA
/// tensors on the same device. Launch and the final device-to-device output copy are asynchronous
/// on `stream`.
#[derive(Debug, Clone, Copy)]
pub struct TrtllmGenFp8BlockScaleMoeSm100Params {
    /// BF16 output `[tokens, hidden]`.
    pub out: TrtllmGenMoeTensor2DDesc,
    /// F32 router logits `[tokens, global_experts]`.
    pub routing_logits: TrtllmGenMoeTensor2DDesc,
    /// Optional F32 router correction bias `[global_experts]`.
    pub routing_bias: Option<TrtllmGenMoeTensor1DDesc>,
    /// F8E4M3FN activations `[tokens, hidden]`.
    pub hidden_states: TrtllmGenMoeTensor2DDesc,
    /// F32 activation scales `[hidden / 128, tokens]`.
    pub hidden_states_scale: TrtllmGenMoeTensor2DDesc,
    /// F8E4M3FN gate/up weights `[local_experts, 2 * intermediate, hidden]` in MajorK layout.
    pub fc1_weights: TrtllmGenMoeTensor3DDesc,
    /// F32 gate/up scales `[local_experts, 2 * intermediate / 128, hidden / 128]`.
    pub fc1_scales: TrtllmGenMoeTensor3DDesc,
    /// F8E4M3FN down weights `[local_experts, hidden, intermediate]` in MajorK layout.
    pub fc2_weights: TrtllmGenMoeTensor3DDesc,
    /// F32 down scales `[local_experts, hidden / 128, intermediate / 128]`.
    pub fc2_scales: TrtllmGenMoeTensor3DDesc,
    /// Number of experts in the global model.
    pub global_experts: i64,
    /// Experts selected per token; the pinned DeepSeek routing kernel supports at most 8.
    pub top_k: i64,
    /// Number of expert groups used by grouped routing.
    pub expert_groups: i64,
    /// Number of groups retained before expert top-k; DeepSeek routing supports at most 4.
    pub topk_group: i64,
    /// First global expert represented by the local weight tensors.
    pub local_expert_offset: i64,
    /// Optional `[tile_n, config]` tactic; `None` asks FlashInfer for its default valid tactic.
    pub tactic: Option<[i64; 2]>,
    /// Multiplier applied to the selected routing weights.
    pub routed_scaling_factor: f64,
    /// Whether to enable programmatic dependent launch.
    pub enable_pdl: bool,
    /// CUDA stream (`cudaStream_t`) used for asynchronous work.
    pub stream: *mut c_void,
}

impl TrtllmGenFp8BlockScaleMoeSm100Params {
    /// Validates shapes, dtypes, layouts, routing parameters, and device placement.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_params(self)
    }
}

/// Launches SM100 TensorRT-LLM Gen DeepSeek FP8 block-scale MoE.
///
/// This function does not synchronize. The pinned FlashInfer 0.6.4 launcher returns an internally
/// allocated BF16 tensor, so the binding enqueues a device-to-device copy into `params.out` on the
/// same stream and releases the temporary allocation in stream order.
pub fn trtllm_gen_fp8_block_scale_moe_sm100(
    params: &TrtllmGenFp8BlockScaleMoeSm100Params,
) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: validation establishes the tensor and scalar ABI contracts.
    unsafe { launch_with_runtime(runtime, params) }
}

unsafe fn launch_with_runtime(
    runtime: &FlashInferRuntime,
    params: &TrtllmGenFp8BlockScaleMoeSm100Params,
) -> Result<(), FlashInferError> {
    let mut out_shape = [params.out.rows, params.out.cols];
    let mut out_strides = [params.out.stride_row, params.out.stride_col];
    let out = tensor_2d(params.out, &mut out_shape, &mut out_strides);

    let mut logits_shape = [params.routing_logits.rows, params.routing_logits.cols];
    let mut logits_strides = [
        params.routing_logits.stride_row,
        params.routing_logits.stride_col,
    ];
    let logits = tensor_2d(
        params.routing_logits,
        &mut logits_shape,
        &mut logits_strides,
    );

    let mut bias_shape = [0_i64];
    let mut bias_strides = [0_i64];
    let bias = params.routing_bias.map(|desc| {
        bias_shape[0] = desc.len;
        bias_strides[0] = desc.stride;
        tensor_1d(desc, &mut bias_shape, &mut bias_strides)
    });

    let mut hidden_shape = [params.hidden_states.rows, params.hidden_states.cols];
    let mut hidden_strides = [
        params.hidden_states.stride_row,
        params.hidden_states.stride_col,
    ];
    let hidden = tensor_2d(params.hidden_states, &mut hidden_shape, &mut hidden_strides);

    let mut hidden_scale_shape = [
        params.hidden_states_scale.rows,
        params.hidden_states_scale.cols,
    ];
    let mut hidden_scale_strides = [
        params.hidden_states_scale.stride_row,
        params.hidden_states_scale.stride_col,
    ];
    let hidden_scale = tensor_2d(
        params.hidden_states_scale,
        &mut hidden_scale_shape,
        &mut hidden_scale_strides,
    );

    let mut fc1_shape = [
        params.fc1_weights.dim0,
        params.fc1_weights.dim1,
        params.fc1_weights.dim2,
    ];
    let mut fc1_strides = [
        params.fc1_weights.stride0,
        params.fc1_weights.stride1,
        params.fc1_weights.stride2,
    ];
    let fc1 = tensor_3d(params.fc1_weights, &mut fc1_shape, &mut fc1_strides);

    let mut fc1_scale_shape = [
        params.fc1_scales.dim0,
        params.fc1_scales.dim1,
        params.fc1_scales.dim2,
    ];
    let mut fc1_scale_strides = [
        params.fc1_scales.stride0,
        params.fc1_scales.stride1,
        params.fc1_scales.stride2,
    ];
    let fc1_scale = tensor_3d(
        params.fc1_scales,
        &mut fc1_scale_shape,
        &mut fc1_scale_strides,
    );

    let mut fc2_shape = [
        params.fc2_weights.dim0,
        params.fc2_weights.dim1,
        params.fc2_weights.dim2,
    ];
    let mut fc2_strides = [
        params.fc2_weights.stride0,
        params.fc2_weights.stride1,
        params.fc2_weights.stride2,
    ];
    let fc2 = tensor_3d(params.fc2_weights, &mut fc2_shape, &mut fc2_strides);

    let mut fc2_scale_shape = [
        params.fc2_scales.dim0,
        params.fc2_scales.dim1,
        params.fc2_scales.dim2,
    ];
    let mut fc2_scale_strides = [
        params.fc2_scales.stride0,
        params.fc2_scales.stride1,
        params.fc2_scales.stride2,
    ];
    let fc2_scale = tensor_3d(
        params.fc2_scales,
        &mut fc2_scale_shape,
        &mut fc2_scale_strides,
    );

    let mut empty_i32_shape = [0_i64, 0_i64];
    let mut empty_i32_strides = [0_i64, 1_i64];
    let empty_i32 = DLTensor {
        data: std::ptr::null_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.out.device_id,
        },
        ndim: 2,
        dtype: DLDataType {
            code: KDL_INT,
            bits: 32,
            lanes: 1,
        },
        shape: empty_i32_shape.as_mut_ptr(),
        strides: empty_i32_strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let mut empty_bf16_shape = [0_i64, 0_i64];
    let mut empty_bf16_strides = [0_i64, 1_i64];
    let empty_bf16 = DLTensor {
        data: std::ptr::null_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.out.device_id,
        },
        ndim: 2,
        dtype: TrtllmGenMoeDType::BF16.as_dl_dtype(),
        shape: empty_bf16_shape.as_mut_ptr(),
        strides: empty_bf16_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let tactic = params.tactic.unwrap_or([-1, -1]);
    let tactic_array = unsafe { make_i64_array(runtime, &tactic)? };
    let mut tactic_guard = AnyObjectGuard::new(runtime, &tactic_array);
    let args: [TVMFFIAny; 25] = [
        any_dltensor_ptr(&logits),
        any_dltensor_ptr(&empty_i32),
        any_dltensor_ptr(&empty_bf16),
        bias.as_ref()
            .map(|tensor| any_dltensor_ptr(tensor))
            .unwrap_or_else(any_none),
        any_dltensor_ptr(&hidden),
        any_dltensor_ptr(&hidden_scale),
        any_dltensor_ptr(&fc1),
        any_dltensor_ptr(&fc1_scale),
        any_dltensor_ptr(&fc2),
        any_dltensor_ptr(&fc2_scale),
        any_dltensor_ptr(&out),
        any_i64(params.global_experts),
        any_i64(params.top_k),
        any_i64(params.expert_groups),
        any_i64(params.topk_group),
        any_i64(params.fc2_weights.dim2),
        any_i64(params.local_expert_offset),
        any_i64(params.fc1_weights.dim0),
        any_f64(params.routed_scaling_factor),
        any_i64(DEEPSEEK_V3_ROUTING_METHOD),
        any_bool(false),
        any_i64(MAJOR_K_WEIGHT_LAYOUT),
        any_bool(params.enable_pdl),
        tactic_array,
        any_i64(DEEPSEEK_FP8_QUANTIZATION),
    ];

    // SAFETY: stream context API contract comes from TVM-FFI.
    let previous_stream = unsafe { runtime.set_stream(params.out.device_id, params.stream)? };
    let mut stream_guard = StreamRestoreGuard::new(runtime, params.out.device_id, previous_stream);
    let mut result = any_none();
    let call_result = (|| -> Result<(), FlashInferError> {
        // SAFETY: argument order follows FlashInfer v0.6.4's exported typed function.
        unsafe {
            runtime.call_trtllm_gen_fp8_block_scale_moe_sm100(
                args.as_ptr(),
                args.len() as i32,
                &mut result as *mut _,
            )?;
        }
        let mut result_guard = AnyObjectGuard::new(runtime, &result);
        let result_handle = any_object_handle(&result).ok_or_else(|| {
            FlashInferError::invalid_argument(
                "TensorRT-LLM Gen MoE returned a non-tensor TVM result",
            )
        })?;
        // SAFETY: result_handle is an owned TVM tensor returned by the kernel safe call.
        let managed = unsafe { runtime.tensor_to_dlpack_versioned(result_handle)? };
        let mut managed_guard = ManagedTensorGuard::new(managed);
        // SAFETY: managed is non-null and owned by managed_guard.
        let returned = unsafe { &(*managed).dl_tensor };
        validate_returned_tensor(returned, params)?;
        let output_bytes = output_bytes(params)?;
        // SAFETY: both buffers are validated CUDA allocations of at least output_bytes bytes.
        unsafe {
            runtime.copy_device_to_device_async(
                params.out.ptr.cast_mut(),
                returned.data.cast_const(),
                output_bytes,
                params.stream,
            )?;
        }
        managed_guard.release_now();
        result_guard.release_now();
        Ok(())
    })();

    tactic_guard.release_now();
    let restore_result = stream_guard.restore_now();
    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn validate_params(params: &TrtllmGenFp8BlockScaleMoeSm100Params) -> Result<(), FlashInferError> {
    validate_2d("out", params.out)?;
    validate_2d("routing_logits", params.routing_logits)?;
    validate_2d("hidden_states", params.hidden_states)?;
    validate_2d("hidden_states_scale", params.hidden_states_scale)?;
    validate_3d("fc1_weights", params.fc1_weights)?;
    validate_3d("fc1_scales", params.fc1_scales)?;
    validate_3d("fc2_weights", params.fc2_weights)?;
    validate_3d("fc2_scales", params.fc2_scales)?;
    if let Some(bias) = params.routing_bias {
        validate_1d("routing_bias", bias)?;
    }

    if params.out.dtype != TrtllmGenMoeDType::BF16
        || params.routing_logits.dtype != TrtllmGenMoeDType::F32
        || params.hidden_states.dtype != TrtllmGenMoeDType::F8E4M3FN
        || params.hidden_states_scale.dtype != TrtllmGenMoeDType::F32
        || params.fc1_weights.dtype != TrtllmGenMoeDType::F8E4M3FN
        || params.fc1_scales.dtype != TrtllmGenMoeDType::F32
        || params.fc2_weights.dtype != TrtllmGenMoeDType::F8E4M3FN
        || params.fc2_scales.dtype != TrtllmGenMoeDType::F32
    {
        return Err(FlashInferError::invalid_argument(
            "SM100 FP8 block-scale MoE requires F32 logits/scales, FP8 activations/weights, and BF16 output",
        ));
    }
    if let Some(bias) = params.routing_bias
        && bias.dtype != TrtllmGenMoeDType::F32
    {
        return Err(FlashInferError::invalid_argument(
            "routing_bias must have dtype F32",
        ));
    }

    let tokens = params.hidden_states.rows;
    let hidden = params.hidden_states.cols;
    let local_experts = params.fc1_weights.dim0;
    let fc1_size = params.fc1_weights.dim1;
    let intermediate = params.fc2_weights.dim2;
    let expected_fc1_size = intermediate
        .checked_mul(2)
        .ok_or_else(|| FlashInferError::invalid_argument("2 * intermediate overflow"))?;
    if hidden % 128 != 0 || intermediate % 128 != 0 || fc1_size != expected_fc1_size {
        return Err(FlashInferError::invalid_argument(
            "hidden and intermediate dimensions must be divisible by 128 and fc1 dim1 must equal 2 * intermediate",
        ));
    }
    expect_shape_2d("out", params.out, tokens, hidden)?;
    expect_shape_2d(
        "routing_logits",
        params.routing_logits,
        tokens,
        params.global_experts,
    )?;
    expect_shape_2d(
        "hidden_states_scale",
        params.hidden_states_scale,
        hidden / 128,
        tokens,
    )?;
    expect_shape_3d(
        "fc1_weights",
        params.fc1_weights,
        local_experts,
        expected_fc1_size,
        hidden,
    )?;
    expect_shape_3d(
        "fc1_scales",
        params.fc1_scales,
        local_experts,
        expected_fc1_size / 128,
        hidden / 128,
    )?;
    expect_shape_3d(
        "fc2_weights",
        params.fc2_weights,
        local_experts,
        hidden,
        intermediate,
    )?;
    expect_shape_3d(
        "fc2_scales",
        params.fc2_scales,
        local_experts,
        hidden / 128,
        intermediate / 128,
    )?;
    if let Some(bias) = params.routing_bias
        && bias.len != params.global_experts
    {
        return Err(FlashInferError::invalid_argument(format!(
            "routing_bias shape must be [{}]",
            params.global_experts
        )));
    }

    let experts_in_selected_groups = params
        .global_experts
        .checked_div(params.expert_groups.max(1))
        .and_then(|experts_per_group| params.topk_group.checked_mul(experts_per_group));
    if params.global_experts <= 0
        || params.top_k <= 0
        || params.top_k > params.global_experts
        || params.top_k > 8
        || params.global_experts % 4 != 0
        || params.global_experts <= params.top_k
        || params.expert_groups <= 0
        || params.global_experts % params.expert_groups != 0
        || params.topk_group <= 0
        || params.topk_group > params.expert_groups
        || params.topk_group > 4
        || experts_in_selected_groups.is_none_or(|candidates| params.top_k >= candidates)
        || params.local_expert_offset < 0
        || params
            .local_expert_offset
            .checked_add(local_experts)
            .is_none_or(|end| end > params.global_experts)
    {
        return Err(FlashInferError::invalid_argument(
            "invalid DeepSeekV3 grouped-routing or local-expert configuration",
        ));
    }
    if !params.routed_scaling_factor.is_finite() || params.routed_scaling_factor <= 0.0 {
        return Err(FlashInferError::invalid_argument(
            "routed_scaling_factor must be finite and positive",
        ));
    }
    if let Some([tile_n, config]) = params.tactic
        && (tile_n < 0 || config < 0)
    {
        return Err(FlashInferError::invalid_argument(
            "explicit TensorRT-LLM Gen tactic values must be non-negative",
        ));
    }

    let device = params.out.device_id;
    for (name, actual) in [
        ("routing_logits", params.routing_logits.device_id),
        ("hidden_states", params.hidden_states.device_id),
        ("hidden_states_scale", params.hidden_states_scale.device_id),
        ("fc1_weights", params.fc1_weights.device_id),
        ("fc1_scales", params.fc1_scales.device_id),
        ("fc2_weights", params.fc2_weights.device_id),
        ("fc2_scales", params.fc2_scales.device_id),
    ] {
        if actual != device {
            return Err(FlashInferError::invalid_argument(format!(
                "device mismatch: {name} is on CUDA device {actual}, expected {device}"
            )));
        }
    }
    if let Some(bias) = params.routing_bias
        && bias.device_id != device
    {
        return Err(FlashInferError::invalid_argument(
            "device mismatch: routing_bias must be on the output device",
        ));
    }
    output_bytes(params)?;
    Ok(())
}

fn validate_returned_tensor(
    tensor: &DLTensor,
    params: &TrtllmGenFp8BlockScaleMoeSm100Params,
) -> Result<(), FlashInferError> {
    if tensor.data.is_null()
        || tensor.device.device_type != KDL_CUDA
        || tensor.device.device_id != params.out.device_id
        || tensor.ndim != 2
        || tensor.dtype != TrtllmGenMoeDType::BF16.as_dl_dtype()
        || tensor.shape.is_null()
        || tensor.byte_offset != 0
    {
        return Err(FlashInferError::invalid_argument(
            "TensorRT-LLM Gen MoE returned an invalid BF16 CUDA tensor",
        ));
    }
    // SAFETY: ndim is exactly two and the DLPack tensor owns a two-element shape array.
    let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 2) };
    if shape != [params.out.rows, params.out.cols] {
        return Err(FlashInferError::invalid_argument(format!(
            "TensorRT-LLM Gen MoE returned shape {shape:?}, expected [{}, {}]",
            params.out.rows, params.out.cols
        )));
    }
    if !tensor.strides.is_null() {
        // SAFETY: ndim is exactly two and a non-null strides pointer has two elements.
        let strides = unsafe { std::slice::from_raw_parts(tensor.strides, 2) };
        if strides != [params.out.cols, 1] {
            return Err(FlashInferError::invalid_argument(format!(
                "TensorRT-LLM Gen MoE returned non-contiguous strides {strides:?}"
            )));
        }
    }
    Ok(())
}

fn output_bytes(params: &TrtllmGenFp8BlockScaleMoeSm100Params) -> Result<usize, FlashInferError> {
    usize::try_from(params.out.rows)
        .ok()
        .and_then(|rows| {
            usize::try_from(params.out.cols)
                .ok()
                .and_then(|cols| rows.checked_mul(cols))
        })
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| FlashInferError::invalid_argument("output byte size overflow"))
}

fn validate_1d(name: &str, desc: TrtllmGenMoeTensor1DDesc) -> Result<(), FlashInferError> {
    if desc.ptr.is_null() || desc.len <= 0 || desc.stride != 1 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be a non-empty contiguous rank-1 tensor"
        )));
    }
    Ok(())
}

fn validate_2d(name: &str, desc: TrtllmGenMoeTensor2DDesc) -> Result<(), FlashInferError> {
    if desc.ptr.is_null()
        || desc.rows <= 0
        || desc.cols <= 0
        || desc.stride_col != 1
        || desc.stride_row != desc.cols
    {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be a non-empty contiguous rank-2 tensor"
        )));
    }
    Ok(())
}

fn validate_3d(name: &str, desc: TrtllmGenMoeTensor3DDesc) -> Result<(), FlashInferError> {
    let expected_stride0 = desc.dim1.checked_mul(desc.dim2).ok_or_else(|| {
        FlashInferError::invalid_argument(format!("{name} contiguous stride overflow"))
    })?;
    if desc.ptr.is_null()
        || desc.dim0 <= 0
        || desc.dim1 <= 0
        || desc.dim2 <= 0
        || desc.stride2 != 1
        || desc.stride1 != desc.dim2
        || desc.stride0 != expected_stride0
    {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be a non-empty contiguous rank-3 tensor"
        )));
    }
    Ok(())
}

fn expect_shape_2d(
    name: &str,
    desc: TrtllmGenMoeTensor2DDesc,
    rows: i64,
    cols: i64,
) -> Result<(), FlashInferError> {
    if desc.rows != rows || desc.cols != cols {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} shape must be [{rows}, {cols}], got [{}, {}]",
            desc.rows, desc.cols
        )));
    }
    Ok(())
}

fn expect_shape_3d(
    name: &str,
    desc: TrtllmGenMoeTensor3DDesc,
    dim0: i64,
    dim1: i64,
    dim2: i64,
) -> Result<(), FlashInferError> {
    if desc.dim0 != dim0 || desc.dim1 != dim1 || desc.dim2 != dim2 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} shape must be [{dim0}, {dim1}, {dim2}], got [{}, {}, {}]",
            desc.dim0, desc.dim1, desc.dim2
        )));
    }
    Ok(())
}

fn tensor_1d(
    desc: TrtllmGenMoeTensor1DDesc,
    shape: &mut [i64; 1],
    strides: &mut [i64; 1],
) -> DLTensor {
    DLTensor {
        data: desc.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: desc.device_id,
        },
        ndim: 1,
        dtype: desc.dtype.as_dl_dtype(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn tensor_2d(
    desc: TrtllmGenMoeTensor2DDesc,
    shape: &mut [i64; 2],
    strides: &mut [i64; 2],
) -> DLTensor {
    DLTensor {
        data: desc.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: desc.device_id,
        },
        ndim: 2,
        dtype: desc.dtype.as_dl_dtype(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn tensor_3d(
    desc: TrtllmGenMoeTensor3DDesc,
    shape: &mut [i64; 3],
    strides: &mut [i64; 3],
) -> DLTensor {
    DLTensor {
        data: desc.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: desc.device_id,
        },
        ndim: 3,
        dtype: desc.dtype.as_dl_dtype(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

unsafe fn make_i64_array(
    runtime: &FlashInferRuntime,
    values: &[i64],
) -> Result<TVMFFIAny, FlashInferError> {
    // SAFETY: global function lookup returns an owned TVM function handle.
    let constructor = unsafe { runtime.get_global_function(ARRAY_GLOBAL)? };
    let mut constructor_guard = RawObjectGuard::new(runtime, constructor);
    let mut args = values.iter().copied().map(any_i64).collect::<Vec<_>>();
    let mut result = any_none();
    // SAFETY: ffi.Array accepts a positional sequence of TVM integer values.
    unsafe {
        runtime.call_function(
            constructor,
            args.as_mut_ptr(),
            args.len() as i32,
            &mut result as *mut _,
        )?;
    }
    constructor_guard.release_now();
    Ok(result)
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

struct AnyObjectGuard<'a> {
    runtime: &'a FlashInferRuntime,
    object: Option<TVMFFIObjectHandle>,
}

impl<'a> AnyObjectGuard<'a> {
    fn new(runtime: &'a FlashInferRuntime, value: &TVMFFIAny) -> Self {
        Self {
            runtime,
            object: any_object_handle(value),
        }
    }

    fn release_now(&mut self) {
        if let Some(object) = self.object.take() {
            // SAFETY: object is owned by this guard and is decref'd exactly once.
            unsafe { self.runtime.object_dec_ref(object) };
        }
    }
}

impl Drop for AnyObjectGuard<'_> {
    fn drop(&mut self) {
        self.release_now();
    }
}

struct RawObjectGuard<'a> {
    runtime: &'a FlashInferRuntime,
    object: Option<TVMFFIObjectHandle>,
}

impl<'a> RawObjectGuard<'a> {
    fn new(runtime: &'a FlashInferRuntime, object: TVMFFIObjectHandle) -> Self {
        Self {
            runtime,
            object: Some(object),
        }
    }

    fn release_now(&mut self) {
        if let Some(object) = self.object.take() {
            // SAFETY: object is owned by this guard and is decref'd exactly once.
            unsafe { self.runtime.object_dec_ref(object) };
        }
    }
}

impl Drop for RawObjectGuard<'_> {
    fn drop(&mut self) {
        self.release_now();
    }
}

struct ManagedTensorGuard {
    tensor: *mut DLManagedTensorVersioned,
}

impl ManagedTensorGuard {
    fn new(tensor: *mut DLManagedTensorVersioned) -> Self {
        Self { tensor }
    }

    fn release_now(&mut self) {
        if self.tensor.is_null() {
            return;
        }
        let tensor = std::mem::replace(&mut self.tensor, std::ptr::null_mut());
        // SAFETY: this guard owns the DLPack managed tensor and invokes its deleter once.
        if let Some(deleter) = unsafe { (*tensor).deleter } {
            unsafe { deleter(tensor) };
        }
    }
}

impl Drop for ManagedTensorGuard {
    fn drop(&mut self) {
        self.release_now();
    }
}

/// Options for the `cudarc` SM100 TensorRT-LLM Gen FP8 block-scale MoE wrapper.
#[cfg(feature = "cudarc")]
#[derive(Debug, Clone, Copy)]
pub struct TrtllmGenFp8BlockScaleMoeSm100CudarcOptions {
    /// Optional `[tile_n, config]` tactic; `None` selects FlashInfer's default.
    pub tactic: Option<[i64; 2]>,
    /// Whether to enable programmatic dependent launch.
    pub enable_pdl: bool,
}

#[cfg(feature = "cudarc")]
impl Default for TrtllmGenFp8BlockScaleMoeSm100CudarcOptions {
    fn default() -> Self {
        Self {
            tactic: None,
            enable_pdl: true,
        }
    }
}

/// `cudarc` convenience wrapper for SM100 TensorRT-LLM Gen FP8 block-scale MoE.
///
/// Flat buffers use the same contiguous dimensions documented on
/// [`TrtllmGenFp8BlockScaleMoeSm100Params`]. `routing_bias` is optional F32. The output buffer
/// stores BF16 bit patterns in `u16` elements. The launch is asynchronous on `stream`.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn trtllm_gen_fp8_block_scale_moe_sm100_cudarc<R, B, H, HS, W1, S1, W2, S2, O>(
    stream: &cudarc::driver::CudaStream,
    routing_logits: &R,
    routing_bias: Option<&B>,
    hidden_states: &H,
    hidden_states_scale: &HS,
    fc1_weights: &W1,
    fc1_scales: &S1,
    fc2_weights: &W2,
    fc2_scales: &S2,
    out: &mut O,
    num_tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
    global_experts: usize,
    top_k: usize,
    expert_groups: usize,
    topk_group: usize,
    local_expert_offset: usize,
    local_experts: usize,
    routed_scaling_factor: f64,
    options: TrtllmGenFp8BlockScaleMoeSm100CudarcOptions,
) -> Result<(), FlashInferError>
where
    R: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    B: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    H: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    HS: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    W1: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    S1: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    W2: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    S2: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtrMut<u16>,
{
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| FlashInferError::invalid_argument(error.to_string()))?;
    let (major, minor) = stream
        .context()
        .compute_capability()
        .map_err(|error| FlashInferError::invalid_argument(error.to_string()))?;
    if (major, minor) != (10, 0) {
        return Err(FlashInferError::invalid_argument(format!(
            "TensorRT-LLM Gen SM100 MoE requires compute capability 10.0, got {major}.{minor}"
        )));
    }

    let fc1_size = intermediate_size
        .checked_mul(2)
        .ok_or_else(|| FlashInferError::invalid_argument("2 * intermediate_size overflow"))?;
    let expected = CudarcExpectedLengths::new(
        num_tokens,
        hidden_size,
        intermediate_size,
        global_experts,
        local_experts,
    )?;
    check_len(
        "routing_logits",
        routing_logits.len(),
        expected.routing_logits,
    )?;
    if let Some(bias) = routing_bias {
        check_len("routing_bias", bias.len(), global_experts)?;
    }
    check_len("hidden_states", hidden_states.len(), expected.hidden)?;
    check_len(
        "hidden_states_scale",
        hidden_states_scale.len(),
        expected.hidden_scales,
    )?;
    check_len("fc1_weights", fc1_weights.len(), expected.fc1)?;
    check_len("fc1_scales", fc1_scales.len(), expected.fc1_scales)?;
    check_len("fc2_weights", fc2_weights.len(), expected.fc2)?;
    check_len("fc2_scales", fc2_scales.len(), expected.fc2_scales)?;
    check_len("out", out.len(), expected.hidden)?;

    let (routing_logits_ptr, _routing_logits_sync) = routing_logits.device_ptr(stream);
    let (routing_bias_ptr, _routing_bias_sync) = if let Some(bias) = routing_bias {
        let (ptr, sync) = bias.device_ptr(stream);
        (Some(ptr), Some(sync))
    } else {
        (None, None)
    };
    let (hidden_ptr, _hidden_sync) = hidden_states.device_ptr(stream);
    let (hidden_scale_ptr, _hidden_scale_sync) = hidden_states_scale.device_ptr(stream);
    let (fc1_ptr, _fc1_sync) = fc1_weights.device_ptr(stream);
    let (fc1_scale_ptr, _fc1_scale_sync) = fc1_scales.device_ptr(stream);
    let (fc2_ptr, _fc2_sync) = fc2_weights.device_ptr(stream);
    let (fc2_scale_ptr, _fc2_scale_sync) = fc2_scales.device_ptr(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);

    let tokens = to_i64(num_tokens, "num_tokens")?;
    let hidden = to_i64(hidden_size, "hidden_size")?;
    let intermediate = to_i64(intermediate_size, "intermediate_size")?;
    let global = to_i64(global_experts, "global_experts")?;
    let local = to_i64(local_experts, "local_experts")?;
    let fc1_size = to_i64(fc1_size, "fc1_size")?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("CUDA device id does not fit in i32"))?;

    let params = TrtllmGenFp8BlockScaleMoeSm100Params {
        out: desc_2d(
            out_ptr as usize as *const c_void,
            tokens,
            hidden,
            TrtllmGenMoeDType::BF16,
            device_id,
        ),
        routing_logits: desc_2d(
            routing_logits_ptr as usize as *const c_void,
            tokens,
            global,
            TrtllmGenMoeDType::F32,
            device_id,
        ),
        routing_bias: routing_bias_ptr.map(|ptr| TrtllmGenMoeTensor1DDesc {
            ptr: ptr as usize as *const c_void,
            len: global,
            stride: 1,
            dtype: TrtllmGenMoeDType::F32,
            device_id,
        }),
        hidden_states: desc_2d(
            hidden_ptr as usize as *const c_void,
            tokens,
            hidden,
            TrtllmGenMoeDType::F8E4M3FN,
            device_id,
        ),
        hidden_states_scale: desc_2d(
            hidden_scale_ptr as usize as *const c_void,
            hidden / 128,
            tokens,
            TrtllmGenMoeDType::F32,
            device_id,
        ),
        fc1_weights: desc_3d(
            fc1_ptr as usize as *const c_void,
            local,
            fc1_size,
            hidden,
            TrtllmGenMoeDType::F8E4M3FN,
            device_id,
        )?,
        fc1_scales: desc_3d(
            fc1_scale_ptr as usize as *const c_void,
            local,
            fc1_size / 128,
            hidden / 128,
            TrtllmGenMoeDType::F32,
            device_id,
        )?,
        fc2_weights: desc_3d(
            fc2_ptr as usize as *const c_void,
            local,
            hidden,
            intermediate,
            TrtllmGenMoeDType::F8E4M3FN,
            device_id,
        )?,
        fc2_scales: desc_3d(
            fc2_scale_ptr as usize as *const c_void,
            local,
            hidden / 128,
            intermediate / 128,
            TrtllmGenMoeDType::F32,
            device_id,
        )?,
        global_experts: global,
        top_k: to_i64(top_k, "top_k")?,
        expert_groups: to_i64(expert_groups, "expert_groups")?,
        topk_group: to_i64(topk_group, "topk_group")?,
        local_expert_offset: to_i64(local_expert_offset, "local_expert_offset")?,
        tactic: options.tactic,
        routed_scaling_factor,
        enable_pdl: options.enable_pdl,
        stream: stream.cu_stream().cast(),
    };
    trtllm_gen_fp8_block_scale_moe_sm100(&params)
}

#[cfg(feature = "cudarc")]
struct CudarcExpectedLengths {
    routing_logits: usize,
    hidden: usize,
    hidden_scales: usize,
    fc1: usize,
    fc1_scales: usize,
    fc2: usize,
    fc2_scales: usize,
}

#[cfg(feature = "cudarc")]
impl CudarcExpectedLengths {
    fn new(
        tokens: usize,
        hidden: usize,
        intermediate: usize,
        global_experts: usize,
        local_experts: usize,
    ) -> Result<Self, FlashInferError> {
        if hidden % 128 != 0 || intermediate % 128 != 0 {
            return Err(FlashInferError::invalid_argument(
                "hidden_size and intermediate_size must be divisible by 128",
            ));
        }
        let mul = |values: &[usize], name: &str| {
            values.iter().try_fold(1_usize, |acc, value| {
                acc.checked_mul(*value)
                    .ok_or_else(|| FlashInferError::invalid_argument(format!("{name} overflow")))
            })
        };
        let fc1_size = intermediate
            .checked_mul(2)
            .ok_or_else(|| FlashInferError::invalid_argument("2 * intermediate_size overflow"))?;
        Ok(Self {
            routing_logits: mul(&[tokens, global_experts], "routing_logits length")?,
            hidden: mul(&[tokens, hidden], "hidden length")?,
            hidden_scales: mul(&[hidden / 128, tokens], "hidden scale length")?,
            fc1: mul(&[local_experts, 2, intermediate, hidden], "fc1 length")?,
            fc1_scales: mul(
                &[local_experts, fc1_size / 128, hidden / 128],
                "fc1 scale length",
            )?,
            fc2: mul(&[local_experts, hidden, intermediate], "fc2 length")?,
            fc2_scales: mul(
                &[local_experts, hidden / 128, intermediate / 128],
                "fc2 scale length",
            )?,
        })
    }
}

#[cfg(feature = "cudarc")]
fn check_len(name: &str, actual: usize, expected: usize) -> Result<(), FlashInferError> {
    if actual != expected {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} length must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

#[cfg(feature = "cudarc")]
fn to_i64(value: usize, name: &str) -> Result<i64, FlashInferError> {
    i64::try_from(value)
        .map_err(|_| FlashInferError::invalid_argument(format!("{name} does not fit in i64")))
}

#[cfg(feature = "cudarc")]
fn desc_2d(
    ptr: *const c_void,
    rows: i64,
    cols: i64,
    dtype: TrtllmGenMoeDType,
    device_id: i32,
) -> TrtllmGenMoeTensor2DDesc {
    TrtllmGenMoeTensor2DDesc {
        ptr,
        rows,
        cols,
        stride_row: cols,
        stride_col: 1,
        dtype,
        device_id,
    }
}

#[cfg(feature = "cudarc")]
fn desc_3d(
    ptr: *const c_void,
    dim0: i64,
    dim1: i64,
    dim2: i64,
    dtype: TrtllmGenMoeDType,
    device_id: i32,
) -> Result<TrtllmGenMoeTensor3DDesc, FlashInferError> {
    let stride0 = dim1
        .checked_mul(dim2)
        .ok_or_else(|| FlashInferError::invalid_argument("rank-3 tensor stride overflow"))?;
    Ok(TrtllmGenMoeTensor3DDesc {
        ptr,
        dim0,
        dim1,
        dim2,
        stride0,
        stride1: dim2,
        stride2: 1,
        dtype,
        device_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr() -> *const c_void {
        std::ptr::NonNull::<u8>::dangling().as_ptr().cast()
    }

    fn desc2(rows: i64, cols: i64, dtype: TrtllmGenMoeDType) -> TrtllmGenMoeTensor2DDesc {
        TrtllmGenMoeTensor2DDesc {
            ptr: ptr(),
            rows,
            cols,
            stride_row: cols,
            stride_col: 1,
            dtype,
            device_id: 0,
        }
    }

    fn desc3(
        dim0: i64,
        dim1: i64,
        dim2: i64,
        dtype: TrtllmGenMoeDType,
    ) -> TrtllmGenMoeTensor3DDesc {
        TrtllmGenMoeTensor3DDesc {
            ptr: ptr(),
            dim0,
            dim1,
            dim2,
            stride0: dim1 * dim2,
            stride1: dim2,
            stride2: 1,
            dtype,
            device_id: 0,
        }
    }

    fn valid_params() -> TrtllmGenFp8BlockScaleMoeSm100Params {
        let tokens = 2;
        let hidden = 256;
        let intermediate = 128;
        let global_experts = 8;
        let local_experts = 4;
        TrtllmGenFp8BlockScaleMoeSm100Params {
            out: desc2(tokens, hidden, TrtllmGenMoeDType::BF16),
            routing_logits: desc2(tokens, global_experts, TrtllmGenMoeDType::F32),
            routing_bias: Some(TrtllmGenMoeTensor1DDesc {
                ptr: ptr(),
                len: global_experts,
                stride: 1,
                dtype: TrtllmGenMoeDType::F32,
                device_id: 0,
            }),
            hidden_states: desc2(tokens, hidden, TrtllmGenMoeDType::F8E4M3FN),
            hidden_states_scale: desc2(hidden / 128, tokens, TrtllmGenMoeDType::F32),
            fc1_weights: desc3(
                local_experts,
                2 * intermediate,
                hidden,
                TrtllmGenMoeDType::F8E4M3FN,
            ),
            fc1_scales: desc3(
                local_experts,
                2 * intermediate / 128,
                hidden / 128,
                TrtllmGenMoeDType::F32,
            ),
            fc2_weights: desc3(
                local_experts,
                hidden,
                intermediate,
                TrtllmGenMoeDType::F8E4M3FN,
            ),
            fc2_scales: desc3(
                local_experts,
                hidden / 128,
                intermediate / 128,
                TrtllmGenMoeDType::F32,
            ),
            global_experts,
            top_k: 2,
            expert_groups: 2,
            topk_group: 1,
            local_expert_offset: 0,
            tactic: None,
            routed_scaling_factor: 1.0,
            enable_pdl: true,
            stream: std::ptr::null_mut(),
        }
    }

    #[test]
    fn validates_glm_style_shapes() {
        valid_params().validate().expect("valid params");
    }

    #[test]
    fn rejects_wrong_activation_scale_layout() {
        let mut params = valid_params();
        params.hidden_states_scale = desc2(1, 2, TrtllmGenMoeDType::F32);
        let error = params.validate().expect_err("invalid activation scales");
        assert!(error.to_string().contains("hidden_states_scale shape"));
    }

    #[test]
    fn rejects_unsupported_deepseek_top_k() {
        let mut params = valid_params();
        params.global_experts = 32;
        params.routing_logits = desc2(2, 32, TrtllmGenMoeDType::F32);
        params.routing_bias = Some(TrtllmGenMoeTensor1DDesc {
            len: 32,
            ..params.routing_bias.expect("bias")
        });
        params.top_k = 23;
        let error = params.validate().expect_err("top_k above kernel limit");
        assert!(error.to_string().contains("grouped-routing"));
    }

    #[test]
    fn v064_abi_uses_deepseek_quantization_discriminant() {
        assert_eq!(DEEPSEEK_FP8_QUANTIZATION, 1);
    }
}
