use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLManagedTensorVersioned, DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION,
    DLPackVersion, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_FLOAT, KDL_FLOAT8_E4M3FN, KDL_INT,
    KDL_UINT, TVMFFIAny, any_bool, any_dltensor_ptr, any_dtype, any_i64, any_none,
    any_object_handle, any_tensor_object,
};
use crate::norm::DType;
use crate::runtime::FlashInferRuntime;

const MODULE_GET_FUNCTION_GLOBAL: &str = "ffi.ModuleGetFunction";
const ARRAY_GLOBAL: &str = "ffi.Array";
const RUN_MOE_METHOD_NAME: &str = "run_moe";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedMoeBackend {
    Sm90,
    Sm100,
    Sm103,
    Sm120,
}

impl FusedMoeBackend {
    fn kernel_uri(self) -> &'static str {
        match self {
            FusedMoeBackend::Sm90 => "fused_moe_90",
            FusedMoeBackend::Sm100 => "fused_moe_100",
            FusedMoeBackend::Sm103 => "fused_moe_103",
            FusedMoeBackend::Sm120 => "fused_moe_120",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedMoeActivationType {
    Gelu,
    Relu,
    Silu,
    Swiglu,
    Geglu,
    SwigluBias,
    Relu2,
    Identity,
}

impl FusedMoeActivationType {
    fn as_i64(self) -> i64 {
        match self {
            FusedMoeActivationType::Gelu => 0,
            FusedMoeActivationType::Relu => 1,
            FusedMoeActivationType::Silu => 2,
            FusedMoeActivationType::Swiglu => 3,
            FusedMoeActivationType::Geglu => 4,
            FusedMoeActivationType::SwigluBias => 5,
            FusedMoeActivationType::Relu2 => 6,
            FusedMoeActivationType::Identity => 7,
        }
    }

    fn is_gated(self) -> bool {
        matches!(
            self,
            FusedMoeActivationType::Swiglu
                | FusedMoeActivationType::Geglu
                | FusedMoeActivationType::SwigluBias
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMoeTensor2DDesc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub dtype: DType,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMoeTensor3DDesc {
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
pub struct FusedMoeTensor2DI32Desc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMoeTensor2DF32Desc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMoeTensor0DF32Desc {
    pub ptr: *const c_void,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMoeTensor1DF32Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMoeTensor3DF32Desc {
    pub ptr: *const c_void,
    pub dim0: i64,
    pub dim1: i64,
    pub dim2: i64,
    pub stride0: i64,
    pub stride1: i64,
    pub stride2: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub enum FusedMoeFp8ActScaleDesc {
    Scalar(FusedMoeTensor0DF32Desc),
    PerExpert(FusedMoeTensor1DF32Desc),
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMoeFp8PerTensorQuantParams {
    /// FC1 dequant scale, rank-1 f32: `[num_experts_on_rank]`.
    pub fc1_dequant: FusedMoeTensor1DF32Desc,
    /// FC2 activation quant scale, either rank-0 f32 scalar or rank-1 f32:
    /// `[num_experts_on_rank]`.
    pub fc2_quant: FusedMoeFp8ActScaleDesc,
    /// FC2 dequant scale, rank-1 f32: `[num_experts_on_rank]`.
    pub fc2_dequant: FusedMoeTensor1DF32Desc,
    /// FC1 input dequant scale, rank-0 f32 scalar.
    pub fc1_input_dequant: FusedMoeTensor0DF32Desc,
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMoeDeepSeekFp8BlockScaleQuantParams {
    /// FC1 block scales, rank-3 f32:
    /// `[num_experts_on_rank, fc1_expert_weights.dim1 / 128, hidden_size / 128]`.
    pub fc1_scales: FusedMoeTensor3DF32Desc,
    /// FC2 block scales, rank-3 f32:
    /// `[num_experts_on_rank, hidden_size / 128, inter_size / 128]`.
    pub fc2_scales: FusedMoeTensor3DF32Desc,
}

/// Quantization parameters for grouped INT4 (W4) MoE.
///
/// Maps to FlashInfer's `isInt4Quant()` quant-scales array (8 entries). Slots
/// that are `None` are emitted as zero-`numel` placeholder tensors, which the
/// C++ binding interprets as `nullptr` and the kernel substitutes with
/// sensible defaults (alpha=1.0, no zero-points, internal per-token FP8 quant
/// scales when `use_w4_group_scaling=true`).
///
/// Cross-reference: `flashinfer/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu::getQuantParams`,
/// `Expecting 8 quant scales for INT4 quantization`.
#[derive(Debug, Clone, Copy)]
pub struct FusedMoeInt4GroupScaleQuantParams {
    /// FC1 per-group weight dequant scales, rank-3:
    /// `[num_experts_on_rank, fc1_inter_size, hidden_size / group_size]`.
    ///
    /// Stored as the activation dtype (BF16/F16) per upstream contract; the
    /// kernel re-interprets the underlying bytes.
    pub fc1_weight_scales: FusedMoeTensor3DDesc,
    /// FC2 per-group weight dequant scales, rank-3:
    /// `[num_experts_on_rank, hidden_size, inter_size / group_size]`.
    pub fc2_weight_scales: FusedMoeTensor3DDesc,
    /// Optional FC1 pre-quant activation scales (W4AFP8 path), rank-2:
    /// `[num_experts_on_rank, hidden_size]`, activation dtype.
    pub fc1_act_scales: Option<FusedMoeTensor2DDesc>,
    /// Optional FC2 pre-quant activation scales, rank-2:
    /// `[num_experts_on_rank, inter_size, 1]` or `[num_experts_on_rank, inter_size]`,
    /// activation dtype.
    pub fc2_act_scales: Option<FusedMoeTensor2DDesc>,
    /// Optional FC1 per-group weight zero points (asymmetric quant), rank-3
    /// same shape as `fc1_weight_scales`.
    pub fc1_weight_zeros: Option<FusedMoeTensor3DDesc>,
    /// Optional FC2 per-group weight zero points, rank-3, same shape as
    /// `fc2_weight_scales`.
    pub fc2_weight_zeros: Option<FusedMoeTensor3DDesc>,
    /// Optional FC1 per-expert alpha (= `weight_scale_2 * input_scale_max`),
    /// rank-1 f32: `[num_experts_on_rank]`.
    pub fc1_alpha: Option<FusedMoeTensor1DF32Desc>,
    /// Optional FC2 per-expert alpha, rank-1 f32: `[num_experts_on_rank]`.
    pub fc2_alpha: Option<FusedMoeTensor1DF32Desc>,
}

/// Quantization parameters for NVFP4 fused MoE.
///
/// Maps to FlashInfer's `isNvfp4Quant()` quant-scales array (6 entries) in
/// `flashinfer/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu::getQuantParams`
/// (`isNvfp4Quant()` branch):
///   [0] fc1_act_global   (rank-0 F32)            — `1 / amax_fc1_input * (E4M3_MAX*E2M1_MAX)`
///   [1] fc1_weight_block (rank-3 FP8 E4M3)       — swizzled `[E, round_up(2n,128), round_up(k/16,4)]`
///   [2] fc1_global       (rank-1 F32 `[E]`)      — `1 / (fc1_act_global * weight_scale_2_fc1)`
///   [3] fc2_act_global   (rank-0 F32)            — for FC2 input (= FC1 output post-SwiGLU)
///   [4] fc2_weight_block (rank-3 FP8 E4M3)       — swizzled `[E, round_up(k,128), round_up(n/16,4)]`
///   [5] fc2_global       (rank-1 F32 `[E]`)
///
/// Weight tensors themselves are FP4 E2M1 packed (16 nibbles per int64 element);
/// the descriptor on `FusedMoeParams::fc1/fc2_expert_weights` must use
/// `dim2 = in / 16` and the binding sets `mWeightDtype = dl_int64` via the
/// `Nvfp4` variant of this enum.
#[derive(Debug, Clone, Copy)]
pub struct FusedMoeNvfp4QuantParams {
    pub fc1_act_global: FusedMoeTensor0DF32Desc,
    pub fc1_weight_block: FusedMoeTensor3DDesc,
    pub fc1_global: FusedMoeTensor1DF32Desc,
    pub fc2_act_global: FusedMoeTensor0DF32Desc,
    pub fc2_weight_block: FusedMoeTensor3DDesc,
    pub fc2_global: FusedMoeTensor1DF32Desc,
}

#[derive(Debug, Clone, Copy)]
pub enum FusedMoeQuantization {
    Fp8PerTensor(FusedMoeFp8PerTensorQuantParams),
    DeepSeekFp8BlockScale(FusedMoeDeepSeekFp8BlockScaleQuantParams),
    /// Grouped INT4 weights with optional FP8 activation pre-quant.
    ///
    /// - W4A16 (BF16 activations, INT4 weights): set
    ///   `use_w4_group_scaling=false`, `use_packed_weights=true`,
    ///   leave act-scales / alpha as `None`.
    /// - W4AFP8 (BF16->FP8 prequant, INT4 weights): set
    ///   `use_w4_group_scaling=true`, `use_packed_weights=true`,
    ///   leave act-scales / alpha as `None` to let the kernel derive them
    ///   from per-token statistics, or supply explicit scales for parity with
    ///   trained quant configs.
    Int4GroupScale(FusedMoeInt4GroupScaleQuantParams),
    /// NVFP4 (FP4 E2M1 weights + FP8 E4M3 block scales + F32 per-tensor scales,
    /// BF16/F16 activations). Requires Blackwell (SM100/SM103/SM120). Weight
    /// tensors are passed as `int64` storage (16 FP4 nibbles per element).
    Nvfp4(FusedMoeNvfp4QuantParams),
}

#[derive(Debug, Clone, Copy)]
pub struct FusedMoeParams {
    /// Output tensor, rank-2: `[num_tokens, hidden_size]`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu::runMoe`.
    pub out: FusedMoeTensor2DDesc,
    /// Input activations, rank-2: `[num_tokens, hidden_size]`.
    pub input: FusedMoeTensor2DDesc,
    /// Selected expert indices, rank-2 i32: `[num_tokens, top_k]`.
    pub token_selected_experts: FusedMoeTensor2DI32Desc,
    /// Optional expert routing scales, rank-2 f32: `[num_tokens, top_k]`.
    pub token_final_scales: Option<FusedMoeTensor2DF32Desc>,
    /// First expert weight tensor, rank-3:
    /// `[num_experts_on_rank, inter_size_or_2x, hidden_size]`.
    pub fc1_expert_weights: FusedMoeTensor3DDesc,
    /// Optional first expert bias tensor, rank-2: `[num_experts_on_rank, fc1_inter_size]`.
    pub fc1_expert_biases: Option<FusedMoeTensor2DDesc>,
    /// Second expert weight tensor, rank-3:
    /// `[num_experts_on_rank, hidden_size, inter_size]`.
    pub fc2_expert_weights: FusedMoeTensor3DDesc,
    /// Optional second expert bias tensor, rank-2: `[num_experts_on_rank, hidden_size]`.
    pub fc2_expert_biases: Option<FusedMoeTensor2DDesc>,
    /// Kernel backend URI family (`fused_moe_90/100/103/120`).
    pub backend: FusedMoeBackend,
    /// Activation mode enum from FlashInfer/TensorRT-LLM fused MoE bindings.
    pub activation: FusedMoeActivationType,
    /// Whether to enable producer/consumer launch in compatible kernels.
    pub enable_pdl: bool,
    /// Tensor-parallel world size.
    pub tp_size: i64,
    /// Tensor-parallel rank.
    pub tp_rank: i64,
    /// Expert-parallel world size.
    pub ep_size: i64,
    /// Expert-parallel rank.
    pub ep_rank: i64,
    /// Whether to enable alltoall in expert-parallel flow.
    pub enable_alltoall: bool,
    /// Optional CUTLASS profile ids as `[gemm1_profile_id, gemm2_profile_id]`.
    ///
    /// If omitted, FlashInfer host code selects defaults.
    pub profile_ids: Option<[i64; 2]>,
    /// Optional quantization mode/scales consumed by CUTLASS fused MoE.
    pub quantization: Option<FusedMoeQuantization>,
    /// Whether expert weight tensors are bit-packed (e.g. INT4 packed
    /// 2 nibbles per byte, surfaced to flashinfer as `dl_uint8`).
    ///
    /// Required to be `true` for the INT4 (`Int4GroupScale`) variant; the
    /// flag is also forwarded to the C++ binding via `use_packed_weights`
    /// in `init`.
    pub use_packed_weights: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl FusedMoeParams {
    pub fn new(
        out: FusedMoeTensor2DDesc,
        input: FusedMoeTensor2DDesc,
        token_selected_experts: FusedMoeTensor2DI32Desc,
        fc1_expert_weights: FusedMoeTensor3DDesc,
        fc2_expert_weights: FusedMoeTensor3DDesc,
        backend: FusedMoeBackend,
        stream: *mut c_void,
    ) -> Self {
        Self {
            out,
            input,
            token_selected_experts,
            token_final_scales: None,
            fc1_expert_weights,
            fc1_expert_biases: None,
            fc2_expert_weights,
            fc2_expert_biases: None,
            backend,
            activation: FusedMoeActivationType::Swiglu,
            enable_pdl: false,
            tp_size: 1,
            tp_rank: 0,
            ep_size: 1,
            ep_rank: 0,
            enable_alltoall: false,
            profile_ids: None,
            quantization: None,
            use_packed_weights: false,
            stream,
        }
    }

    pub fn with_token_final_scales(mut self, token_final_scales: FusedMoeTensor2DF32Desc) -> Self {
        self.token_final_scales = Some(token_final_scales);
        self
    }

    pub fn with_fc1_expert_biases(mut self, fc1_expert_biases: FusedMoeTensor2DDesc) -> Self {
        self.fc1_expert_biases = Some(fc1_expert_biases);
        self
    }

    pub fn with_fc2_expert_biases(mut self, fc2_expert_biases: FusedMoeTensor2DDesc) -> Self {
        self.fc2_expert_biases = Some(fc2_expert_biases);
        self
    }

    pub fn with_activation(mut self, activation: FusedMoeActivationType) -> Self {
        self.activation = activation;
        self
    }

    pub fn with_enable_pdl(mut self, enable_pdl: bool) -> Self {
        self.enable_pdl = enable_pdl;
        self
    }

    pub fn with_tensor_parallel(mut self, tp_size: i64, tp_rank: i64) -> Self {
        self.tp_size = tp_size;
        self.tp_rank = tp_rank;
        self
    }

    pub fn with_expert_parallel(mut self, ep_size: i64, ep_rank: i64) -> Self {
        self.ep_size = ep_size;
        self.ep_rank = ep_rank;
        self
    }

    pub fn with_enable_alltoall(mut self, enable_alltoall: bool) -> Self {
        self.enable_alltoall = enable_alltoall;
        self
    }

    pub fn with_profile_ids(mut self, gemm1_profile_id: i64, gemm2_profile_id: i64) -> Self {
        self.profile_ids = Some([gemm1_profile_id, gemm2_profile_id]);
        self
    }

    pub fn with_quantization(mut self, quantization: FusedMoeQuantization) -> Self {
        self.quantization = Some(quantization);
        self
    }

    pub fn with_fp8_per_tensor_quantization(
        mut self,
        quantization: FusedMoeFp8PerTensorQuantParams,
    ) -> Self {
        self.quantization = Some(FusedMoeQuantization::Fp8PerTensor(quantization));
        self
    }

    pub fn with_deepseek_fp8_block_scale_quantization(
        mut self,
        quantization: FusedMoeDeepSeekFp8BlockScaleQuantParams,
    ) -> Self {
        self.quantization = Some(FusedMoeQuantization::DeepSeekFp8BlockScale(quantization));
        self
    }

    /// Configure grouped INT4 quantization. Implies `use_packed_weights=true`.
    pub fn with_int4_group_scale_quantization(
        mut self,
        quantization: FusedMoeInt4GroupScaleQuantParams,
    ) -> Self {
        self.quantization = Some(FusedMoeQuantization::Int4GroupScale(quantization));
        self.use_packed_weights = true;
        self
    }

    /// Configure NVFP4 (W4FP4) quantization for Blackwell. The caller must
    /// pass `fc1/fc2_expert_weights` with `dim2 = in / 16` (storage as
    /// `int64`, 16 FP4 nibbles per element). Pre-swizzled weight block scales
    /// are required (see `flashinfer::block_scale_interleave_sm100`).
    pub fn with_nvfp4_quantization(mut self, quantization: FusedMoeNvfp4QuantParams) -> Self {
        self.quantization = Some(FusedMoeQuantization::Nvfp4(quantization));
        self.use_packed_weights = false;
        self
    }

    pub fn with_packed_weights(mut self, use_packed_weights: bool) -> Self {
        self.use_packed_weights = use_packed_weights;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_fused_moe(self)
    }
}

pub fn fused_moe(params: &FusedMoeParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: all FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { fused_moe_with_runtime(runtime, params) }
}

unsafe fn fused_moe_with_runtime(
    runtime: &FlashInferRuntime,
    params: &FusedMoeParams,
) -> Result<(), FlashInferError> {
    let mut out_shape = [params.out.rows, params.out.cols];
    let mut out_strides = [params.out.stride_row, params.out.stride_col];
    let out_tensor = tensor_2d(
        params.out.ptr,
        params.out.dtype,
        params.out.device_id,
        &mut out_shape,
        &mut out_strides,
    );

    let mut input_shape = [params.input.rows, params.input.cols];
    let mut input_strides = [params.input.stride_row, params.input.stride_col];
    let input_tensor = tensor_2d(
        params.input.ptr,
        params.input.dtype,
        params.input.device_id,
        &mut input_shape,
        &mut input_strides,
    );

    let mut token_selected_experts_shape = [
        params.token_selected_experts.rows,
        params.token_selected_experts.cols,
    ];
    let mut token_selected_experts_strides = [
        params.token_selected_experts.stride_row,
        params.token_selected_experts.stride_col,
    ];
    let token_selected_experts_tensor = tensor_2d_i32(
        params.token_selected_experts.ptr,
        params.token_selected_experts.device_id,
        &mut token_selected_experts_shape,
        &mut token_selected_experts_strides,
    );

    let mut token_final_scales_shape = [0_i64, 0_i64];
    let mut token_final_scales_strides = [0_i64, 0_i64];
    let token_final_scales_tensor = params.token_final_scales.map(|desc| {
        tensor_2d_f32(
            desc.ptr,
            desc.device_id,
            &mut token_final_scales_shape,
            &mut token_final_scales_strides,
            desc.rows,
            desc.cols,
            desc.stride_row,
            desc.stride_col,
        )
    });

    let use_deepseek_fp8_block_scale = matches!(
        params.quantization,
        Some(FusedMoeQuantization::DeepSeekFp8BlockScale(_))
    );
    let use_w4_group_scaling = matches!(
        params.quantization,
        Some(FusedMoeQuantization::Int4GroupScale(_))
    );
    let use_nvfp4 = matches!(params.quantization, Some(FusedMoeQuantization::Nvfp4(_)));
    let use_packed_weights = params.use_packed_weights;

    // Weight-dtype selection mirrors the host-side dispatch in
    // `flashinfer_cutlass_fused_moe_binding.cu::isInt4Quant()` /
    // `isNvfp4Quant()`:
    //
    //   - Grouped INT4 / packed: `mWeightDtype == dl_uint8` (2 nibbles per
    //     byte; kernel template uses `cutlass::uint4b_t`).
    //   - NVFP4: `mWeightDtype == dl_int64` (16 FP4 E2M1 nibbles per int64
    //     element; kernel template uses `__nv_fp4_e2m1`).
    //   - Otherwise: native weight dtype.
    let weight_dl_dtype = if use_nvfp4 {
        DLDataType {
            code: KDL_INT,
            bits: 64,
            lanes: 1,
        }
    } else if use_w4_group_scaling || use_packed_weights {
        DLDataType {
            code: KDL_UINT,
            bits: 8,
            lanes: 1,
        }
    } else {
        dl_dtype_from_dtype(params.fc1_expert_weights.dtype)
    };

    let mut fc1_expert_weights_shape = [
        params.fc1_expert_weights.dim0,
        params.fc1_expert_weights.dim1,
        params.fc1_expert_weights.dim2,
    ];
    let mut fc1_expert_weights_strides = [
        params.fc1_expert_weights.stride0,
        params.fc1_expert_weights.stride1,
        params.fc1_expert_weights.stride2,
    ];
    let fc1_expert_weights_tensor = tensor_3d_with_dl_dtype(
        params.fc1_expert_weights.ptr,
        weight_dl_dtype,
        params.fc1_expert_weights.device_id,
        &mut fc1_expert_weights_shape,
        &mut fc1_expert_weights_strides,
    );

    let mut fc1_expert_biases_shape = [0_i64, 0_i64];
    let mut fc1_expert_biases_strides = [0_i64, 0_i64];
    let fc1_expert_biases_tensor = params.fc1_expert_biases.map(|desc| {
        tensor_2d(
            desc.ptr,
            desc.dtype,
            desc.device_id,
            &mut fc1_expert_biases_shape,
            &mut fc1_expert_biases_strides,
        )
    });

    let mut fc2_expert_weights_shape = [
        params.fc2_expert_weights.dim0,
        params.fc2_expert_weights.dim1,
        params.fc2_expert_weights.dim2,
    ];
    let mut fc2_expert_weights_strides = [
        params.fc2_expert_weights.stride0,
        params.fc2_expert_weights.stride1,
        params.fc2_expert_weights.stride2,
    ];
    let fc2_expert_weights_tensor = tensor_3d_with_dl_dtype(
        params.fc2_expert_weights.ptr,
        weight_dl_dtype,
        params.fc2_expert_weights.device_id,
        &mut fc2_expert_weights_shape,
        &mut fc2_expert_weights_strides,
    );

    let mut fc2_expert_biases_shape = [0_i64, 0_i64];
    let mut fc2_expert_biases_strides = [0_i64, 0_i64];
    let fc2_expert_biases_tensor = params.fc2_expert_biases.map(|desc| {
        tensor_2d(
            desc.ptr,
            desc.dtype,
            desc.device_id,
            &mut fc2_expert_biases_shape,
            &mut fc2_expert_biases_strides,
        )
    });

    let init_args: [TVMFFIAny; 7] = [
        any_dtype(dl_dtype_from_dtype(params.input.dtype)),
        any_dtype(weight_dl_dtype),
        any_dtype(dl_dtype_from_dtype(params.out.dtype)),
        any_bool(use_deepseek_fp8_block_scale),
        any_bool(use_w4_group_scaling),
        any_bool(false), // use_mxfp8_act_scaling (TODO expose)
        any_bool(use_packed_weights),
    ];
    let mut module_result_view = any_none();

    // SAFETY: stream context API contract comes from tvm ffi and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.input.device_id, params.stream)? };
    let mut restore_guard =
        StreamRestoreGuard::new(runtime, params.input.device_id, previous_stream);

    let call_result = (|| -> Result<(), FlashInferError> {
        // SAFETY: argument packing follows TVMFFIAny ABI.
        unsafe {
            runtime.call_fused_moe_init(
                params.backend.kernel_uri(),
                init_args.as_ptr(),
                init_args.len() as i32,
                &mut module_result_view as *mut _,
            )?;
        }

        // SAFETY: result view is produced by TVM-FFI and copied to owned Any.
        let module_result_owned = unsafe { runtime.any_view_to_owned(&module_result_view)? };
        let mut module_guard = AnyObjectDecRefGuard::new(runtime, &module_result_owned);

        // SAFETY: function is resolved from global table; caller owns reference and must decref.
        let module_get_function_handle =
            unsafe { runtime.get_global_function(MODULE_GET_FUNCTION_GLOBAL)? };
        let mut module_get_function_guard =
            RawObjectDecRefGuard::new(runtime, module_get_function_handle);

        // SAFETY: string object is created by TVM-FFI and must be decref'd if boxed.
        let run_moe_name = unsafe { runtime.string_to_any(RUN_MOE_METHOD_NAME)? };
        let mut run_moe_name_guard = AnyObjectDecRefGuard::new(runtime, &run_moe_name);

        let mut module_get_args: [TVMFFIAny; 3] =
            [module_result_owned, run_moe_name, any_bool(false)];
        let mut run_moe_result_view = any_none();

        // SAFETY: invoking TVM Function object with ABI-packed args.
        unsafe {
            runtime.call_function(
                module_get_function_handle,
                module_get_args.as_mut_ptr(),
                module_get_args.len() as i32,
                &mut run_moe_result_view as *mut _,
            )?;
        }

        // SAFETY: result view is produced by TVM-FFI and copied to owned Any.
        let run_moe_result_owned = unsafe { runtime.any_view_to_owned(&run_moe_result_view)? };
        let mut run_moe_function_guard = AnyObjectDecRefGuard::new(runtime, &run_moe_result_owned);

        let run_moe_function_handle =
            any_object_handle(&run_moe_result_owned).ok_or_else(|| {
                FlashInferError::invalid_argument(
                    "fused moe module did not provide `run_moe` callable",
                )
            })?;

        let profile_ids_owned =
            if let Some([gemm1_profile_id, gemm2_profile_id]) = params.profile_ids {
                // SAFETY: function is resolved from global table; caller owns reference and must decref.
                let array_ctor_handle = unsafe { runtime.get_global_function(ARRAY_GLOBAL)? };
                let mut array_ctor_guard = RawObjectDecRefGuard::new(runtime, array_ctor_handle);

                let mut profile_id_args = [any_i64(gemm1_profile_id), any_i64(gemm2_profile_id)];
                let mut profile_ids_view = any_none();

                // SAFETY: invoking TVM Function object with ABI-packed args.
                unsafe {
                    runtime.call_function(
                        array_ctor_handle,
                        profile_id_args.as_mut_ptr(),
                        profile_id_args.len() as i32,
                        &mut profile_ids_view as *mut _,
                    )?;
                }

                // SAFETY: result view is produced by TVM-FFI and copied to owned Any.
                let profile_ids_owned = unsafe { runtime.any_view_to_owned(&profile_ids_view)? };
                array_ctor_guard.release_now();
                Some(profile_ids_owned)
            } else {
                None
            };
        let mut profile_ids_guard = profile_ids_owned
            .as_ref()
            .map(|any| AnyObjectDecRefGuard::new(runtime, any));
        let profile_ids_any = profile_ids_owned.as_ref().copied().unwrap_or_else(any_none);

        let mut quant_tensor_guards: Vec<RawObjectDecRefGuard<'_>> = Vec::new();
        let mut quant_scales_array_guard: Option<AnyObjectDecRefGuard<'_>> = None;
        let quant_scales_any = if let Some(quantization) = params.quantization {
            let mut scale_args = Vec::<TVMFFIAny>::new();
            match quantization {
                FusedMoeQuantization::Fp8PerTensor(quant) => {
                    let mut fc1_dequant_shape = [quant.fc1_dequant.len];
                    let mut fc1_dequant_strides = [quant.fc1_dequant.stride];
                    let fc1_dequant_tensor = tensor_1d_f32(
                        quant.fc1_dequant.ptr,
                        quant.fc1_dequant.device_id,
                        &mut fc1_dequant_shape,
                        &mut fc1_dequant_strides,
                        quant.fc1_dequant.len,
                        quant.fc1_dequant.stride,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc1_dequant_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    match quant.fc2_quant {
                        FusedMoeFp8ActScaleDesc::Scalar(scalar) => {
                            let fc2_quant_tensor = tensor_0d_f32(scalar.ptr, scalar.device_id);
                            unsafe {
                                push_quant_scale_tensor_arg(
                                    runtime,
                                    &fc2_quant_tensor,
                                    &mut scale_args,
                                    &mut quant_tensor_guards,
                                )?
                            };
                        }
                        FusedMoeFp8ActScaleDesc::PerExpert(per_expert) => {
                            let mut fc2_quant_shape = [per_expert.len];
                            let mut fc2_quant_strides = [per_expert.stride];
                            let fc2_quant_tensor = tensor_1d_f32(
                                per_expert.ptr,
                                per_expert.device_id,
                                &mut fc2_quant_shape,
                                &mut fc2_quant_strides,
                                per_expert.len,
                                per_expert.stride,
                            );
                            unsafe {
                                push_quant_scale_tensor_arg(
                                    runtime,
                                    &fc2_quant_tensor,
                                    &mut scale_args,
                                    &mut quant_tensor_guards,
                                )?
                            };
                        }
                    }

                    let mut fc2_dequant_shape = [quant.fc2_dequant.len];
                    let mut fc2_dequant_strides = [quant.fc2_dequant.stride];
                    let fc2_dequant_tensor = tensor_1d_f32(
                        quant.fc2_dequant.ptr,
                        quant.fc2_dequant.device_id,
                        &mut fc2_dequant_shape,
                        &mut fc2_dequant_strides,
                        quant.fc2_dequant.len,
                        quant.fc2_dequant.stride,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc2_dequant_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let fc1_input_dequant_tensor = tensor_0d_f32(
                        quant.fc1_input_dequant.ptr,
                        quant.fc1_input_dequant.device_id,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc1_input_dequant_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };
                }
                FusedMoeQuantization::DeepSeekFp8BlockScale(quant) => {
                    let mut fc1_scales_shape = [
                        quant.fc1_scales.dim0,
                        quant.fc1_scales.dim1,
                        quant.fc1_scales.dim2,
                    ];
                    let mut fc1_scales_strides = [
                        quant.fc1_scales.stride0,
                        quant.fc1_scales.stride1,
                        quant.fc1_scales.stride2,
                    ];
                    let fc1_scales_tensor = tensor_3d_f32(
                        quant.fc1_scales.ptr,
                        quant.fc1_scales.device_id,
                        &mut fc1_scales_shape,
                        &mut fc1_scales_strides,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc1_scales_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc2_scales_shape = [
                        quant.fc2_scales.dim0,
                        quant.fc2_scales.dim1,
                        quant.fc2_scales.dim2,
                    ];
                    let mut fc2_scales_strides = [
                        quant.fc2_scales.stride0,
                        quant.fc2_scales.stride1,
                        quant.fc2_scales.stride2,
                    ];
                    let fc2_scales_tensor = tensor_3d_f32(
                        quant.fc2_scales.ptr,
                        quant.fc2_scales.device_id,
                        &mut fc2_scales_shape,
                        &mut fc2_scales_strides,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc2_scales_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };
                }
                FusedMoeQuantization::Int4GroupScale(quant) => {
                    // Emit the 8-entry quant scales array expected by
                    // `flashinfer_cutlass_fused_moe_binding.cu::getQuantParams`
                    // (`isInt4Quant()` branch):
                    //   [0] fc1_weight_scales (3-D, BF16/F16)
                    //   [1] fc2_weight_scales (3-D, BF16/F16)
                    //   [2] fc1_act_scales    (optional, 2-D, BF16/F16)
                    //   [3] fc2_act_scales    (optional, 2-D, BF16/F16)
                    //   [4] fc1_weight_zeros  (optional, 3-D, BF16/F16)
                    //   [5] fc2_weight_zeros  (optional, 3-D, BF16/F16)
                    //   [6] fc1_alpha         (optional, 1-D, F32)
                    //   [7] fc2_alpha         (optional, 1-D, F32)
                    //
                    // Optional slots are passed as zero-`numel` rank-1
                    // placeholders; the C++ binding treats `numel() == 0` as
                    // nullptr.
                    let fc1_w_dev = quant.fc1_weight_scales.device_id;
                    let fc2_w_dev = quant.fc2_weight_scales.device_id;
                    let fc1_w_dtype = quant.fc1_weight_scales.dtype;
                    let fc2_w_dtype = quant.fc2_weight_scales.dtype;

                    let mut fc1_ws_shape = [
                        quant.fc1_weight_scales.dim0,
                        quant.fc1_weight_scales.dim1,
                        quant.fc1_weight_scales.dim2,
                    ];
                    let mut fc1_ws_strides = [
                        quant.fc1_weight_scales.stride0,
                        quant.fc1_weight_scales.stride1,
                        quant.fc1_weight_scales.stride2,
                    ];
                    let fc1_ws_tensor = tensor_3d(
                        quant.fc1_weight_scales.ptr,
                        fc1_w_dtype,
                        fc1_w_dev,
                        &mut fc1_ws_shape,
                        &mut fc1_ws_strides,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc1_ws_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc2_ws_shape = [
                        quant.fc2_weight_scales.dim0,
                        quant.fc2_weight_scales.dim1,
                        quant.fc2_weight_scales.dim2,
                    ];
                    let mut fc2_ws_strides = [
                        quant.fc2_weight_scales.stride0,
                        quant.fc2_weight_scales.stride1,
                        quant.fc2_weight_scales.stride2,
                    ];
                    let fc2_ws_tensor = tensor_3d(
                        quant.fc2_weight_scales.ptr,
                        fc2_w_dtype,
                        fc2_w_dev,
                        &mut fc2_ws_shape,
                        &mut fc2_ws_strides,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc2_ws_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc1_as_shape = [0_i64, 0_i64];
                    let mut fc1_as_strides = [0_i64, 0_i64];
                    let fc1_act_tensor = quant.fc1_act_scales.map(|desc| {
                        tensor_2d(
                            desc.ptr,
                            desc.dtype,
                            desc.device_id,
                            &mut fc1_as_shape,
                            &mut fc1_as_strides,
                        )
                    });
                    let mut fc1_as_empty_shape = [0_i64];
                    let mut fc1_as_empty_strides = [1_i64];
                    let fc1_as_placeholder =
                        empty_placeholder_tensor(fc1_w_dtype, fc1_w_dev, &mut fc1_as_empty_shape, &mut fc1_as_empty_strides);
                    let fc1_as_ref = fc1_act_tensor.as_ref().unwrap_or(&fc1_as_placeholder);
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            fc1_as_ref,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc2_as_shape = [0_i64, 0_i64];
                    let mut fc2_as_strides = [0_i64, 0_i64];
                    let fc2_act_tensor = quant.fc2_act_scales.map(|desc| {
                        tensor_2d(
                            desc.ptr,
                            desc.dtype,
                            desc.device_id,
                            &mut fc2_as_shape,
                            &mut fc2_as_strides,
                        )
                    });
                    let mut fc2_as_empty_shape = [0_i64];
                    let mut fc2_as_empty_strides = [1_i64];
                    let fc2_as_placeholder =
                        empty_placeholder_tensor(fc2_w_dtype, fc2_w_dev, &mut fc2_as_empty_shape, &mut fc2_as_empty_strides);
                    let fc2_as_ref = fc2_act_tensor.as_ref().unwrap_or(&fc2_as_placeholder);
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            fc2_as_ref,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc1_wz_shape = [0_i64, 0_i64, 0_i64];
                    let mut fc1_wz_strides = [0_i64, 0_i64, 0_i64];
                    let fc1_wz_tensor = quant.fc1_weight_zeros.map(|desc| {
                        tensor_3d(
                            desc.ptr,
                            desc.dtype,
                            desc.device_id,
                            &mut fc1_wz_shape,
                            &mut fc1_wz_strides,
                        )
                    });
                    let mut fc1_wz_empty_shape = [0_i64];
                    let mut fc1_wz_empty_strides = [1_i64];
                    let fc1_wz_placeholder = empty_placeholder_tensor(
                        fc1_w_dtype,
                        fc1_w_dev,
                        &mut fc1_wz_empty_shape,
                        &mut fc1_wz_empty_strides,
                    );
                    let fc1_wz_ref = fc1_wz_tensor.as_ref().unwrap_or(&fc1_wz_placeholder);
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            fc1_wz_ref,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc2_wz_shape = [0_i64, 0_i64, 0_i64];
                    let mut fc2_wz_strides = [0_i64, 0_i64, 0_i64];
                    let fc2_wz_tensor = quant.fc2_weight_zeros.map(|desc| {
                        tensor_3d(
                            desc.ptr,
                            desc.dtype,
                            desc.device_id,
                            &mut fc2_wz_shape,
                            &mut fc2_wz_strides,
                        )
                    });
                    let mut fc2_wz_empty_shape = [0_i64];
                    let mut fc2_wz_empty_strides = [1_i64];
                    let fc2_wz_placeholder = empty_placeholder_tensor(
                        fc2_w_dtype,
                        fc2_w_dev,
                        &mut fc2_wz_empty_shape,
                        &mut fc2_wz_empty_strides,
                    );
                    let fc2_wz_ref = fc2_wz_tensor.as_ref().unwrap_or(&fc2_wz_placeholder);
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            fc2_wz_ref,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc1_alpha_shape = [0_i64];
                    let mut fc1_alpha_strides = [1_i64];
                    let fc1_alpha_tensor = match quant.fc1_alpha {
                        Some(desc) => tensor_1d_f32(
                            desc.ptr,
                            desc.device_id,
                            &mut fc1_alpha_shape,
                            &mut fc1_alpha_strides,
                            desc.len,
                            desc.stride,
                        ),
                        None => tensor_1d_f32(
                            std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
                            fc1_w_dev,
                            &mut fc1_alpha_shape,
                            &mut fc1_alpha_strides,
                            0,
                            1,
                        ),
                    };
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc1_alpha_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc2_alpha_shape = [0_i64];
                    let mut fc2_alpha_strides = [1_i64];
                    let fc2_alpha_tensor = match quant.fc2_alpha {
                        Some(desc) => tensor_1d_f32(
                            desc.ptr,
                            desc.device_id,
                            &mut fc2_alpha_shape,
                            &mut fc2_alpha_strides,
                            desc.len,
                            desc.stride,
                        ),
                        None => tensor_1d_f32(
                            std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
                            fc2_w_dev,
                            &mut fc2_alpha_shape,
                            &mut fc2_alpha_strides,
                            0,
                            1,
                        ),
                    };
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc2_alpha_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };
                }
                FusedMoeQuantization::Nvfp4(quant) => {
                    // Emit the 6-entry quant scales array expected by
                    // `flashinfer_cutlass_fused_moe_binding.cu::getQuantParams`
                    // (`isNvfp4Quant()` branch):
                    //   [0] fc1_act_global    (rank-0 F32)
                    //   [1] fc1_weight_block  (rank-3 FP8 E4M3, pre-swizzled)
                    //   [2] fc1_global        (rank-1 F32 [E])
                    //   [3] fc2_act_global    (rank-0 F32)
                    //   [4] fc2_weight_block  (rank-3 FP8 E4M3, pre-swizzled)
                    //   [5] fc2_global        (rank-1 F32 [E])

                    let fc1_act_global_tensor =
                        tensor_0d_f32(quant.fc1_act_global.ptr, quant.fc1_act_global.device_id);
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc1_act_global_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc1_wb_shape = [
                        quant.fc1_weight_block.dim0,
                        quant.fc1_weight_block.dim1,
                        quant.fc1_weight_block.dim2,
                    ];
                    let mut fc1_wb_strides = [
                        quant.fc1_weight_block.stride0,
                        quant.fc1_weight_block.stride1,
                        quant.fc1_weight_block.stride2,
                    ];
                    // The upstream FlashInfer NVFP4 binding
                    // (`flashinfer_cutlass_fused_moe_binding.cu::
                    // getQuantParams::isNvfp4Quant`) hard-checks
                    // `fc1_weight_block.dtype() == dl_int32` (4 packed FP8
                    // E4M3 bytes per int32 element). The caller-side dim2 /
                    // strides on this descriptor are already in *int32*
                    // element units; we override the DLPack dtype here so
                    // the metadata matches what the kernel expects while
                    // the underlying device buffer is unchanged.
                    let nvfp4_block_scale_dl_dtype = DLDataType {
                        code: KDL_INT,
                        bits: 32,
                        lanes: 1,
                    };
                    let fc1_wb_tensor = tensor_3d_with_dl_dtype(
                        quant.fc1_weight_block.ptr,
                        nvfp4_block_scale_dl_dtype,
                        quant.fc1_weight_block.device_id,
                        &mut fc1_wb_shape,
                        &mut fc1_wb_strides,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc1_wb_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc1_global_shape = [quant.fc1_global.len];
                    let mut fc1_global_strides = [quant.fc1_global.stride];
                    let fc1_global_tensor = tensor_1d_f32(
                        quant.fc1_global.ptr,
                        quant.fc1_global.device_id,
                        &mut fc1_global_shape,
                        &mut fc1_global_strides,
                        quant.fc1_global.len,
                        quant.fc1_global.stride,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc1_global_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let fc2_act_global_tensor =
                        tensor_0d_f32(quant.fc2_act_global.ptr, quant.fc2_act_global.device_id);
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc2_act_global_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc2_wb_shape = [
                        quant.fc2_weight_block.dim0,
                        quant.fc2_weight_block.dim1,
                        quant.fc2_weight_block.dim2,
                    ];
                    let mut fc2_wb_strides = [
                        quant.fc2_weight_block.stride0,
                        quant.fc2_weight_block.stride1,
                        quant.fc2_weight_block.stride2,
                    ];
                    // See fc1_weight_block comment above — same dl_int32 dtype
                    // override applies to fc2.
                    let fc2_wb_tensor = tensor_3d_with_dl_dtype(
                        quant.fc2_weight_block.ptr,
                        nvfp4_block_scale_dl_dtype,
                        quant.fc2_weight_block.device_id,
                        &mut fc2_wb_shape,
                        &mut fc2_wb_strides,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc2_wb_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };

                    let mut fc2_global_shape = [quant.fc2_global.len];
                    let mut fc2_global_strides = [quant.fc2_global.stride];
                    let fc2_global_tensor = tensor_1d_f32(
                        quant.fc2_global.ptr,
                        quant.fc2_global.device_id,
                        &mut fc2_global_shape,
                        &mut fc2_global_strides,
                        quant.fc2_global.len,
                        quant.fc2_global.stride,
                    );
                    unsafe {
                        push_quant_scale_tensor_arg(
                            runtime,
                            &fc2_global_tensor,
                            &mut scale_args,
                            &mut quant_tensor_guards,
                        )?
                    };
                }
            }

            // SAFETY: function is resolved from global table; caller owns reference and must decref.
            let array_ctor_handle = unsafe { runtime.get_global_function(ARRAY_GLOBAL)? };
            let mut array_ctor_guard = RawObjectDecRefGuard::new(runtime, array_ctor_handle);
            let mut quant_scales_view = any_none();
            // SAFETY: invoking TVM Function object with ABI-packed args.
            unsafe {
                runtime.call_function(
                    array_ctor_handle,
                    scale_args.as_mut_ptr(),
                    scale_args.len() as i32,
                    &mut quant_scales_view as *mut _,
                )?;
            }

            // SAFETY: result view is produced by TVM-FFI and copied to owned Any.
            let quant_scales_owned = unsafe { runtime.any_view_to_owned(&quant_scales_view)? };
            array_ctor_guard.release_now();
            quant_scales_array_guard =
                Some(AnyObjectDecRefGuard::new(runtime, &quant_scales_owned));
            quant_scales_owned
        } else {
            any_none()
        };

        let mut run_args: [TVMFFIAny; 24] = [
            any_dltensor_ptr(&out_tensor),
            any_dltensor_ptr(&input_tensor),
            any_dltensor_ptr(&token_selected_experts_tensor),
            optional_dltensor_any(token_final_scales_tensor.as_ref()),
            any_dltensor_ptr(&fc1_expert_weights_tensor),
            optional_dltensor_any(fc1_expert_biases_tensor.as_ref()),
            any_dltensor_ptr(&fc2_expert_weights_tensor),
            optional_dltensor_any(fc2_expert_biases_tensor.as_ref()),
            quant_scales_any,
            any_none(), // input_sf
            any_none(), // swiglu_alpha
            any_none(), // swiglu_beta
            any_none(), // swiglu_limit
            any_i64(params.tp_size),
            any_i64(params.tp_rank),
            any_i64(params.ep_size),
            any_i64(params.ep_rank),
            any_i64(1), // cluster_size
            any_i64(0), // cluster_rank
            any_bool(params.enable_alltoall),
            any_bool(false), // min_latency_mode
            profile_ids_any,
            any_bool(params.enable_pdl),
            any_i64(params.activation.as_i64()),
        ];
        let mut run_result = any_none();

        // SAFETY: invoking TVM Function object with ABI-packed args.
        unsafe {
            runtime.call_function(
                run_moe_function_handle,
                run_args.as_mut_ptr(),
                run_args.len() as i32,
                &mut run_result as *mut _,
            )?;
        }

        if let Some(profile_ids_guard) = profile_ids_guard.as_mut() {
            profile_ids_guard.release_now();
        }
        if let Some(quant_scales_array_guard) = quant_scales_array_guard.as_mut() {
            quant_scales_array_guard.release_now();
        }
        for quant_tensor_guard in quant_tensor_guards.iter_mut().rev() {
            quant_tensor_guard.release_now();
        }
        run_moe_function_guard.release_now();
        run_moe_name_guard.release_now();
        module_get_function_guard.release_now();
        module_guard.release_now();
        Ok(())
    })();

    let restore_result = restore_guard.restore_now();

    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn validate_fused_moe(params: &FusedMoeParams) -> Result<(), FlashInferError> {
    check_non_null(params.out.ptr, "out")?;
    check_non_null(params.input.ptr, "input")?;
    check_non_null(params.token_selected_experts.ptr, "token_selected_experts")?;
    check_non_null(params.fc1_expert_weights.ptr, "fc1_expert_weights")?;
    check_non_null(params.fc2_expert_weights.ptr, "fc2_expert_weights")?;
    if let Some(desc) = params.token_final_scales {
        check_non_null(desc.ptr, "token_final_scales")?;
    }
    if let Some(desc) = params.fc1_expert_biases {
        check_non_null(desc.ptr, "fc1_expert_biases")?;
    }
    if let Some(desc) = params.fc2_expert_biases {
        check_non_null(desc.ptr, "fc2_expert_biases")?;
    }

    check_positive("out.rows", params.out.rows)?;
    check_positive("out.cols", params.out.cols)?;
    check_positive("input.rows", params.input.rows)?;
    check_positive("input.cols", params.input.cols)?;
    check_positive(
        "token_selected_experts.rows",
        params.token_selected_experts.rows,
    )?;
    check_positive(
        "token_selected_experts.cols",
        params.token_selected_experts.cols,
    )?;
    check_positive("fc1_expert_weights.dim0", params.fc1_expert_weights.dim0)?;
    check_positive("fc1_expert_weights.dim1", params.fc1_expert_weights.dim1)?;
    check_positive("fc1_expert_weights.dim2", params.fc1_expert_weights.dim2)?;
    check_positive("fc2_expert_weights.dim0", params.fc2_expert_weights.dim0)?;
    check_positive("fc2_expert_weights.dim1", params.fc2_expert_weights.dim1)?;
    check_positive("fc2_expert_weights.dim2", params.fc2_expert_weights.dim2)?;
    check_positive("tp_size", params.tp_size)?;
    check_positive("ep_size", params.ep_size)?;

    check_last_contiguous_2d("out", params.out.stride_col)?;
    check_last_contiguous_2d("input", params.input.stride_col)?;
    check_last_contiguous_2d(
        "token_selected_experts",
        params.token_selected_experts.stride_col,
    )?;
    check_last_contiguous_3d("fc1_expert_weights", params.fc1_expert_weights.stride2)?;
    check_last_contiguous_3d("fc2_expert_weights", params.fc2_expert_weights.stride2)?;

    if params.out.rows != params.input.rows || params.out.cols != params.input.cols {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: out ({}, {}) must match input ({}, {})",
            params.out.rows, params.out.cols, params.input.rows, params.input.cols
        )));
    }

    if params.input.rows != params.token_selected_experts.rows {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: token_selected_experts.rows ({}) must match input.rows ({})",
            params.token_selected_experts.rows, params.input.rows
        )));
    }

    if params.fc1_expert_weights.dim0 != params.fc2_expert_weights.dim0 {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: fc1_expert_weights.dim0 ({}) must match fc2_expert_weights.dim0 ({})",
            params.fc1_expert_weights.dim0, params.fc2_expert_weights.dim0
        )));
    }

    // For packed INT4 / W4 paths, weights are bit-packed (2 nibbles per byte).
    // `inner_dim_multiplier` matches the C++ binding's `mInnerDimMultiplier`
    // for the active quantization mode and converts byte-counts back into
    // logical element counts for the shape contract.
    //
    // Cross-reference: `flashinfer_cutlass_fused_moe_binding.cu::runMoe`:
    //   `int64_t inter_size = fc2_expert_weights.size(2) * mInnerDimMultiplier;`
    let inner_dim_multiplier: i64 = match params.quantization {
        Some(FusedMoeQuantization::Int4GroupScale(_)) => 2,
        Some(FusedMoeQuantization::Nvfp4(_)) => 16,
        _ => 1,
    };

    let expected_fc1_dim2 = params
        .input
        .cols
        .checked_div(inner_dim_multiplier)
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "input.cols not divisible by INT4 inner dim multiplier",
            )
        })?;
    if params.fc1_expert_weights.dim2 != expected_fc1_dim2 {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: fc1_expert_weights.dim2 ({}) must equal input.cols / inner_dim_multiplier ({expected_fc1_dim2})",
            params.fc1_expert_weights.dim2
        )));
    }

    if params.fc2_expert_weights.dim1 != params.input.cols {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: fc2_expert_weights.dim1 ({}) must match input.cols ({})",
            params.fc2_expert_weights.dim1, params.input.cols
        )));
    }

    let logical_fc2_dim2 = params
        .fc2_expert_weights
        .dim2
        .checked_mul(inner_dim_multiplier)
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "fc2_expert_weights.dim2 * inner_dim_multiplier overflow",
            )
        })?;
    let expected_fc1_dim1 = if params.activation.is_gated() {
        logical_fc2_dim2.checked_mul(2).ok_or_else(|| {
            FlashInferError::invalid_argument("logical_fc2_dim2 * 2 overflow")
        })?
    } else {
        logical_fc2_dim2
    };

    if params.fc1_expert_weights.dim1 != expected_fc1_dim1 {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: fc1_expert_weights.dim1 ({}) must equal expected inter size ({expected_fc1_dim1})",
            params.fc1_expert_weights.dim1
        )));
    }

    match params.quantization {
        None => {
            if params.input.dtype == DType::F8E4M3FN
                || params.fc1_expert_weights.dtype == DType::F8E4M3FN
                || params.fc2_expert_weights.dtype == DType::F8E4M3FN
                || params.out.dtype == DType::F8E4M3FN
            {
                return Err(FlashInferError::invalid_argument(
                    "fp8 fused_moe requires explicit quantization metadata",
                ));
            }

            if params.input.dtype != params.fc1_expert_weights.dtype
                || params.input.dtype != params.fc2_expert_weights.dtype
                || params.input.dtype != params.out.dtype
            {
                return Err(FlashInferError::invalid_argument(
                    "dtype mismatch: fused_moe requires input/weights/out to share the same dtype when quantization is disabled",
                ));
            }
        }
        Some(FusedMoeQuantization::Fp8PerTensor(quant)) => {
            validate_fp8_per_tensor_quantization(params, quant)?;
        }
        Some(FusedMoeQuantization::DeepSeekFp8BlockScale(quant)) => {
            validate_deepseek_fp8_block_scale_quantization(params, quant)?;
        }
        Some(FusedMoeQuantization::Int4GroupScale(quant)) => {
            validate_int4_group_scale_quantization(params, quant)?;
        }
        Some(FusedMoeQuantization::Nvfp4(quant)) => {
            validate_nvfp4_quantization(params, quant)?;
        }
    }

    if let Some(desc) = params.token_final_scales {
        check_positive("token_final_scales.rows", desc.rows)?;
        check_positive("token_final_scales.cols", desc.cols)?;
        check_last_contiguous_2d("token_final_scales", desc.stride_col)?;
        if desc.rows != params.input.rows || desc.cols != params.token_selected_experts.cols {
            return Err(FlashInferError::invalid_argument(format!(
                "shape mismatch: token_final_scales ({}, {}) must match [input.rows={}, top_k={}]",
                desc.rows, desc.cols, params.input.rows, params.token_selected_experts.cols
            )));
        }
        if desc.device_id != params.input.device_id {
            return Err(FlashInferError::invalid_argument(
                "device mismatch: token_final_scales must be on same device as input",
            ));
        }
    }

    if let Some(desc) = params.fc1_expert_biases {
        check_positive("fc1_expert_biases.rows", desc.rows)?;
        check_positive("fc1_expert_biases.cols", desc.cols)?;
        check_last_contiguous_2d("fc1_expert_biases", desc.stride_col)?;
        if desc.rows != params.fc1_expert_weights.dim0
            || desc.cols != params.fc1_expert_weights.dim1
        {
            return Err(FlashInferError::invalid_argument(format!(
                "shape mismatch: fc1_expert_biases ({}, {}) must match [{}, {}]",
                desc.rows,
                desc.cols,
                params.fc1_expert_weights.dim0,
                params.fc1_expert_weights.dim1
            )));
        }
        if desc.dtype != params.out.dtype {
            return Err(FlashInferError::invalid_argument(
                "dtype mismatch: fc1_expert_biases must match out dtype",
            ));
        }
        if desc.device_id != params.input.device_id {
            return Err(FlashInferError::invalid_argument(
                "device mismatch: fc1_expert_biases must be on same device as input",
            ));
        }
    }

    if let Some(desc) = params.fc2_expert_biases {
        check_positive("fc2_expert_biases.rows", desc.rows)?;
        check_positive("fc2_expert_biases.cols", desc.cols)?;
        check_last_contiguous_2d("fc2_expert_biases", desc.stride_col)?;
        if desc.rows != params.fc2_expert_weights.dim0
            || desc.cols != params.fc2_expert_weights.dim1
        {
            return Err(FlashInferError::invalid_argument(format!(
                "shape mismatch: fc2_expert_biases ({}, {}) must match [{}, {}]",
                desc.rows,
                desc.cols,
                params.fc2_expert_weights.dim0,
                params.fc2_expert_weights.dim1
            )));
        }
        if desc.dtype != params.out.dtype {
            return Err(FlashInferError::invalid_argument(
                "dtype mismatch: fc2_expert_biases must match out dtype",
            ));
        }
        if desc.device_id != params.input.device_id {
            return Err(FlashInferError::invalid_argument(
                "device mismatch: fc2_expert_biases must be on same device as input",
            ));
        }
    }

    if params.out.device_id != params.input.device_id
        || params.token_selected_experts.device_id != params.input.device_id
        || params.fc1_expert_weights.device_id != params.input.device_id
        || params.fc2_expert_weights.device_id != params.input.device_id
    {
        return Err(FlashInferError::invalid_argument(
            "device mismatch across fused_moe tensors",
        ));
    }

    if params.tp_rank < 0 || params.tp_rank >= params.tp_size {
        return Err(FlashInferError::invalid_argument(format!(
            "tp_rank ({}) must be in [0, tp_size={})",
            params.tp_rank, params.tp_size
        )));
    }

    if params.ep_rank < 0 || params.ep_rank >= params.ep_size {
        return Err(FlashInferError::invalid_argument(format!(
            "ep_rank ({}) must be in [0, ep_size={})",
            params.ep_rank, params.ep_size
        )));
    }

    Ok(())
}

fn validate_fp8_per_tensor_quantization(
    params: &FusedMoeParams,
    quant: FusedMoeFp8PerTensorQuantParams,
) -> Result<(), FlashInferError> {
    check_non_null(quant.fc1_dequant.ptr, "quant.fc1_dequant")?;
    check_non_null(quant.fc2_dequant.ptr, "quant.fc2_dequant")?;
    check_non_null(quant.fc1_input_dequant.ptr, "quant.fc1_input_dequant")?;
    match quant.fc2_quant {
        FusedMoeFp8ActScaleDesc::Scalar(desc) => {
            check_non_null(desc.ptr, "quant.fc2_quant")?;
            if desc.device_id != params.input.device_id {
                return Err(FlashInferError::invalid_argument(
                    "device mismatch: quant.fc2_quant must be on same device as input",
                ));
            }
        }
        FusedMoeFp8ActScaleDesc::PerExpert(desc) => {
            check_non_null(desc.ptr, "quant.fc2_quant")?;
            check_positive("quant.fc2_quant.len", desc.len)?;
            if desc.stride != 1 {
                return Err(FlashInferError::invalid_argument(
                    "quant.fc2_quant last-dimension stride must be 1",
                ));
            }
            if desc.device_id != params.input.device_id {
                return Err(FlashInferError::invalid_argument(
                    "device mismatch: quant.fc2_quant must be on same device as input",
                ));
            }
        }
    }

    if params.input.dtype != DType::F8E4M3FN
        || params.fc1_expert_weights.dtype != DType::F8E4M3FN
        || params.fc2_expert_weights.dtype != DType::F8E4M3FN
    {
        return Err(FlashInferError::invalid_argument(
            "fp8 per-tensor fused_moe requires input/fc1/fc2 dtypes to be F8E4M3FN",
        ));
    }

    if params.out.dtype != DType::F16 && params.out.dtype != DType::BF16 {
        return Err(FlashInferError::invalid_argument(
            "fp8 per-tensor fused_moe requires out dtype to be F16 or BF16",
        ));
    }

    check_positive("quant.fc1_dequant.len", quant.fc1_dequant.len)?;
    check_positive("quant.fc2_dequant.len", quant.fc2_dequant.len)?;
    if quant.fc1_dequant.stride != 1 {
        return Err(FlashInferError::invalid_argument(
            "quant.fc1_dequant last-dimension stride must be 1",
        ));
    }
    if quant.fc2_dequant.stride != 1 {
        return Err(FlashInferError::invalid_argument(
            "quant.fc2_dequant last-dimension stride must be 1",
        ));
    }

    let num_experts_on_rank = params.fc1_expert_weights.dim0;
    if quant.fc1_dequant.len != num_experts_on_rank {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: quant.fc1_dequant.len ({}) must equal num_experts_on_rank ({num_experts_on_rank})",
            quant.fc1_dequant.len
        )));
    }
    if quant.fc2_dequant.len != num_experts_on_rank {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: quant.fc2_dequant.len ({}) must equal num_experts_on_rank ({num_experts_on_rank})",
            quant.fc2_dequant.len
        )));
    }
    if let FusedMoeFp8ActScaleDesc::PerExpert(desc) = quant.fc2_quant {
        if desc.len != num_experts_on_rank {
            return Err(FlashInferError::invalid_argument(format!(
                "shape mismatch: quant.fc2_quant.len ({}) must equal num_experts_on_rank ({num_experts_on_rank})",
                desc.len
            )));
        }
    }

    if quant.fc1_dequant.device_id != params.input.device_id
        || quant.fc2_dequant.device_id != params.input.device_id
        || quant.fc1_input_dequant.device_id != params.input.device_id
    {
        return Err(FlashInferError::invalid_argument(
            "device mismatch: fp8 per-tensor quant scales must be on same device as input",
        ));
    }

    Ok(())
}

fn validate_deepseek_fp8_block_scale_quantization(
    params: &FusedMoeParams,
    quant: FusedMoeDeepSeekFp8BlockScaleQuantParams,
) -> Result<(), FlashInferError> {
    check_non_null(quant.fc1_scales.ptr, "quant.fc1_scales")?;
    check_non_null(quant.fc2_scales.ptr, "quant.fc2_scales")?;
    check_positive("quant.fc1_scales.dim0", quant.fc1_scales.dim0)?;
    check_positive("quant.fc1_scales.dim1", quant.fc1_scales.dim1)?;
    check_positive("quant.fc1_scales.dim2", quant.fc1_scales.dim2)?;
    check_positive("quant.fc2_scales.dim0", quant.fc2_scales.dim0)?;
    check_positive("quant.fc2_scales.dim1", quant.fc2_scales.dim1)?;
    check_positive("quant.fc2_scales.dim2", quant.fc2_scales.dim2)?;
    check_last_contiguous_3d("quant.fc1_scales", quant.fc1_scales.stride2)?;
    check_last_contiguous_3d("quant.fc2_scales", quant.fc2_scales.stride2)?;

    if params.backend != FusedMoeBackend::Sm90 {
        return Err(FlashInferError::invalid_argument(
            "deepseek fp8 block-scale fused_moe currently requires backend Sm90",
        ));
    }
    if params.input.dtype != DType::BF16 || params.out.dtype != DType::BF16 {
        return Err(FlashInferError::invalid_argument(
            "deepseek fp8 block-scale fused_moe requires input/out dtype to be BF16",
        ));
    }
    if params.fc1_expert_weights.dtype != DType::F8E4M3FN
        || params.fc2_expert_weights.dtype != DType::F8E4M3FN
    {
        return Err(FlashInferError::invalid_argument(
            "deepseek fp8 block-scale fused_moe requires fc1/fc2 weight dtype to be F8E4M3FN",
        ));
    }

    if params.input.cols % 128 != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "deepseek fp8 block-scale requires hidden_size (input.cols={}) divisible by 128",
            params.input.cols
        )));
    }
    if params.fc2_expert_weights.dim2 % 128 != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "deepseek fp8 block-scale requires inter_size (fc2_expert_weights.dim2={}) divisible by 128",
            params.fc2_expert_weights.dim2
        )));
    }
    if params.fc1_expert_weights.dim1 % 128 != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "deepseek fp8 block-scale requires fc1_expert_weights.dim1 ({}) divisible by 128",
            params.fc1_expert_weights.dim1
        )));
    }

    let expected_fc1_dim1 = params.fc1_expert_weights.dim1 / 128;
    let expected_hidden_scale = params.input.cols / 128;
    let expected_fc2_dim2 = params.fc2_expert_weights.dim2 / 128;
    let num_experts_on_rank = params.fc1_expert_weights.dim0;

    if quant.fc1_scales.dim0 != num_experts_on_rank
        || quant.fc1_scales.dim1 != expected_fc1_dim1
        || quant.fc1_scales.dim2 != expected_hidden_scale
    {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: quant.fc1_scales must be [{num_experts_on_rank}, {expected_fc1_dim1}, {expected_hidden_scale}]",
        )));
    }
    if quant.fc2_scales.dim0 != num_experts_on_rank
        || quant.fc2_scales.dim1 != expected_hidden_scale
        || quant.fc2_scales.dim2 != expected_fc2_dim2
    {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: quant.fc2_scales must be [{num_experts_on_rank}, {expected_hidden_scale}, {expected_fc2_dim2}]",
        )));
    }

    if quant.fc1_scales.device_id != params.input.device_id
        || quant.fc2_scales.device_id != params.input.device_id
    {
        return Err(FlashInferError::invalid_argument(
            "device mismatch: deepseek fp8 block scales must be on same device as input",
        ));
    }

    Ok(())
}

fn validate_int4_group_scale_quantization(
    params: &FusedMoeParams,
    quant: FusedMoeInt4GroupScaleQuantParams,
) -> Result<(), FlashInferError> {
    check_non_null(quant.fc1_weight_scales.ptr, "quant.fc1_weight_scales")?;
    check_non_null(quant.fc2_weight_scales.ptr, "quant.fc2_weight_scales")?;
    check_positive("quant.fc1_weight_scales.dim0", quant.fc1_weight_scales.dim0)?;
    check_positive("quant.fc1_weight_scales.dim1", quant.fc1_weight_scales.dim1)?;
    check_positive("quant.fc1_weight_scales.dim2", quant.fc1_weight_scales.dim2)?;
    check_positive("quant.fc2_weight_scales.dim0", quant.fc2_weight_scales.dim0)?;
    check_positive("quant.fc2_weight_scales.dim1", quant.fc2_weight_scales.dim1)?;
    check_positive("quant.fc2_weight_scales.dim2", quant.fc2_weight_scales.dim2)?;
    check_last_contiguous_3d("quant.fc1_weight_scales", quant.fc1_weight_scales.stride2)?;
    check_last_contiguous_3d("quant.fc2_weight_scales", quant.fc2_weight_scales.stride2)?;

    // Weight scales must live on the same device as the inputs.
    if quant.fc1_weight_scales.device_id != params.input.device_id
        || quant.fc2_weight_scales.device_id != params.input.device_id
    {
        return Err(FlashInferError::invalid_argument(
            "device mismatch: int4 group-wise weight scales must be on same device as input",
        ));
    }

    // Scales are interpreted as the activation dtype (BF16 or F16). FP8 is
    // not a valid scale dtype.
    if quant.fc1_weight_scales.dtype == DType::F8E4M3FN
        || quant.fc2_weight_scales.dtype == DType::F8E4M3FN
    {
        return Err(FlashInferError::invalid_argument(
            "int4 group-wise weight scales must be BF16 or F16 (not F8E4M3FN)",
        ));
    }

    // Activations are BF16 or F16; out matches input dtype. Weight dtype on
    // the descriptor is informational; the FFI overrides it to `dl_uint8`.
    if params.input.dtype == DType::F8E4M3FN || params.out.dtype == DType::F8E4M3FN {
        return Err(FlashInferError::invalid_argument(
            "int4 group-wise fused_moe requires BF16/F16 input/out dtype",
        ));
    }
    if params.input.dtype != params.out.dtype {
        return Err(FlashInferError::invalid_argument(
            "int4 group-wise fused_moe requires input/out dtype to match",
        ));
    }

    if !params.use_packed_weights {
        return Err(FlashInferError::invalid_argument(
            "int4 group-wise fused_moe requires use_packed_weights=true",
        ));
    }

    Ok(())
}

fn validate_nvfp4_quantization(
    params: &FusedMoeParams,
    quant: FusedMoeNvfp4QuantParams,
) -> Result<(), FlashInferError> {
    check_non_null(quant.fc1_act_global.ptr, "quant.fc1_act_global")?;
    check_non_null(quant.fc1_weight_block.ptr, "quant.fc1_weight_block")?;
    check_non_null(quant.fc1_global.ptr, "quant.fc1_global")?;
    check_non_null(quant.fc2_act_global.ptr, "quant.fc2_act_global")?;
    check_non_null(quant.fc2_weight_block.ptr, "quant.fc2_weight_block")?;
    check_non_null(quant.fc2_global.ptr, "quant.fc2_global")?;

    check_positive("quant.fc1_weight_block.dim0", quant.fc1_weight_block.dim0)?;
    check_positive("quant.fc1_weight_block.dim1", quant.fc1_weight_block.dim1)?;
    check_positive("quant.fc1_weight_block.dim2", quant.fc1_weight_block.dim2)?;
    check_positive("quant.fc2_weight_block.dim0", quant.fc2_weight_block.dim0)?;
    check_positive("quant.fc2_weight_block.dim1", quant.fc2_weight_block.dim1)?;
    check_positive("quant.fc2_weight_block.dim2", quant.fc2_weight_block.dim2)?;
    check_positive("quant.fc1_global.len", quant.fc1_global.len)?;
    check_positive("quant.fc2_global.len", quant.fc2_global.len)?;
    check_last_contiguous_3d("quant.fc1_weight_block", quant.fc1_weight_block.stride2)?;
    check_last_contiguous_3d("quant.fc2_weight_block", quant.fc2_weight_block.stride2)?;

    let num_experts = params.fc1_expert_weights.dim0;
    if quant.fc1_weight_block.dim0 != num_experts
        || quant.fc2_weight_block.dim0 != num_experts
        || quant.fc1_global.len != num_experts
        || quant.fc2_global.len != num_experts
    {
        return Err(FlashInferError::invalid_argument(format!(
            "nvfp4 quant scales must be sized per local expert (= {num_experts}): \
             fc1_weight_block.dim0={}, fc2_weight_block.dim0={}, fc1_global.len={}, fc2_global.len={}",
            quant.fc1_weight_block.dim0,
            quant.fc2_weight_block.dim0,
            quant.fc1_global.len,
            quant.fc2_global.len
        )));
    }

    // FP8 E4M3 block scales for weights — the on-disk dtype is what callers
    // declare here. The fused-MoE NVFP4 kernel internally reinterprets the
    // byte buffer as `dl_int32` (4 packed FP8 E4M3 bytes per i32 element),
    // which we emit on the DLTensor side of the FFI (see the descriptor
    // construction below). Callers just supply the FP8 dtype as a
    // self-documenting tag.
    if quant.fc1_weight_block.dtype != DType::F8E4M3FN
        || quant.fc2_weight_block.dtype != DType::F8E4M3FN
    {
        return Err(FlashInferError::invalid_argument(
            "nvfp4 weight block scales must be F8E4M3FN",
        ));
    }

    // Activations and output are BF16 (or F16); FP8 activations would use
    // the W4FP8 / WMxfp4-AMxfp8 path rather than NVFP4.
    if params.input.dtype == DType::F8E4M3FN || params.out.dtype == DType::F8E4M3FN {
        return Err(FlashInferError::invalid_argument(
            "nvfp4 fused_moe requires BF16/F16 input/out (FP8 activations would route to W4FP8/MXFP4)",
        ));
    }
    if params.input.dtype != params.out.dtype {
        return Err(FlashInferError::invalid_argument(
            "nvfp4 fused_moe requires input/out dtype to match",
        ));
    }

    // The packed-weights flag is independent of NVFP4; the binding uses
    // `mWeightDtype == dl_int64` to detect FP4 storage and we must NOT also
    // set `use_packed_weights=true` (that flag is dedicated to the INT4
    // path).
    if params.use_packed_weights {
        return Err(FlashInferError::invalid_argument(
            "nvfp4 fused_moe must not set use_packed_weights=true (reserved for INT4 W4 path)",
        ));
    }

    // Devices must match across all scale tensors.
    let dev = params.input.device_id;
    for (name, did) in [
        ("fc1_act_global", quant.fc1_act_global.device_id),
        ("fc1_weight_block", quant.fc1_weight_block.device_id),
        ("fc1_global", quant.fc1_global.device_id),
        ("fc2_act_global", quant.fc2_act_global.device_id),
        ("fc2_weight_block", quant.fc2_weight_block.device_id),
        ("fc2_global", quant.fc2_global.device_id),
    ] {
        if did != dev {
            return Err(FlashInferError::invalid_argument(format!(
                "device mismatch: nvfp4 scale `{name}` (device {did}) must be on same device as input (device {dev})"
            )));
        }
    }

    Ok(())
}

fn optional_dltensor_any(tensor: Option<&DLTensor>) -> TVMFFIAny {
    match tensor {
        Some(tensor) => any_dltensor_ptr(tensor),
        None => any_none(),
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
        DType::F8E4M3FN => DLDataType {
            code: KDL_FLOAT8_E4M3FN,
            bits: 8,
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

fn tensor_3d(
    ptr: *const c_void,
    dtype: DType,
    device_id: i32,
    shape: &mut [i64; 3],
    strides: &mut [i64; 3],
) -> DLTensor {
    tensor_3d_with_dl_dtype(ptr, dl_dtype_from_dtype(dtype), device_id, shape, strides)
}

/// Build a rank-1 zero-`numel` placeholder `DLTensor` used to encode an
/// `Optional<TensorView>` slot of the FlashInfer fused-moe `quant_scales`
/// array as nullptr. The C++ binding checks `tensor.numel() > 0` before
/// dereferencing.
fn empty_placeholder_tensor(
    dtype: DType,
    device_id: i32,
    shape: &mut [i64; 1],
    strides: &mut [i64; 1],
) -> DLTensor {
    // Use a non-null dangling pointer to satisfy any non-null assertions in the
    // FFI path while keeping `numel() == 0` to signal the nullptr semantic.
    let placeholder_ptr = std::ptr::NonNull::<u8>::dangling().as_ptr().cast();
    DLTensor {
        data: placeholder_ptr,
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

fn tensor_3d_with_dl_dtype(
    ptr: *const c_void,
    dtype: DLDataType,
    device_id: i32,
    shape: &mut [i64; 3],
    strides: &mut [i64; 3],
) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim: 3,
        dtype,
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn tensor_2d_i32(
    ptr: *const c_void,
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
        dtype: DLDataType {
            code: KDL_INT,
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn tensor_2d_f32(
    ptr: *const c_void,
    device_id: i32,
    shape: &mut [i64; 2],
    strides: &mut [i64; 2],
    rows: i64,
    cols: i64,
    stride_row: i64,
    stride_col: i64,
) -> DLTensor {
    shape[0] = rows;
    shape[1] = cols;
    strides[0] = stride_row;
    strides[1] = stride_col;

    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim: 2,
        dtype: DLDataType {
            code: KDL_FLOAT,
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn tensor_0d_f32(ptr: *const c_void, device_id: i32) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim: 0,
        dtype: DLDataType {
            code: KDL_FLOAT,
            bits: 32,
            lanes: 1,
        },
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    }
}

fn tensor_1d_f32(
    ptr: *const c_void,
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
        dtype: DLDataType {
            code: KDL_FLOAT,
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn tensor_3d_f32(
    ptr: *const c_void,
    device_id: i32,
    shape: &mut [i64; 3],
    strides: &mut [i64; 3],
) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim: 3,
        dtype: DLDataType {
            code: KDL_FLOAT,
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

struct BorrowedDLPackTensorContext {
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

struct ManagedDLPackTensor {
    raw: *mut DLManagedTensorVersioned,
}

impl ManagedDLPackTensor {
    fn from_dltensor(tensor: &DLTensor) -> Result<Self, FlashInferError> {
        let ndim = usize::try_from(tensor.ndim).map_err(|_| {
            FlashInferError::invalid_argument(format!("negative tensor ndim {}", tensor.ndim))
        })?;

        let mut shape = Vec::with_capacity(ndim);
        let mut strides = Vec::with_capacity(ndim);
        if ndim > 0 {
            if tensor.shape.is_null() {
                return Err(FlashInferError::invalid_argument(
                    "tensor shape pointer is null for ndim > 0",
                ));
            }
            if tensor.strides.is_null() {
                return Err(FlashInferError::invalid_argument(
                    "tensor strides pointer is null for ndim > 0",
                ));
            }
            // SAFETY: `shape`/`strides` pointers must be valid for `ndim` elements for this call.
            let shape_src = unsafe { std::slice::from_raw_parts(tensor.shape, ndim) };
            // SAFETY: `shape`/`strides` pointers must be valid for `ndim` elements for this call.
            let strides_src = unsafe { std::slice::from_raw_parts(tensor.strides, ndim) };
            shape.extend_from_slice(shape_src);
            strides.extend_from_slice(strides_src);
        }

        let mut ctx = Box::new(BorrowedDLPackTensorContext {
            shape: shape.into_boxed_slice(),
            strides: strides.into_boxed_slice(),
        });
        let shape_ptr = if ndim == 0 {
            std::ptr::null_mut()
        } else {
            ctx.shape.as_mut_ptr()
        };
        let strides_ptr = if ndim == 0 {
            std::ptr::null_mut()
        } else {
            ctx.strides.as_mut_ptr()
        };
        let ctx_ptr = Box::into_raw(ctx).cast::<c_void>();

        let managed = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion {
                major: DLPACK_MAJOR_VERSION,
                minor: DLPACK_MINOR_VERSION,
            },
            manager_ctx: ctx_ptr,
            deleter: Some(borrowed_dlpack_tensor_deleter),
            flags: 0,
            dl_tensor: DLTensor {
                data: tensor.data,
                device: tensor.device,
                ndim: tensor.ndim,
                dtype: tensor.dtype,
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset: tensor.byte_offset,
            },
        });

        Ok(Self {
            raw: Box::into_raw(managed),
        })
    }

    fn as_mut_ptr(&mut self) -> *mut DLManagedTensorVersioned {
        self.raw
    }

    fn release_ownership(&mut self) {
        self.raw = std::ptr::null_mut();
    }
}

impl Drop for ManagedDLPackTensor {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        // SAFETY: `self.raw` is uniquely owned by this wrapper and not transferred.
        unsafe {
            free_borrowed_dlpack_tensor(self.raw);
        }
        self.raw = std::ptr::null_mut();
    }
}

unsafe extern "C" fn borrowed_dlpack_tensor_deleter(raw: *mut DLManagedTensorVersioned) {
    // SAFETY: called by TVM-FFI when it owns this DLManagedTensorVersioned.
    unsafe {
        free_borrowed_dlpack_tensor(raw);
    }
}

unsafe fn free_borrowed_dlpack_tensor(raw: *mut DLManagedTensorVersioned) {
    if raw.is_null() {
        return;
    }
    // SAFETY: pointer must have originated from `Box::into_raw` in `ManagedDLPackTensor`.
    let managed = unsafe { Box::from_raw(raw) };
    if !managed.manager_ctx.is_null() {
        // SAFETY: manager_ctx is allocated as `BorrowedDLPackTensorContext`.
        let _ = unsafe { Box::from_raw(managed.manager_ctx.cast::<BorrowedDLPackTensorContext>()) };
    }
}

unsafe fn push_quant_scale_tensor_arg<'a>(
    runtime: &'a FlashInferRuntime,
    tensor: &DLTensor,
    args: &mut Vec<TVMFFIAny>,
    guards: &mut Vec<RawObjectDecRefGuard<'a>>,
) -> Result<(), FlashInferError> {
    let mut managed = ManagedDLPackTensor::from_dltensor(tensor)?;
    // SAFETY: arguments follow TVM-FFI C API contract.
    let tensor_obj =
        unsafe { runtime.tensor_from_dlpack_versioned(managed.as_mut_ptr(), 1, false)? };
    managed.release_ownership();
    args.push(any_tensor_object(tensor_obj));
    guards.push(RawObjectDecRefGuard::new(runtime, tensor_obj));
    Ok(())
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

struct AnyObjectDecRefGuard<'a> {
    runtime: &'a FlashInferRuntime,
    obj: *mut c_void,
    active: bool,
}

impl<'a> AnyObjectDecRefGuard<'a> {
    fn new(runtime: &'a FlashInferRuntime, any: &TVMFFIAny) -> Self {
        let obj = any_object_handle(any).unwrap_or(std::ptr::null_mut());
        Self {
            runtime,
            obj,
            active: true,
        }
    }

    fn release_now(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        // SAFETY: object handle was returned by TVM-FFI and is valid to decref once.
        unsafe {
            self.runtime.object_dec_ref(self.obj);
        }
    }
}

impl Drop for AnyObjectDecRefGuard<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        // SAFETY: best-effort object decref in drop path.
        unsafe {
            self.runtime.object_dec_ref(self.obj);
        }
    }
}

struct RawObjectDecRefGuard<'a> {
    runtime: &'a FlashInferRuntime,
    obj: *mut c_void,
    active: bool,
}

impl<'a> RawObjectDecRefGuard<'a> {
    fn new(runtime: &'a FlashInferRuntime, obj: *mut c_void) -> Self {
        Self {
            runtime,
            obj,
            active: true,
        }
    }

    fn release_now(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        // SAFETY: object handle was returned by TVM-FFI and is valid to decref once.
        unsafe {
            self.runtime.object_dec_ref(self.obj);
        }
    }
}

impl Drop for RawObjectDecRefGuard<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        // SAFETY: best-effort object decref in drop path.
        unsafe {
            self.runtime.object_dec_ref(self.obj);
        }
    }
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

fn check_last_contiguous_2d(name: &str, stride_col: i64) -> Result<(), FlashInferError> {
    if stride_col != 1 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} last-dimension stride must be 1"
        )));
    }
    Ok(())
}

fn check_last_contiguous_3d(name: &str, stride2: i64) -> Result<(), FlashInferError> {
    if stride2 != 1 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} last-dimension stride must be 1"
        )));
    }
    Ok(())
}

#[cfg(feature = "cudarc")]
#[derive(Debug, Clone, Copy)]
pub struct FusedMoeCudarcOptions {
    pub activation: FusedMoeActivationType,
    pub enable_pdl: bool,
    pub tp_size: i64,
    pub tp_rank: i64,
    pub ep_size: i64,
    pub ep_rank: i64,
    pub enable_alltoall: bool,
    pub profile_ids: Option<[i64; 2]>,
}

#[cfg(feature = "cudarc")]
impl Default for FusedMoeCudarcOptions {
    fn default() -> Self {
        Self {
            activation: FusedMoeActivationType::Swiglu,
            enable_pdl: false,
            tp_size: 1,
            tp_rank: 0,
            ep_size: 1,
            ep_rank: 0,
            enable_alltoall: false,
            profile_ids: None,
        }
    }
}

#[cfg(feature = "cudarc")]
pub fn fused_moe_cudarc<T, I, S, W1, W2, O>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    token_selected_experts: &S,
    fc1_expert_weights: &W1,
    fc2_expert_weights: &W2,
    out: &mut O,
    num_tokens: usize,
    num_experts_on_rank: usize,
    top_k: usize,
    hidden_size: usize,
    inter_size: usize,
    dtype: DType,
    backend: FusedMoeBackend,
    options: FusedMoeCudarcOptions,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    S: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    W1: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    W2: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
{
    if dtype == DType::F8E4M3FN {
        return Err(FlashInferError::invalid_argument(
            "fused_moe_cudarc does not support fp8 quantization; use fused_moe_cudarc_fp8_per_tensor or fused_moe_cudarc_deepseek_fp8_block_scale",
        ));
    }

    let fc1_inter_size = if options.activation.is_gated() {
        inter_size.checked_mul(2).ok_or_else(|| {
            FlashInferError::invalid_argument("inter_size * 2 overflow for gated activation")
        })?
    } else {
        inter_size
    };

    let input_expected = num_tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| FlashInferError::invalid_argument("num_tokens * hidden_size overflow"))?;
    let out_expected = input_expected;
    let topk_expected = num_tokens
        .checked_mul(top_k)
        .ok_or_else(|| FlashInferError::invalid_argument("num_tokens * top_k overflow"))?;
    let fc1_expected = num_experts_on_rank
        .checked_mul(fc1_inter_size)
        .and_then(|v| v.checked_mul(hidden_size))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "num_experts_on_rank * fc1_inter_size * hidden_size overflow",
            )
        })?;
    let fc2_expected = num_experts_on_rank
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(inter_size))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "num_experts_on_rank * hidden_size * inter_size overflow",
            )
        })?;

    if input.len() != input_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "input length ({}) must equal num_tokens * hidden_size ({input_expected})",
            input.len()
        )));
    }
    if out.len() != out_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal num_tokens * hidden_size ({out_expected})",
            out.len()
        )));
    }
    if token_selected_experts.len() != topk_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "token_selected_experts length ({}) must equal num_tokens * top_k ({topk_expected})",
            token_selected_experts.len()
        )));
    }
    if fc1_expert_weights.len() != fc1_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "fc1_expert_weights length ({}) must equal expected ({fc1_expected})",
            fc1_expert_weights.len()
        )));
    }
    if fc2_expert_weights.len() != fc2_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "fc2_expert_weights length ({}) must equal expected ({fc2_expected})",
            fc2_expert_weights.len()
        )));
    }

    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (token_selected_experts_ptr, _token_selected_experts_sync) =
        token_selected_experts.device_ptr(stream);
    let (fc1_expert_weights_ptr, _fc1_expert_weights_sync) = fc1_expert_weights.device_ptr(stream);
    let (fc2_expert_weights_ptr, _fc2_expert_weights_sync) = fc2_expert_weights.device_ptr(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);

    let num_tokens_i64 = i64::try_from(num_tokens)
        .map_err(|_| FlashInferError::invalid_argument("num_tokens does not fit in i64"))?;
    let num_experts_on_rank_i64 = i64::try_from(num_experts_on_rank).map_err(|_| {
        FlashInferError::invalid_argument("num_experts_on_rank does not fit in i64")
    })?;
    let top_k_i64 = i64::try_from(top_k)
        .map_err(|_| FlashInferError::invalid_argument("top_k does not fit in i64"))?;
    let hidden_size_i64 = i64::try_from(hidden_size)
        .map_err(|_| FlashInferError::invalid_argument("hidden_size does not fit in i64"))?;
    let inter_size_i64 = i64::try_from(inter_size)
        .map_err(|_| FlashInferError::invalid_argument("inter_size does not fit in i64"))?;
    let fc1_inter_size_i64 = i64::try_from(fc1_inter_size)
        .map_err(|_| FlashInferError::invalid_argument("fc1_inter_size does not fit in i64"))?;

    let fc1_stride0 = fc1_inter_size_i64
        .checked_mul(hidden_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("fc1 stride overflow"))?;
    let fc2_stride0 = hidden_size_i64
        .checked_mul(inter_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("fc2 stride overflow"))?;

    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let params = FusedMoeParams::new(
        FusedMoeTensor2DDesc {
            ptr: out_ptr as usize as *const c_void,
            rows: num_tokens_i64,
            cols: hidden_size_i64,
            stride_row: hidden_size_i64,
            stride_col: 1,
            dtype,
            device_id,
        },
        FusedMoeTensor2DDesc {
            ptr: input_ptr as usize as *const c_void,
            rows: num_tokens_i64,
            cols: hidden_size_i64,
            stride_row: hidden_size_i64,
            stride_col: 1,
            dtype,
            device_id,
        },
        FusedMoeTensor2DI32Desc {
            ptr: token_selected_experts_ptr as usize as *const c_void,
            rows: num_tokens_i64,
            cols: top_k_i64,
            stride_row: top_k_i64,
            stride_col: 1,
            device_id,
        },
        FusedMoeTensor3DDesc {
            ptr: fc1_expert_weights_ptr as usize as *const c_void,
            dim0: num_experts_on_rank_i64,
            dim1: fc1_inter_size_i64,
            dim2: hidden_size_i64,
            stride0: fc1_stride0,
            stride1: hidden_size_i64,
            stride2: 1,
            dtype,
            device_id,
        },
        FusedMoeTensor3DDesc {
            ptr: fc2_expert_weights_ptr as usize as *const c_void,
            dim0: num_experts_on_rank_i64,
            dim1: hidden_size_i64,
            dim2: inter_size_i64,
            stride0: fc2_stride0,
            stride1: inter_size_i64,
            stride2: 1,
            dtype,
            device_id,
        },
        backend,
        stream.cu_stream().cast(),
    )
    .with_activation(options.activation)
    .with_enable_pdl(options.enable_pdl)
    .with_tensor_parallel(options.tp_size, options.tp_rank)
    .with_expert_parallel(options.ep_size, options.ep_rank)
    .with_enable_alltoall(options.enable_alltoall);

    let params = if let Some([gemm1_profile_id, gemm2_profile_id]) = options.profile_ids {
        params.with_profile_ids(gemm1_profile_id, gemm2_profile_id)
    } else {
        params
    };

    fused_moe(&params)
}

#[cfg(feature = "cudarc")]
pub fn fused_moe_cudarc_fp8_per_tensor<I, S, W1, W2, O, Q1, Q2, Q3, Q4>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    token_selected_experts: &S,
    fc1_expert_weights: &W1,
    fc2_expert_weights: &W2,
    out: &mut O,
    fc1_dequant: &Q1,
    fc2_quant: &Q2,
    fc2_dequant: &Q3,
    fc1_input_dequant: &Q4,
    num_tokens: usize,
    num_experts_on_rank: usize,
    top_k: usize,
    hidden_size: usize,
    inter_size: usize,
    output_dtype: DType,
    backend: FusedMoeBackend,
    options: FusedMoeCudarcOptions,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    S: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    W1: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    W2: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    O: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtrMut<u16>,
    Q1: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    Q2: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    Q3: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    Q4: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
{
    if output_dtype != DType::F16 && output_dtype != DType::BF16 {
        return Err(FlashInferError::invalid_argument(
            "fp8 per-tensor fused_moe requires output_dtype to be F16 or BF16",
        ));
    }

    let fc1_inter_size = if options.activation.is_gated() {
        inter_size.checked_mul(2).ok_or_else(|| {
            FlashInferError::invalid_argument("inter_size * 2 overflow for gated activation")
        })?
    } else {
        inter_size
    };

    let input_expected = num_tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| FlashInferError::invalid_argument("num_tokens * hidden_size overflow"))?;
    let out_expected = input_expected;
    let topk_expected = num_tokens
        .checked_mul(top_k)
        .ok_or_else(|| FlashInferError::invalid_argument("num_tokens * top_k overflow"))?;
    let fc1_expected = num_experts_on_rank
        .checked_mul(fc1_inter_size)
        .and_then(|v| v.checked_mul(hidden_size))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "num_experts_on_rank * fc1_inter_size * hidden_size overflow",
            )
        })?;
    let fc2_expected = num_experts_on_rank
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(inter_size))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "num_experts_on_rank * hidden_size * inter_size overflow",
            )
        })?;

    if input.len() != input_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "input length ({}) must equal num_tokens * hidden_size ({input_expected})",
            input.len()
        )));
    }
    if out.len() != out_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal num_tokens * hidden_size ({out_expected})",
            out.len()
        )));
    }
    if token_selected_experts.len() != topk_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "token_selected_experts length ({}) must equal num_tokens * top_k ({topk_expected})",
            token_selected_experts.len()
        )));
    }
    if fc1_expert_weights.len() != fc1_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "fc1_expert_weights length ({}) must equal expected ({fc1_expected})",
            fc1_expert_weights.len()
        )));
    }
    if fc2_expert_weights.len() != fc2_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "fc2_expert_weights length ({}) must equal expected ({fc2_expected})",
            fc2_expert_weights.len()
        )));
    }
    if fc1_dequant.len() != num_experts_on_rank {
        return Err(FlashInferError::invalid_argument(format!(
            "fc1_dequant length ({}) must equal num_experts_on_rank ({num_experts_on_rank})",
            fc1_dequant.len()
        )));
    }
    if fc2_dequant.len() != num_experts_on_rank {
        return Err(FlashInferError::invalid_argument(format!(
            "fc2_dequant length ({}) must equal num_experts_on_rank ({num_experts_on_rank})",
            fc2_dequant.len()
        )));
    }
    if fc2_quant.len() != 1 && fc2_quant.len() != num_experts_on_rank {
        return Err(FlashInferError::invalid_argument(format!(
            "fc2_quant length ({}) must be 1 or num_experts_on_rank ({num_experts_on_rank})",
            fc2_quant.len()
        )));
    }
    if fc1_input_dequant.len() != 1 {
        return Err(FlashInferError::invalid_argument(format!(
            "fc1_input_dequant length ({}) must be 1",
            fc1_input_dequant.len()
        )));
    }

    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (token_selected_experts_ptr, _token_selected_experts_sync) =
        token_selected_experts.device_ptr(stream);
    let (fc1_expert_weights_ptr, _fc1_expert_weights_sync) = fc1_expert_weights.device_ptr(stream);
    let (fc2_expert_weights_ptr, _fc2_expert_weights_sync) = fc2_expert_weights.device_ptr(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);
    let (fc1_dequant_ptr, _fc1_dequant_sync) = fc1_dequant.device_ptr(stream);
    let (fc2_quant_ptr, _fc2_quant_sync) = fc2_quant.device_ptr(stream);
    let (fc2_dequant_ptr, _fc2_dequant_sync) = fc2_dequant.device_ptr(stream);
    let (fc1_input_dequant_ptr, _fc1_input_dequant_sync) = fc1_input_dequant.device_ptr(stream);

    let num_tokens_i64 = i64::try_from(num_tokens)
        .map_err(|_| FlashInferError::invalid_argument("num_tokens does not fit in i64"))?;
    let num_experts_on_rank_i64 = i64::try_from(num_experts_on_rank).map_err(|_| {
        FlashInferError::invalid_argument("num_experts_on_rank does not fit in i64")
    })?;
    let top_k_i64 = i64::try_from(top_k)
        .map_err(|_| FlashInferError::invalid_argument("top_k does not fit in i64"))?;
    let hidden_size_i64 = i64::try_from(hidden_size)
        .map_err(|_| FlashInferError::invalid_argument("hidden_size does not fit in i64"))?;
    let inter_size_i64 = i64::try_from(inter_size)
        .map_err(|_| FlashInferError::invalid_argument("inter_size does not fit in i64"))?;
    let fc1_inter_size_i64 = i64::try_from(fc1_inter_size)
        .map_err(|_| FlashInferError::invalid_argument("fc1_inter_size does not fit in i64"))?;

    let fc1_stride0 = fc1_inter_size_i64
        .checked_mul(hidden_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("fc1 stride overflow"))?;
    let fc2_stride0 = hidden_size_i64
        .checked_mul(inter_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("fc2 stride overflow"))?;

    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let fc2_quant_desc = if fc2_quant.len() == 1 {
        FusedMoeFp8ActScaleDesc::Scalar(FusedMoeTensor0DF32Desc {
            ptr: fc2_quant_ptr as usize as *const c_void,
            device_id,
        })
    } else {
        FusedMoeFp8ActScaleDesc::PerExpert(FusedMoeTensor1DF32Desc {
            ptr: fc2_quant_ptr as usize as *const c_void,
            len: num_experts_on_rank_i64,
            stride: 1,
            device_id,
        })
    };

    let quantization = FusedMoeFp8PerTensorQuantParams {
        fc1_dequant: FusedMoeTensor1DF32Desc {
            ptr: fc1_dequant_ptr as usize as *const c_void,
            len: num_experts_on_rank_i64,
            stride: 1,
            device_id,
        },
        fc2_quant: fc2_quant_desc,
        fc2_dequant: FusedMoeTensor1DF32Desc {
            ptr: fc2_dequant_ptr as usize as *const c_void,
            len: num_experts_on_rank_i64,
            stride: 1,
            device_id,
        },
        fc1_input_dequant: FusedMoeTensor0DF32Desc {
            ptr: fc1_input_dequant_ptr as usize as *const c_void,
            device_id,
        },
    };

    let params = FusedMoeParams::new(
        FusedMoeTensor2DDesc {
            ptr: out_ptr as usize as *const c_void,
            rows: num_tokens_i64,
            cols: hidden_size_i64,
            stride_row: hidden_size_i64,
            stride_col: 1,
            dtype: output_dtype,
            device_id,
        },
        FusedMoeTensor2DDesc {
            ptr: input_ptr as usize as *const c_void,
            rows: num_tokens_i64,
            cols: hidden_size_i64,
            stride_row: hidden_size_i64,
            stride_col: 1,
            dtype: DType::F8E4M3FN,
            device_id,
        },
        FusedMoeTensor2DI32Desc {
            ptr: token_selected_experts_ptr as usize as *const c_void,
            rows: num_tokens_i64,
            cols: top_k_i64,
            stride_row: top_k_i64,
            stride_col: 1,
            device_id,
        },
        FusedMoeTensor3DDesc {
            ptr: fc1_expert_weights_ptr as usize as *const c_void,
            dim0: num_experts_on_rank_i64,
            dim1: fc1_inter_size_i64,
            dim2: hidden_size_i64,
            stride0: fc1_stride0,
            stride1: hidden_size_i64,
            stride2: 1,
            dtype: DType::F8E4M3FN,
            device_id,
        },
        FusedMoeTensor3DDesc {
            ptr: fc2_expert_weights_ptr as usize as *const c_void,
            dim0: num_experts_on_rank_i64,
            dim1: hidden_size_i64,
            dim2: inter_size_i64,
            stride0: fc2_stride0,
            stride1: inter_size_i64,
            stride2: 1,
            dtype: DType::F8E4M3FN,
            device_id,
        },
        backend,
        stream.cu_stream().cast(),
    )
    .with_activation(options.activation)
    .with_enable_pdl(options.enable_pdl)
    .with_tensor_parallel(options.tp_size, options.tp_rank)
    .with_expert_parallel(options.ep_size, options.ep_rank)
    .with_enable_alltoall(options.enable_alltoall)
    .with_fp8_per_tensor_quantization(quantization);

    let params = if let Some([gemm1_profile_id, gemm2_profile_id]) = options.profile_ids {
        params.with_profile_ids(gemm1_profile_id, gemm2_profile_id)
    } else {
        params
    };

    fused_moe(&params)
}

#[cfg(feature = "cudarc")]
pub fn fused_moe_cudarc_deepseek_fp8_block_scale<I, S, W1, W2, O, Q1, Q2>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    token_selected_experts: &S,
    fc1_expert_weights: &W1,
    fc2_expert_weights: &W2,
    out: &mut O,
    fc1_scales: &Q1,
    fc2_scales: &Q2,
    num_tokens: usize,
    num_experts_on_rank: usize,
    top_k: usize,
    hidden_size: usize,
    inter_size: usize,
    backend: FusedMoeBackend,
    options: FusedMoeCudarcOptions,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtr<u16>,
    S: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    W1: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    W2: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    O: cudarc::driver::DeviceSlice<u16> + cudarc::driver::DevicePtrMut<u16>,
    Q1: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    Q2: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
{
    let fc1_inter_size = if options.activation.is_gated() {
        inter_size.checked_mul(2).ok_or_else(|| {
            FlashInferError::invalid_argument("inter_size * 2 overflow for gated activation")
        })?
    } else {
        inter_size
    };

    if hidden_size % 128 != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "hidden_size ({hidden_size}) must be divisible by 128 for deepseek fp8 block-scale",
        )));
    }
    if inter_size % 128 != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "inter_size ({inter_size}) must be divisible by 128 for deepseek fp8 block-scale",
        )));
    }
    if fc1_inter_size % 128 != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "fc1_inter_size ({fc1_inter_size}) must be divisible by 128 for deepseek fp8 block-scale",
        )));
    }

    let input_expected = num_tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| FlashInferError::invalid_argument("num_tokens * hidden_size overflow"))?;
    let out_expected = input_expected;
    let topk_expected = num_tokens
        .checked_mul(top_k)
        .ok_or_else(|| FlashInferError::invalid_argument("num_tokens * top_k overflow"))?;
    let fc1_expected = num_experts_on_rank
        .checked_mul(fc1_inter_size)
        .and_then(|v| v.checked_mul(hidden_size))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "num_experts_on_rank * fc1_inter_size * hidden_size overflow",
            )
        })?;
    let fc2_expected = num_experts_on_rank
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(inter_size))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "num_experts_on_rank * hidden_size * inter_size overflow",
            )
        })?;

    let fc1_scales_expected = num_experts_on_rank
        .checked_mul(fc1_inter_size / 128)
        .and_then(|v| v.checked_mul(hidden_size / 128))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "num_experts_on_rank * (fc1_inter_size/128) * (hidden_size/128) overflow",
            )
        })?;
    let fc2_scales_expected = num_experts_on_rank
        .checked_mul(hidden_size / 128)
        .and_then(|v| v.checked_mul(inter_size / 128))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(
                "num_experts_on_rank * (hidden_size/128) * (inter_size/128) overflow",
            )
        })?;

    if input.len() != input_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "input length ({}) must equal num_tokens * hidden_size ({input_expected})",
            input.len()
        )));
    }
    if out.len() != out_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal num_tokens * hidden_size ({out_expected})",
            out.len()
        )));
    }
    if token_selected_experts.len() != topk_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "token_selected_experts length ({}) must equal num_tokens * top_k ({topk_expected})",
            token_selected_experts.len()
        )));
    }
    if fc1_expert_weights.len() != fc1_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "fc1_expert_weights length ({}) must equal expected ({fc1_expected})",
            fc1_expert_weights.len()
        )));
    }
    if fc2_expert_weights.len() != fc2_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "fc2_expert_weights length ({}) must equal expected ({fc2_expected})",
            fc2_expert_weights.len()
        )));
    }
    if fc1_scales.len() != fc1_scales_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "fc1_scales length ({}) must equal expected ({fc1_scales_expected})",
            fc1_scales.len()
        )));
    }
    if fc2_scales.len() != fc2_scales_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "fc2_scales length ({}) must equal expected ({fc2_scales_expected})",
            fc2_scales.len()
        )));
    }

    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (token_selected_experts_ptr, _token_selected_experts_sync) =
        token_selected_experts.device_ptr(stream);
    let (fc1_expert_weights_ptr, _fc1_expert_weights_sync) = fc1_expert_weights.device_ptr(stream);
    let (fc2_expert_weights_ptr, _fc2_expert_weights_sync) = fc2_expert_weights.device_ptr(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);
    let (fc1_scales_ptr, _fc1_scales_sync) = fc1_scales.device_ptr(stream);
    let (fc2_scales_ptr, _fc2_scales_sync) = fc2_scales.device_ptr(stream);

    let num_tokens_i64 = i64::try_from(num_tokens)
        .map_err(|_| FlashInferError::invalid_argument("num_tokens does not fit in i64"))?;
    let num_experts_on_rank_i64 = i64::try_from(num_experts_on_rank).map_err(|_| {
        FlashInferError::invalid_argument("num_experts_on_rank does not fit in i64")
    })?;
    let top_k_i64 = i64::try_from(top_k)
        .map_err(|_| FlashInferError::invalid_argument("top_k does not fit in i64"))?;
    let hidden_size_i64 = i64::try_from(hidden_size)
        .map_err(|_| FlashInferError::invalid_argument("hidden_size does not fit in i64"))?;
    let inter_size_i64 = i64::try_from(inter_size)
        .map_err(|_| FlashInferError::invalid_argument("inter_size does not fit in i64"))?;
    let fc1_inter_size_i64 = i64::try_from(fc1_inter_size)
        .map_err(|_| FlashInferError::invalid_argument("fc1_inter_size does not fit in i64"))?;

    let fc1_stride0 = fc1_inter_size_i64
        .checked_mul(hidden_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("fc1 stride overflow"))?;
    let fc2_stride0 = hidden_size_i64
        .checked_mul(inter_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("fc2 stride overflow"))?;

    let fc1_scale_dim1 = fc1_inter_size_i64 / 128;
    let fc1_scale_dim2 = hidden_size_i64 / 128;
    let fc2_scale_dim1 = hidden_size_i64 / 128;
    let fc2_scale_dim2 = inter_size_i64 / 128;

    let fc1_scale_stride2 = 1_i64;
    let fc1_scale_stride1 = fc1_scale_dim2;
    let fc1_scale_stride0 = fc1_scale_dim1
        .checked_mul(fc1_scale_dim2)
        .ok_or_else(|| FlashInferError::invalid_argument("fc1 scale stride overflow"))?;
    let fc2_scale_stride2 = 1_i64;
    let fc2_scale_stride1 = fc2_scale_dim2;
    let fc2_scale_stride0 = fc2_scale_dim1
        .checked_mul(fc2_scale_dim2)
        .ok_or_else(|| FlashInferError::invalid_argument("fc2 scale stride overflow"))?;

    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let quantization = FusedMoeDeepSeekFp8BlockScaleQuantParams {
        fc1_scales: FusedMoeTensor3DF32Desc {
            ptr: fc1_scales_ptr as usize as *const c_void,
            dim0: num_experts_on_rank_i64,
            dim1: fc1_scale_dim1,
            dim2: fc1_scale_dim2,
            stride0: fc1_scale_stride0,
            stride1: fc1_scale_stride1,
            stride2: fc1_scale_stride2,
            device_id,
        },
        fc2_scales: FusedMoeTensor3DF32Desc {
            ptr: fc2_scales_ptr as usize as *const c_void,
            dim0: num_experts_on_rank_i64,
            dim1: fc2_scale_dim1,
            dim2: fc2_scale_dim2,
            stride0: fc2_scale_stride0,
            stride1: fc2_scale_stride1,
            stride2: fc2_scale_stride2,
            device_id,
        },
    };

    let params = FusedMoeParams::new(
        FusedMoeTensor2DDesc {
            ptr: out_ptr as usize as *const c_void,
            rows: num_tokens_i64,
            cols: hidden_size_i64,
            stride_row: hidden_size_i64,
            stride_col: 1,
            dtype: DType::BF16,
            device_id,
        },
        FusedMoeTensor2DDesc {
            ptr: input_ptr as usize as *const c_void,
            rows: num_tokens_i64,
            cols: hidden_size_i64,
            stride_row: hidden_size_i64,
            stride_col: 1,
            dtype: DType::BF16,
            device_id,
        },
        FusedMoeTensor2DI32Desc {
            ptr: token_selected_experts_ptr as usize as *const c_void,
            rows: num_tokens_i64,
            cols: top_k_i64,
            stride_row: top_k_i64,
            stride_col: 1,
            device_id,
        },
        FusedMoeTensor3DDesc {
            ptr: fc1_expert_weights_ptr as usize as *const c_void,
            dim0: num_experts_on_rank_i64,
            dim1: fc1_inter_size_i64,
            dim2: hidden_size_i64,
            stride0: fc1_stride0,
            stride1: hidden_size_i64,
            stride2: 1,
            dtype: DType::F8E4M3FN,
            device_id,
        },
        FusedMoeTensor3DDesc {
            ptr: fc2_expert_weights_ptr as usize as *const c_void,
            dim0: num_experts_on_rank_i64,
            dim1: hidden_size_i64,
            dim2: inter_size_i64,
            stride0: fc2_stride0,
            stride1: inter_size_i64,
            stride2: 1,
            dtype: DType::F8E4M3FN,
            device_id,
        },
        backend,
        stream.cu_stream().cast(),
    )
    .with_activation(options.activation)
    .with_enable_pdl(options.enable_pdl)
    .with_tensor_parallel(options.tp_size, options.tp_rank)
    .with_expert_parallel(options.ep_size, options.ep_rank)
    .with_enable_alltoall(options.enable_alltoall)
    .with_deepseek_fp8_block_scale_quantization(quantization);

    let params = if let Some([gemm1_profile_id, gemm2_profile_id]) = options.profile_ids {
        params.with_profile_ids(gemm1_profile_id, gemm2_profile_id)
    } else {
        params
    };

    fused_moe(&params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ptr() -> *const c_void {
        std::ptr::NonNull::<u8>::dangling()
            .as_ptr()
            .cast::<c_void>()
    }

    fn valid_params() -> FusedMoeParams {
        let ptr = dummy_ptr();
        FusedMoeParams::new(
            FusedMoeTensor2DDesc {
                ptr,
                rows: 4,
                cols: 128,
                stride_row: 128,
                stride_col: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            FusedMoeTensor2DDesc {
                ptr,
                rows: 4,
                cols: 128,
                stride_row: 128,
                stride_col: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            FusedMoeTensor2DI32Desc {
                ptr,
                rows: 4,
                cols: 2,
                stride_row: 2,
                stride_col: 1,
                device_id: 0,
            },
            FusedMoeTensor3DDesc {
                ptr,
                dim0: 8,
                dim1: 512,
                dim2: 128,
                stride0: 512 * 128,
                stride1: 128,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            FusedMoeTensor3DDesc {
                ptr,
                dim0: 8,
                dim1: 128,
                dim2: 256,
                stride0: 128 * 256,
                stride1: 256,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            FusedMoeBackend::Sm120,
            std::ptr::null_mut(),
        )
        .with_activation(FusedMoeActivationType::Swiglu)
    }

    fn valid_fp8_per_tensor_params() -> FusedMoeParams {
        let ptr = dummy_ptr();
        valid_params()
            .with_fp8_per_tensor_quantization(FusedMoeFp8PerTensorQuantParams {
                fc1_dequant: FusedMoeTensor1DF32Desc {
                    ptr,
                    len: 8,
                    stride: 1,
                    device_id: 0,
                },
                fc2_quant: FusedMoeFp8ActScaleDesc::PerExpert(FusedMoeTensor1DF32Desc {
                    ptr,
                    len: 8,
                    stride: 1,
                    device_id: 0,
                }),
                fc2_dequant: FusedMoeTensor1DF32Desc {
                    ptr,
                    len: 8,
                    stride: 1,
                    device_id: 0,
                },
                fc1_input_dequant: FusedMoeTensor0DF32Desc { ptr, device_id: 0 },
            })
            .with_activation(FusedMoeActivationType::Swiglu)
            .with_enable_pdl(false)
            .with_enable_alltoall(false)
            .with_tensor_parallel(1, 0)
            .with_expert_parallel(1, 0)
            .with_profile_ids(0, 0)
    }

    fn valid_deepseek_fp8_block_scale_params() -> FusedMoeParams {
        let ptr = dummy_ptr();
        FusedMoeParams::new(
            FusedMoeTensor2DDesc {
                ptr,
                rows: 4,
                cols: 128,
                stride_row: 128,
                stride_col: 1,
                dtype: DType::BF16,
                device_id: 0,
            },
            FusedMoeTensor2DDesc {
                ptr,
                rows: 4,
                cols: 128,
                stride_row: 128,
                stride_col: 1,
                dtype: DType::BF16,
                device_id: 0,
            },
            FusedMoeTensor2DI32Desc {
                ptr,
                rows: 4,
                cols: 2,
                stride_row: 2,
                stride_col: 1,
                device_id: 0,
            },
            FusedMoeTensor3DDesc {
                ptr,
                dim0: 8,
                dim1: 512,
                dim2: 128,
                stride0: 512 * 128,
                stride1: 128,
                stride2: 1,
                dtype: DType::F8E4M3FN,
                device_id: 0,
            },
            FusedMoeTensor3DDesc {
                ptr,
                dim0: 8,
                dim1: 128,
                dim2: 256,
                stride0: 128 * 256,
                stride1: 256,
                stride2: 1,
                dtype: DType::F8E4M3FN,
                device_id: 0,
            },
            FusedMoeBackend::Sm90,
            std::ptr::null_mut(),
        )
        .with_activation(FusedMoeActivationType::Swiglu)
        .with_deepseek_fp8_block_scale_quantization(
            FusedMoeDeepSeekFp8BlockScaleQuantParams {
                fc1_scales: FusedMoeTensor3DF32Desc {
                    ptr,
                    dim0: 8,
                    dim1: 4,
                    dim2: 1,
                    stride0: 4,
                    stride1: 1,
                    stride2: 1,
                    device_id: 0,
                },
                fc2_scales: FusedMoeTensor3DF32Desc {
                    ptr,
                    dim0: 8,
                    dim1: 1,
                    dim2: 2,
                    stride0: 2,
                    stride1: 2,
                    stride2: 1,
                    device_id: 0,
                },
            },
        )
    }

    #[test]
    fn validate_accepts_base_case() {
        let params = valid_params();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn validate_rejects_dtype_mismatch() {
        let mut params = valid_params();
        params.out.dtype = DType::BF16;
        let err = params.validate().expect_err("expected dtype mismatch");
        assert!(err.to_string().contains("dtype mismatch"));
    }

    #[test]
    fn validate_rejects_gated_fc1_inter_mismatch() {
        let mut params = valid_params();
        params.fc1_expert_weights.dim1 = 255;
        let err = params.validate().expect_err("expected inter-size mismatch");
        assert!(err.to_string().contains("fc1_expert_weights.dim1"));
    }

    #[test]
    fn validate_rejects_device_mismatch() {
        let mut params = valid_params();
        params.fc2_expert_weights.device_id = 1;
        let err = params.validate().expect_err("expected device mismatch");
        assert!(err.to_string().contains("device mismatch"));
    }

    #[test]
    fn validate_rejects_fp8_without_quantization() {
        let mut params = valid_params();
        params.input.dtype = DType::F8E4M3FN;
        params.fc1_expert_weights.dtype = DType::F8E4M3FN;
        params.fc2_expert_weights.dtype = DType::F8E4M3FN;
        params.out.dtype = DType::BF16;
        let err = params
            .validate()
            .expect_err("expected fp8 quantization error");
        assert!(err.to_string().contains("requires explicit quantization"));
    }

    #[test]
    fn validate_accepts_fp8_per_tensor() {
        let mut params = valid_fp8_per_tensor_params();
        params.input.dtype = DType::F8E4M3FN;
        params.fc1_expert_weights.dtype = DType::F8E4M3FN;
        params.fc2_expert_weights.dtype = DType::F8E4M3FN;
        params.out.dtype = DType::BF16;
        assert!(params.validate().is_ok());
    }

    #[test]
    fn validate_rejects_fp8_per_tensor_output_dtype() {
        let mut params = valid_fp8_per_tensor_params();
        params.input.dtype = DType::F8E4M3FN;
        params.fc1_expert_weights.dtype = DType::F8E4M3FN;
        params.fc2_expert_weights.dtype = DType::F8E4M3FN;
        params.out.dtype = DType::F8E4M3FN;
        let err = params.validate().expect_err("expected output dtype error");
        assert!(err.to_string().contains("out dtype"));
    }

    #[test]
    fn validate_rejects_fp8_per_tensor_bad_fc2_quant_len() {
        let mut params = valid_fp8_per_tensor_params();
        params.input.dtype = DType::F8E4M3FN;
        params.fc1_expert_weights.dtype = DType::F8E4M3FN;
        params.fc2_expert_weights.dtype = DType::F8E4M3FN;
        params.out.dtype = DType::BF16;
        params.quantization = Some(FusedMoeQuantization::Fp8PerTensor(
            FusedMoeFp8PerTensorQuantParams {
                fc1_dequant: FusedMoeTensor1DF32Desc {
                    ptr: dummy_ptr(),
                    len: 8,
                    stride: 1,
                    device_id: 0,
                },
                fc2_quant: FusedMoeFp8ActScaleDesc::PerExpert(FusedMoeTensor1DF32Desc {
                    ptr: dummy_ptr(),
                    len: 7,
                    stride: 1,
                    device_id: 0,
                }),
                fc2_dequant: FusedMoeTensor1DF32Desc {
                    ptr: dummy_ptr(),
                    len: 8,
                    stride: 1,
                    device_id: 0,
                },
                fc1_input_dequant: FusedMoeTensor0DF32Desc {
                    ptr: dummy_ptr(),
                    device_id: 0,
                },
            },
        ));
        let err = params
            .validate()
            .expect_err("expected fc2_quant shape error");
        assert!(err.to_string().contains("quant.fc2_quant.len"));
    }

    #[test]
    fn validate_accepts_deepseek_fp8_block_scale() {
        let params = valid_deepseek_fp8_block_scale_params();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn validate_rejects_deepseek_non_sm90_backend() {
        let mut params = valid_deepseek_fp8_block_scale_params();
        params.backend = FusedMoeBackend::Sm100;
        let err = params.validate().expect_err("expected backend validation");
        assert!(err.to_string().contains("backend Sm90"));
    }

    #[test]
    fn validate_rejects_deepseek_dtype_mismatch() {
        let mut params = valid_deepseek_fp8_block_scale_params();
        params.input.dtype = DType::F16;
        let err = params.validate().expect_err("expected dtype validation");
        assert!(err.to_string().contains("input/out dtype"));
    }

    #[test]
    fn validate_rejects_deepseek_scale_shape_mismatch() {
        let mut params = valid_deepseek_fp8_block_scale_params();
        let Some(FusedMoeQuantization::DeepSeekFp8BlockScale(mut quant)) = params.quantization
        else {
            panic!("expected deepseek quantization")
        };
        quant.fc2_scales.dim2 = 3;
        params.quantization = Some(FusedMoeQuantization::DeepSeekFp8BlockScale(quant));
        let err = params
            .validate()
            .expect_err("expected deepseek scale mismatch");
        assert!(err.to_string().contains("quant.fc2_scales"));
    }

    #[test]
    fn with_profile_ids_sets_profile_ids() {
        let params = valid_params().with_profile_ids(7, 11);
        assert_eq!(params.profile_ids, Some([7, 11]));
    }
}
