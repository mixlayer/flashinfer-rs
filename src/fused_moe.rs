use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    any_bool, any_dltensor_ptr, any_dtype, any_i64, any_none, any_object_handle, DLDataType,
    DLDevice, DLTensor, TVMFFIAny, KDL_BFLOAT, KDL_CUDA, KDL_FLOAT, KDL_INT,
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
    let fc1_expert_weights_tensor = tensor_3d(
        params.fc1_expert_weights.ptr,
        params.fc1_expert_weights.dtype,
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
    let fc2_expert_weights_tensor = tensor_3d(
        params.fc2_expert_weights.ptr,
        params.fc2_expert_weights.dtype,
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
        any_dtype(dl_dtype_from_dtype(params.fc1_expert_weights.dtype)),
        any_dtype(dl_dtype_from_dtype(params.out.dtype)),
        any_bool(false), // use_deepseek_fp8_block_scale
        any_bool(false), // use_w4_group_scaling
        any_bool(false), // use_mxfp8_act_scaling
        any_bool(false), // use_packed_weights
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

        let mut run_args: [TVMFFIAny; 24] = [
            any_dltensor_ptr(&out_tensor),
            any_dltensor_ptr(&input_tensor),
            any_dltensor_ptr(&token_selected_experts_tensor),
            optional_dltensor_any(token_final_scales_tensor.as_ref()),
            any_dltensor_ptr(&fc1_expert_weights_tensor),
            optional_dltensor_any(fc1_expert_biases_tensor.as_ref()),
            any_dltensor_ptr(&fc2_expert_weights_tensor),
            optional_dltensor_any(fc2_expert_biases_tensor.as_ref()),
            any_none(), // quant_scales
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

    if params.fc1_expert_weights.dim2 != params.input.cols {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: fc1_expert_weights.dim2 ({}) must match input.cols ({})",
            params.fc1_expert_weights.dim2, params.input.cols
        )));
    }

    if params.fc2_expert_weights.dim1 != params.input.cols {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: fc2_expert_weights.dim1 ({}) must match input.cols ({})",
            params.fc2_expert_weights.dim1, params.input.cols
        )));
    }

    let expected_fc1_dim1 = if params.activation.is_gated() {
        params
            .fc2_expert_weights
            .dim2
            .checked_mul(2)
            .ok_or_else(|| {
                FlashInferError::invalid_argument("fc2_expert_weights.dim2 * 2 overflow")
            })?
    } else {
        params.fc2_expert_weights.dim2
    };

    if params.fc1_expert_weights.dim1 != expected_fc1_dim1 {
        return Err(FlashInferError::invalid_argument(format!(
            "shape mismatch: fc1_expert_weights.dim1 ({}) must equal expected inter size ({expected_fc1_dim1})",
            params.fc1_expert_weights.dim1
        )));
    }

    if params.input.dtype != params.fc1_expert_weights.dtype
        || params.input.dtype != params.fc2_expert_weights.dtype
        || params.input.dtype != params.out.dtype
    {
        return Err(FlashInferError::invalid_argument(
            "dtype mismatch: v1 fused_moe requires input/weights/out to share the same dtype",
        ));
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
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_dtype(dtype),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_params() -> FusedMoeParams {
        let ptr = std::ptr::NonNull::<u8>::dangling()
            .as_ptr()
            .cast::<c_void>();

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
    fn with_profile_ids_sets_profile_ids() {
        let params = valid_params().with_profile_ids(7, 11);
        assert_eq!(params.profile_ids, Some([7, 11]));
    }
}
