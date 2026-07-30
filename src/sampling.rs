//! Bindings for the fixed `sampling/sampling.so` module in the pinned JIT-cache wheel.
//!
//! The shape and argument contracts below follow
//! `flashinfer/csrc/flashinfer_sampling_binding.cu`,
//! `flashinfer/csrc/sampling.cu`, `flashinfer/csrc/renorm.cu`, and
//! `flashinfer/flashinfer/sampling.py`.

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_CUDA, KDL_FLOAT, KDL_INT, KDL_UINT, TVMFFIAny, any_bool,
    any_dltensor_ptr, any_f64, any_i64, any_none, any_object_handle,
};
use crate::runtime::{FlashInferRuntime, SamplingKernel};

#[cfg(feature = "cudarc")]
use cudarc::driver::DevicePtr;

/// Workspace size used by the upstream Python sampling wrapper.
pub const SAMPLING_WORKSPACE_BYTES: usize = 1024 * 1024;

/// Caller-owned FP32 CUDA tensor with shape `[rows, cols]`.
#[derive(Debug, Clone, Copy)]
pub struct SamplingTensor2DF32Desc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub device_id: i32,
}

/// Caller-owned FP32 CUDA tensor with shape `[len]`.
#[derive(Debug, Clone, Copy)]
pub struct SamplingTensor1DF32Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

/// Caller-owned I32 CUDA tensor with shape `[len]`.
#[derive(Debug, Clone, Copy)]
pub struct SamplingTensor1DI32Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

/// Caller-owned U64 CUDA tensor with shape `[len]`.
///
/// FlashInfer 0.6.4 accepts this device-resident RNG state as either a
/// one-element array or an `[output_batch]` array. The pinned kernels currently
/// consume element zero; accepting the batch-sized form preserves upstream ABI
/// compatibility and CUDA Graph update semantics.
#[derive(Debug, Clone, Copy)]
pub struct SamplingTensor1DU64Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

/// Caller-owned contiguous byte workspace on CUDA.
#[derive(Debug, Clone, Copy)]
pub struct SamplingWorkspaceDesc {
    pub ptr: *const c_void,
    pub len_bytes: i64,
    pub device_id: i32,
}

/// Random-number inputs shared by the sampling kernels.
#[derive(Debug, Clone, Copy)]
pub struct SamplingRandomParams {
    /// Whether to select the deterministic FlashInfer kernel path.
    pub deterministic: bool,
    /// Scalar Philox seed used when `seed_arr` is `None`.
    pub seed: u64,
    /// Scalar Philox offset used when `offset_arr` is `None`.
    pub offset: u64,
    /// Optional caller-owned U64 device seed, contiguous `[1]` or
    /// `[output_batch]`.
    pub seed_arr: Option<SamplingTensor1DU64Desc>,
    /// Optional caller-owned U64 device offset, contiguous `[1]` or
    /// `[output_batch]`.
    ///
    /// Seed and offset arrays must either both be present or both be absent.
    pub offset_arr: Option<SamplingTensor1DU64Desc>,
}

impl SamplingRandomParams {
    pub fn new(seed: u64, offset: u64) -> Self {
        Self {
            deterministic: true,
            seed,
            offset,
            seed_arr: None,
            offset_arr: None,
        }
    }

    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    pub fn with_arrays(
        mut self,
        seed_arr: SamplingTensor1DU64Desc,
        offset_arr: SamplingTensor1DU64Desc,
    ) -> Self {
        self.seed_arr = Some(seed_arr);
        self.offset_arr = Some(offset_arr);
        self
    }
}

/// Parameters for sampling I32 token IDs from FP32 probabilities or logits.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    /// FP32 probabilities or logits, contiguous rank-2 `[input_batch, vocab_size]`.
    pub input: SamplingTensor2DF32Desc,
    /// Caller-owned I32 output, contiguous rank-1 `[output_batch]`.
    pub output: SamplingTensor1DI32Desc,
    /// Optional I32 row mapping, contiguous rank-1 `[output_batch]`.
    ///
    /// `row_indices[i]` selects the input row used to produce `output[i]`. Values
    /// must be in `0..input_batch`; device values cannot be checked by Rust.
    pub row_indices: Option<SamplingTensor1DI32Desc>,
    /// Philox seed/offset and deterministic-mode settings.
    pub random: SamplingRandomParams,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    ///
    /// A Candle stream pointer can be passed directly. The previous TVM-FFI
    /// stream is restored before this function returns.
    pub stream: *mut c_void,
}

impl SamplingParams {
    pub fn new(
        input: SamplingTensor2DF32Desc,
        output: SamplingTensor1DI32Desc,
        random: SamplingRandomParams,
        stream: *mut c_void,
    ) -> Self {
        Self {
            input,
            output,
            row_indices: None,
            random,
            stream,
        }
    }

    pub fn with_row_indices(mut self, row_indices: SamplingTensor1DI32Desc) -> Self {
        self.row_indices = Some(row_indices);
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_sampling_base(self.input, self.output, self.row_indices, self.random)
    }
}

pub type SamplingFromProbsParams = SamplingParams;
pub type SamplingFromLogitsParams = SamplingParams;

/// Parameters for top-p sampling from FP32 probabilities.
#[derive(Debug, Clone, Copy)]
pub struct TopPSamplingParams {
    /// Common FP32 input, caller-owned I32 output, row mapping, RNG, and stream.
    pub sampling: SamplingParams,
    /// Scalar top-p threshold in `(0, 1]`, used when `top_p_arr` is `None`.
    pub top_p: f64,
    /// Optional FP32 top-p thresholds, contiguous `[input_batch]`.
    pub top_p_arr: Option<SamplingTensor1DF32Desc>,
}

impl TopPSamplingParams {
    pub fn new(sampling: SamplingParams, top_p: f64) -> Self {
        Self {
            sampling,
            top_p,
            top_p_arr: None,
        }
    }

    pub fn with_top_p_arr(mut self, top_p_arr: SamplingTensor1DF32Desc) -> Self {
        self.top_p_arr = Some(top_p_arr);
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        self.sampling.validate()?;
        validate_probability_threshold("top_p", self.top_p)?;
        validate_f32_param_array("top_p_arr", self.top_p_arr, self.sampling.input)
    }
}

/// Parameters for top-k sampling from FP32 probabilities.
#[derive(Debug, Clone, Copy)]
pub struct TopKSamplingParams {
    /// Common FP32 input, caller-owned I32 output, row mapping, RNG, and stream.
    pub sampling: SamplingParams,
    /// Scalar number of vocabulary entries retained, used when `top_k_arr` is `None`.
    pub top_k: i64,
    /// Optional I32 top-k values, contiguous `[input_batch]`.
    pub top_k_arr: Option<SamplingTensor1DI32Desc>,
}

impl TopKSamplingParams {
    pub fn new(sampling: SamplingParams, top_k: i64) -> Self {
        Self {
            sampling,
            top_k,
            top_k_arr: None,
        }
    }

    pub fn with_top_k_arr(mut self, top_k_arr: SamplingTensor1DI32Desc) -> Self {
        self.top_k_arr = Some(top_k_arr);
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        self.sampling.validate()?;
        validate_top_k(self.top_k, self.sampling.input.cols)?;
        validate_i32_param_array("top_k_arr", self.top_k_arr, self.sampling.input)
    }
}

/// Parameters for min-p sampling from FP32 probabilities.
#[derive(Debug, Clone, Copy)]
pub struct MinPSamplingParams {
    /// Common FP32 input, caller-owned I32 output, row mapping, RNG, and stream.
    pub sampling: SamplingParams,
    /// Scalar min-p threshold in `[0, 1]`, used when `min_p_arr` is `None`.
    pub min_p: f64,
    /// Optional FP32 min-p thresholds, contiguous `[input_batch]`.
    pub min_p_arr: Option<SamplingTensor1DF32Desc>,
}

impl MinPSamplingParams {
    pub fn new(sampling: SamplingParams, min_p: f64) -> Self {
        Self {
            sampling,
            min_p,
            min_p_arr: None,
        }
    }

    pub fn with_min_p_arr(mut self, min_p_arr: SamplingTensor1DF32Desc) -> Self {
        self.min_p_arr = Some(min_p_arr);
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        self.sampling.validate()?;
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(FlashInferError::invalid_argument(
                "min_p must be finite and in [0, 1]",
            ));
        }
        validate_f32_param_array("min_p_arr", self.min_p_arr, self.sampling.input)
    }
}

/// Parameters for fused top-k/top-p sampling from FP32 probabilities.
#[derive(Debug, Clone, Copy)]
pub struct TopKTopPSamplingParams {
    /// Common FP32 input, caller-owned I32 output, row mapping, RNG, and stream.
    pub sampling: SamplingParams,
    /// Scalar top-k value, used when `top_k_arr` is `None`.
    pub top_k: i64,
    /// Optional I32 top-k values, contiguous `[input_batch]`.
    pub top_k_arr: Option<SamplingTensor1DI32Desc>,
    /// Scalar top-p threshold in `(0, 1]`, used when `top_p_arr` is `None`.
    pub top_p: f64,
    /// Optional FP32 top-p thresholds, contiguous `[input_batch]`.
    pub top_p_arr: Option<SamplingTensor1DF32Desc>,
}

impl TopKTopPSamplingParams {
    pub fn new(sampling: SamplingParams, top_k: i64, top_p: f64) -> Self {
        Self {
            sampling,
            top_k,
            top_k_arr: None,
            top_p,
            top_p_arr: None,
        }
    }

    pub fn with_top_k_arr(mut self, top_k_arr: SamplingTensor1DI32Desc) -> Self {
        self.top_k_arr = Some(top_k_arr);
        self
    }

    pub fn with_top_p_arr(mut self, top_p_arr: SamplingTensor1DF32Desc) -> Self {
        self.top_p_arr = Some(top_p_arr);
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        self.sampling.validate()?;
        validate_top_k(self.top_k, self.sampling.input.cols)?;
        validate_probability_threshold("top_p", self.top_p)?;
        validate_i32_param_array("top_k_arr", self.top_k_arr, self.sampling.input)?;
        validate_f32_param_array("top_p_arr", self.top_p_arr, self.sampling.input)
    }
}

/// Parameters for caller-owned FP32 softmax output and workspace.
#[derive(Debug, Clone, Copy)]
pub struct SamplingSoftmaxParams {
    /// Caller-owned byte workspace, contiguous `[workspace_bytes]`.
    pub workspace: SamplingWorkspaceDesc,
    /// FP32 logits, contiguous rank-2 `[batch_size, vocab_size]`.
    pub logits: SamplingTensor2DF32Desc,
    /// Caller-owned FP32 probabilities, same shape and layout as `logits`.
    pub output: SamplingTensor2DF32Desc,
    /// Scalar positive temperature, used when `temperature_arr` is `None`.
    pub temperature: f64,
    /// Optional positive FP32 temperatures, contiguous `[batch_size]`.
    pub temperature_arr: Option<SamplingTensor1DF32Desc>,
    /// Whether to enable Programmatic Dependent Launch.
    pub enable_pdl: bool,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    pub stream: *mut c_void,
}

impl SamplingSoftmaxParams {
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_f32_matrix("logits", self.logits)?;
        validate_f32_matrix("output", self.output)?;
        validate_same_matrix("logits", self.logits, "output", self.output)?;
        validate_workspace(self.workspace, self.logits.device_id)?;
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(FlashInferError::invalid_argument(
                "temperature must be finite and positive",
            ));
        }
        validate_f32_param_array("temperature_arr", self.temperature_arr, self.logits)
    }
}

/// Parameters for top-p probability renormalization.
#[derive(Debug, Clone, Copy)]
pub struct TopPRenormParams {
    /// FP32 probabilities, contiguous rank-2 `[batch_size, vocab_size]`.
    pub probs: SamplingTensor2DF32Desc,
    /// Caller-owned FP32 output, same shape and layout as `probs`.
    pub output: SamplingTensor2DF32Desc,
    /// Scalar top-p threshold in `(0, 1]`, used when `top_p_arr` is `None`.
    pub top_p: f64,
    /// Optional FP32 top-p thresholds, contiguous `[batch_size]`.
    pub top_p_arr: Option<SamplingTensor1DF32Desc>,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    pub stream: *mut c_void,
}

impl TopPRenormParams {
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_f32_transform(self.probs, self.output)?;
        validate_probability_threshold("top_p", self.top_p)?;
        validate_f32_param_array("top_p_arr", self.top_p_arr, self.probs)
    }
}

/// Parameters for top-k probability renormalization.
#[derive(Debug, Clone, Copy)]
pub struct TopKRenormParams {
    /// FP32 probabilities, contiguous rank-2 `[batch_size, vocab_size]`.
    pub probs: SamplingTensor2DF32Desc,
    /// Caller-owned FP32 output, same shape and layout as `probs`.
    pub output: SamplingTensor2DF32Desc,
    /// Scalar top-k value, used when `top_k_arr` is `None`.
    pub top_k: i64,
    /// Optional I32 top-k values, contiguous `[batch_size]`.
    pub top_k_arr: Option<SamplingTensor1DI32Desc>,
    /// Caller-owned byte row-state workspace, at least 1 MiB and zeroed before first use.
    pub workspace: SamplingWorkspaceDesc,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    pub stream: *mut c_void,
}

impl TopKRenormParams {
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_f32_transform(self.probs, self.output)?;
        validate_top_k(self.top_k, self.probs.cols)?;
        validate_i32_param_array("top_k_arr", self.top_k_arr, self.probs)?;
        validate_workspace(self.workspace, self.probs.device_id)
    }
}

/// Parameters for top-k masking of FP32 logits.
#[derive(Debug, Clone, Copy)]
pub struct TopKMaskLogitsParams {
    /// FP32 logits, contiguous rank-2 `[batch_size, vocab_size]`.
    pub logits: SamplingTensor2DF32Desc,
    /// Caller-owned FP32 output, same shape and layout as `logits`.
    pub output: SamplingTensor2DF32Desc,
    /// Scalar top-k value, used when `top_k_arr` is `None`.
    pub top_k: i64,
    /// Optional I32 top-k values, contiguous `[batch_size]`.
    pub top_k_arr: Option<SamplingTensor1DI32Desc>,
    /// Caller-owned byte row-state workspace, at least 1 MiB and zeroed before first use.
    pub workspace: SamplingWorkspaceDesc,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    pub stream: *mut c_void,
}

impl TopKMaskLogitsParams {
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_f32_transform(self.logits, self.output)?;
        validate_top_k(self.top_k, self.logits.cols)?;
        validate_i32_param_array("top_k_arr", self.top_k_arr, self.logits)?;
        validate_workspace(self.workspace, self.logits.device_id)
    }
}

pub fn sampling_from_probs(params: &SamplingFromProbsParams) -> Result<(), FlashInferError> {
    params.validate()?;
    call_basic_sampling(SamplingKernel::SamplingFromProbs, params)
}

pub fn sampling_from_logits(params: &SamplingFromLogitsParams) -> Result<(), FlashInferError> {
    params.validate()?;
    call_basic_sampling(SamplingKernel::SamplingFromLogits, params)
}

pub fn top_p_sampling_from_probs(params: &TopPSamplingParams) -> Result<(), FlashInferError> {
    params.validate()?;
    call_filtered_sampling(
        SamplingKernel::TopPSamplingFromProbs,
        &params.sampling,
        None,
        params.top_p_arr,
        0,
        params.top_p,
    )
}

pub fn top_k_sampling_from_probs(params: &TopKSamplingParams) -> Result<(), FlashInferError> {
    params.validate()?;
    call_filtered_sampling(
        SamplingKernel::TopKSamplingFromProbs,
        &params.sampling,
        params.top_k_arr,
        None,
        params.top_k,
        0.0,
    )
}

pub fn min_p_sampling_from_probs(params: &MinPSamplingParams) -> Result<(), FlashInferError> {
    params.validate()?;
    call_filtered_sampling(
        SamplingKernel::MinPSamplingFromProbs,
        &params.sampling,
        None,
        params.min_p_arr,
        0,
        params.min_p,
    )
}

pub fn top_k_top_p_sampling_from_probs(
    params: &TopKTopPSamplingParams,
) -> Result<(), FlashInferError> {
    params.validate()?;
    call_filtered_sampling(
        SamplingKernel::TopKTopPSamplingFromProbs,
        &params.sampling,
        params.top_k_arr,
        params.top_p_arr,
        params.top_k,
        params.top_p,
    )
}

pub fn sampling_softmax(params: &SamplingSoftmaxParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;

    let mut workspace_shape = [params.workspace.len_bytes];
    let mut workspace_strides = [1];
    let workspace = tensor_1d(
        params.workspace.ptr,
        params.workspace.device_id,
        &mut workspace_shape,
        &mut workspace_strides,
        dl_u8(),
    );
    let mut logits_shape = [params.logits.rows, params.logits.cols];
    let mut logits_strides = [params.logits.stride_row, params.logits.stride_col];
    let logits = tensor_2d_f32(params.logits, &mut logits_shape, &mut logits_strides);
    let mut output_shape = [params.output.rows, params.output.cols];
    let mut output_strides = [params.output.stride_row, params.output.stride_col];
    let output = tensor_2d_f32(params.output, &mut output_shape, &mut output_strides);
    let mut temperature_shape = [params.temperature_arr.map_or(0, |value| value.len)];
    let mut temperature_strides = [params.temperature_arr.map_or(1, |value| value.stride)];
    let temperature = params.temperature_arr.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut temperature_shape,
            &mut temperature_strides,
            dl_f32(),
        )
    });

    let args = [
        any_dltensor_ptr(&workspace),
        any_dltensor_ptr(&logits),
        any_dltensor_ptr(&output),
        optional_tensor_any(temperature.as_ref()),
        any_f64(params.temperature),
        any_bool(params.enable_pdl),
    ];
    call_kernel_with_stream(
        runtime,
        SamplingKernel::Softmax,
        params.logits.device_id,
        params.stream,
        &args,
    )
}

pub fn top_p_renorm_probs(params: &TopPRenormParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let mut input_shape = [params.probs.rows, params.probs.cols];
    let mut input_strides = [params.probs.stride_row, params.probs.stride_col];
    let input = tensor_2d_f32(params.probs, &mut input_shape, &mut input_strides);
    let mut output_shape = [params.output.rows, params.output.cols];
    let mut output_strides = [params.output.stride_row, params.output.stride_col];
    let output = tensor_2d_f32(params.output, &mut output_shape, &mut output_strides);
    let mut param_shape = [params.top_p_arr.map_or(0, |value| value.len)];
    let mut param_strides = [params.top_p_arr.map_or(1, |value| value.stride)];
    let param = params.top_p_arr.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut param_shape,
            &mut param_strides,
            dl_f32(),
        )
    });
    let args = [
        any_dltensor_ptr(&input),
        any_dltensor_ptr(&output),
        optional_tensor_any(param.as_ref()),
        any_f64(params.top_p),
    ];
    call_global_kernel(
        SamplingKernel::TopPRenormProbs,
        params.probs.device_id,
        params.stream,
        &args,
    )
}

pub fn top_k_renorm_probs(params: &TopKRenormParams) -> Result<(), FlashInferError> {
    params.validate()?;
    call_top_k_transform(
        SamplingKernel::TopKRenormProbs,
        params.probs,
        params.output,
        params.top_k_arr,
        params.top_k,
        params.workspace,
        params.stream,
    )
}

pub fn top_k_mask_logits(params: &TopKMaskLogitsParams) -> Result<(), FlashInferError> {
    params.validate()?;
    call_top_k_transform(
        SamplingKernel::TopKMaskLogits,
        params.logits,
        params.output,
        params.top_k_arr,
        params.top_k,
        params.workspace,
        params.stream,
    )
}

fn call_basic_sampling(
    kernel: SamplingKernel,
    params: &SamplingParams,
) -> Result<(), FlashInferError> {
    let runtime = FlashInferRuntime::global()?;
    let mut input_shape = [params.input.rows, params.input.cols];
    let mut input_strides = [params.input.stride_row, params.input.stride_col];
    let input = tensor_2d_f32(params.input, &mut input_shape, &mut input_strides);
    let mut output_shape = [params.output.len];
    let mut output_strides = [params.output.stride];
    let output = tensor_1d(
        params.output.ptr,
        params.output.device_id,
        &mut output_shape,
        &mut output_strides,
        dl_i32(),
    );
    let mut indices_shape = [params.row_indices.map_or(0, |value| value.len)];
    let mut indices_strides = [params.row_indices.map_or(1, |value| value.stride)];
    let indices = params.row_indices.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut indices_shape,
            &mut indices_strides,
            dl_i32(),
        )
    });
    let mut seed_shape = [params.random.seed_arr.map_or(0, |value| value.len)];
    let mut seed_strides = [params.random.seed_arr.map_or(1, |value| value.stride)];
    let seed_arr = params.random.seed_arr.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut seed_shape,
            &mut seed_strides,
            dl_u64(),
        )
    });
    let mut offset_shape = [params.random.offset_arr.map_or(0, |value| value.len)];
    let mut offset_strides = [params.random.offset_arr.map_or(1, |value| value.stride)];
    let offset_arr = params.random.offset_arr.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut offset_shape,
            &mut offset_strides,
            dl_u64(),
        )
    });
    let args = pack_basic_sampling_args(
        &input,
        &output,
        indices.as_ref(),
        seed_arr.as_ref(),
        offset_arr.as_ref(),
        params.random,
    );
    call_kernel_with_stream(
        runtime,
        kernel,
        params.input.device_id,
        params.stream,
        &args,
    )
}

#[allow(clippy::too_many_arguments)]
fn call_filtered_sampling(
    kernel: SamplingKernel,
    params: &SamplingParams,
    i32_param: Option<SamplingTensor1DI32Desc>,
    f32_param: Option<SamplingTensor1DF32Desc>,
    i64_value: i64,
    f64_value: f64,
) -> Result<(), FlashInferError> {
    let runtime = FlashInferRuntime::global()?;
    let mut input_shape = [params.input.rows, params.input.cols];
    let mut input_strides = [params.input.stride_row, params.input.stride_col];
    let input = tensor_2d_f32(params.input, &mut input_shape, &mut input_strides);
    let mut output_shape = [params.output.len];
    let mut output_strides = [params.output.stride];
    let output = tensor_1d(
        params.output.ptr,
        params.output.device_id,
        &mut output_shape,
        &mut output_strides,
        dl_i32(),
    );
    let mut indices_shape = [params.row_indices.map_or(0, |value| value.len)];
    let mut indices_strides = [params.row_indices.map_or(1, |value| value.stride)];
    let indices = params.row_indices.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut indices_shape,
            &mut indices_strides,
            dl_i32(),
        )
    });
    let mut i32_param_shape = [i32_param.map_or(0, |value| value.len)];
    let mut i32_param_strides = [i32_param.map_or(1, |value| value.stride)];
    let i32_param_tensor = i32_param.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut i32_param_shape,
            &mut i32_param_strides,
            dl_i32(),
        )
    });
    let mut f32_param_shape = [f32_param.map_or(0, |value| value.len)];
    let mut f32_param_strides = [f32_param.map_or(1, |value| value.stride)];
    let f32_param_tensor = f32_param.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut f32_param_shape,
            &mut f32_param_strides,
            dl_f32(),
        )
    });
    let mut seed_shape = [params.random.seed_arr.map_or(0, |value| value.len)];
    let mut seed_strides = [params.random.seed_arr.map_or(1, |value| value.stride)];
    let seed_arr = params.random.seed_arr.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut seed_shape,
            &mut seed_strides,
            dl_u64(),
        )
    });
    let mut offset_shape = [params.random.offset_arr.map_or(0, |value| value.len)];
    let mut offset_strides = [params.random.offset_arr.map_or(1, |value| value.stride)];
    let offset_arr = params.random.offset_arr.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut offset_shape,
            &mut offset_strides,
            dl_u64(),
        )
    });
    let args = pack_filtered_sampling_args(
        kernel,
        &input,
        &output,
        indices.as_ref(),
        i32_param_tensor.as_ref(),
        f32_param_tensor.as_ref(),
        i64_value,
        f64_value,
        seed_arr.as_ref(),
        offset_arr.as_ref(),
        params.random,
    )?;

    call_kernel_with_stream(
        runtime,
        kernel,
        params.input.device_id,
        params.stream,
        &args,
    )
}

#[allow(clippy::too_many_arguments)]
fn call_top_k_transform(
    kernel: SamplingKernel,
    input_desc: SamplingTensor2DF32Desc,
    output_desc: SamplingTensor2DF32Desc,
    top_k_arr: Option<SamplingTensor1DI32Desc>,
    top_k: i64,
    workspace_desc: SamplingWorkspaceDesc,
    stream: *mut c_void,
) -> Result<(), FlashInferError> {
    let mut input_shape = [input_desc.rows, input_desc.cols];
    let mut input_strides = [input_desc.stride_row, input_desc.stride_col];
    let input = tensor_2d_f32(input_desc, &mut input_shape, &mut input_strides);
    let mut output_shape = [output_desc.rows, output_desc.cols];
    let mut output_strides = [output_desc.stride_row, output_desc.stride_col];
    let output = tensor_2d_f32(output_desc, &mut output_shape, &mut output_strides);
    let mut param_shape = [top_k_arr.map_or(0, |value| value.len)];
    let mut param_strides = [top_k_arr.map_or(1, |value| value.stride)];
    let param = top_k_arr.map(|value| {
        tensor_1d(
            value.ptr,
            value.device_id,
            &mut param_shape,
            &mut param_strides,
            dl_i32(),
        )
    });
    let mut workspace_shape = [workspace_desc.len_bytes];
    let mut workspace_strides = [1];
    let workspace = tensor_1d(
        workspace_desc.ptr,
        workspace_desc.device_id,
        &mut workspace_shape,
        &mut workspace_strides,
        dl_u8(),
    );
    let args = [
        any_dltensor_ptr(&input),
        any_dltensor_ptr(&output),
        optional_tensor_any(param.as_ref()),
        any_i64(top_k),
        any_dltensor_ptr(&workspace),
    ];
    call_global_kernel(kernel, input_desc.device_id, stream, &args)
}

fn call_global_kernel(
    kernel: SamplingKernel,
    device_id: i32,
    stream: *mut c_void,
    args: &[TVMFFIAny],
) -> Result<(), FlashInferError> {
    let runtime = FlashInferRuntime::global()?;
    call_kernel_with_stream(runtime, kernel, device_id, stream, args)
}

fn call_kernel_with_stream(
    runtime: &FlashInferRuntime,
    kernel: SamplingKernel,
    device_id: i32,
    stream: *mut c_void,
    args: &[TVMFFIAny],
) -> Result<(), FlashInferError> {
    let mut result = any_none();
    // SAFETY: the stream context API is resolved and validated during runtime initialization.
    let previous_stream = unsafe { runtime.set_stream(device_id, stream)? };
    let mut restore_guard = StreamRestoreGuard::new(runtime, device_id, previous_stream);
    // SAFETY: every descriptor and scalar has been validated and packed in the pinned ABI order.
    let call_result = unsafe {
        runtime.call_sampling(
            kernel,
            args.as_ptr(),
            args.len() as i32,
            &mut result as *mut _,
        )
    };
    // Safe-call results are owned. Sampling normally returns None, but guard
    // any future object result immediately so ABI changes cannot leak it.
    let _result_guard = AnyObjectDecRefGuard::new(runtime, &result);
    let restore_result = restore_guard.restore_now();
    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

struct AnyObjectDecRefGuard<'a> {
    runtime: &'a FlashInferRuntime,
    object: *mut c_void,
}

impl<'a> AnyObjectDecRefGuard<'a> {
    fn new(runtime: &'a FlashInferRuntime, value: &TVMFFIAny) -> Self {
        Self {
            runtime,
            object: any_object_handle(value).unwrap_or(std::ptr::null_mut()),
        }
    }
}

impl Drop for AnyObjectDecRefGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: an object-like safe-call result is owned by Rust and must be
        // decref'd exactly once; None/POD/rvalue-ref results produce null.
        unsafe {
            self.runtime.object_dec_ref(self.object);
        }
    }
}

fn tensor_2d_f32(
    desc: SamplingTensor2DF32Desc,
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
        dtype: dl_f32(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn tensor_1d(
    ptr: *const c_void,
    device_id: i32,
    shape: &mut [i64; 1],
    strides: &mut [i64; 1],
    dtype: DLDataType,
) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim: 1,
        dtype,
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn optional_tensor_any(tensor: Option<&DLTensor>) -> TVMFFIAny {
    tensor.map_or_else(any_none, |tensor| any_dltensor_ptr(tensor))
}

fn pack_basic_sampling_args(
    input: &DLTensor,
    output: &DLTensor,
    indices: Option<&DLTensor>,
    seed_arr: Option<&DLTensor>,
    offset_arr: Option<&DLTensor>,
    random: SamplingRandomParams,
) -> [TVMFFIAny; 8] {
    [
        any_dltensor_ptr(input),
        any_dltensor_ptr(output),
        optional_tensor_any(indices),
        any_bool(random.deterministic),
        optional_tensor_any(seed_arr),
        any_u64(random.seed),
        optional_tensor_any(offset_arr),
        any_u64(random.offset),
    ]
}

#[allow(clippy::too_many_arguments)]
fn pack_filtered_sampling_args(
    kernel: SamplingKernel,
    input: &DLTensor,
    output: &DLTensor,
    indices: Option<&DLTensor>,
    i32_param: Option<&DLTensor>,
    f32_param: Option<&DLTensor>,
    i64_value: i64,
    f64_value: f64,
    seed_arr: Option<&DLTensor>,
    offset_arr: Option<&DLTensor>,
    random: SamplingRandomParams,
) -> Result<Vec<TVMFFIAny>, FlashInferError> {
    let tail = [
        any_bool(random.deterministic),
        optional_tensor_any(seed_arr),
        any_u64(random.seed),
        optional_tensor_any(offset_arr),
        any_u64(random.offset),
    ];
    let args = match kernel {
        SamplingKernel::TopPSamplingFromProbs | SamplingKernel::MinPSamplingFromProbs => vec![
            any_dltensor_ptr(input),
            any_dltensor_ptr(output),
            optional_tensor_any(indices),
            optional_tensor_any(f32_param),
            any_f64(f64_value),
            tail[0],
            tail[1],
            tail[2],
            tail[3],
            tail[4],
        ],
        SamplingKernel::TopKSamplingFromProbs => vec![
            any_dltensor_ptr(input),
            any_dltensor_ptr(output),
            optional_tensor_any(indices),
            optional_tensor_any(i32_param),
            any_i64(i64_value),
            tail[0],
            tail[1],
            tail[2],
            tail[3],
            tail[4],
        ],
        SamplingKernel::TopKTopPSamplingFromProbs => vec![
            any_dltensor_ptr(input),
            any_dltensor_ptr(output),
            optional_tensor_any(indices),
            optional_tensor_any(i32_param),
            // The pinned binding declares top_k_val as double for this fused entry point.
            any_f64(i64_value as f64),
            optional_tensor_any(f32_param),
            any_f64(f64_value),
            tail[0],
            tail[1],
            tail[2],
            tail[3],
            tail[4],
        ],
        _ => {
            return Err(FlashInferError::invalid_argument(
                "internal sampling kernel/argument mismatch",
            ));
        }
    };
    Ok(args)
}

fn any_u64(value: u64) -> TVMFFIAny {
    any_i64(value as i64)
}

fn dl_f32() -> DLDataType {
    DLDataType {
        code: KDL_FLOAT,
        bits: 32,
        lanes: 1,
    }
}

fn dl_i32() -> DLDataType {
    DLDataType {
        code: KDL_INT,
        bits: 32,
        lanes: 1,
    }
}

fn dl_u64() -> DLDataType {
    DLDataType {
        code: KDL_UINT,
        bits: 64,
        lanes: 1,
    }
}

fn dl_u8() -> DLDataType {
    DLDataType {
        code: KDL_UINT,
        bits: 8,
        lanes: 1,
    }
}

fn validate_sampling_base(
    input: SamplingTensor2DF32Desc,
    output: SamplingTensor1DI32Desc,
    row_indices: Option<SamplingTensor1DI32Desc>,
    random: SamplingRandomParams,
) -> Result<(), FlashInferError> {
    validate_f32_matrix("input", input)?;
    validate_i32_vector("output", output)?;
    if let Some(indices) = row_indices {
        validate_i32_vector("row_indices", indices)?;
        if indices.len != output.len {
            return Err(FlashInferError::invalid_argument(
                "row_indices length must equal output length",
            ));
        }
        if indices.device_id != input.device_id {
            return Err(FlashInferError::invalid_argument(
                "row_indices must be on the input CUDA device",
            ));
        }
    } else if output.len != input.rows {
        return Err(FlashInferError::invalid_argument(
            "output length must equal input batch when row_indices is absent",
        ));
    }
    if output.device_id != input.device_id {
        return Err(FlashInferError::invalid_argument(
            "output must be on the input CUDA device",
        ));
    }
    validate_random_params(random, output.len, input.device_id)?;
    Ok(())
}

fn validate_random_params(
    random: SamplingRandomParams,
    output_batch: i64,
    device_id: i32,
) -> Result<(), FlashInferError> {
    match (random.seed_arr, random.offset_arr) {
        (None, None) => Ok(()),
        (Some(seed_arr), Some(offset_arr)) => {
            validate_u64_vector("seed_arr", seed_arr)?;
            validate_u64_vector("offset_arr", offset_arr)?;
            for (name, desc) in [("seed_arr", seed_arr), ("offset_arr", offset_arr)] {
                if desc.len != 1 && desc.len != output_batch {
                    return Err(FlashInferError::invalid_argument(format!(
                        "{name} length must be 1 or output batch ({output_batch})"
                    )));
                }
                if desc.device_id != device_id {
                    return Err(FlashInferError::invalid_argument(format!(
                        "{name} must be on the input CUDA device"
                    )));
                }
            }
            if seed_arr.len != offset_arr.len {
                return Err(FlashInferError::invalid_argument(
                    "seed_arr and offset_arr lengths must match",
                ));
            }
            Ok(())
        }
        _ => Err(FlashInferError::invalid_argument(
            "seed_arr and offset_arr must either both be present or both be absent",
        )),
    }
}

fn validate_f32_transform(
    input: SamplingTensor2DF32Desc,
    output: SamplingTensor2DF32Desc,
) -> Result<(), FlashInferError> {
    validate_f32_matrix("input", input)?;
    validate_f32_matrix("output", output)?;
    validate_same_matrix("input", input, "output", output)
}

fn validate_same_matrix(
    lhs_name: &str,
    lhs: SamplingTensor2DF32Desc,
    rhs_name: &str,
    rhs: SamplingTensor2DF32Desc,
) -> Result<(), FlashInferError> {
    if lhs.rows != rhs.rows || lhs.cols != rhs.cols {
        return Err(FlashInferError::invalid_argument(format!(
            "{rhs_name} shape must match {lhs_name} shape"
        )));
    }
    if lhs.stride_row != rhs.stride_row || lhs.stride_col != rhs.stride_col {
        return Err(FlashInferError::invalid_argument(format!(
            "{rhs_name} layout must match {lhs_name} layout"
        )));
    }
    if lhs.device_id != rhs.device_id {
        return Err(FlashInferError::invalid_argument(format!(
            "{rhs_name} must be on the same CUDA device as {lhs_name}"
        )));
    }
    Ok(())
}

fn validate_f32_matrix(name: &str, desc: SamplingTensor2DF32Desc) -> Result<(), FlashInferError> {
    if desc.ptr.is_null() {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} pointer must be non-null"
        )));
    }
    if desc.rows <= 0 || desc.cols <= 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} dimensions must be positive"
        )));
    }
    if desc.rows > u32::MAX as i64 || desc.cols > u32::MAX as i64 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} dimensions must fit in u32"
        )));
    }
    if desc.stride_col != 1 || desc.stride_row != desc.cols {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be contiguous row-major"
        )));
    }
    if desc.device_id < 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} device_id must be non-negative"
        )));
    }
    Ok(())
}

fn validate_i32_vector(name: &str, desc: SamplingTensor1DI32Desc) -> Result<(), FlashInferError> {
    validate_vector_parts(name, desc.ptr, desc.len, desc.stride, desc.device_id)
}

fn validate_f32_vector(name: &str, desc: SamplingTensor1DF32Desc) -> Result<(), FlashInferError> {
    validate_vector_parts(name, desc.ptr, desc.len, desc.stride, desc.device_id)
}

fn validate_u64_vector(name: &str, desc: SamplingTensor1DU64Desc) -> Result<(), FlashInferError> {
    validate_vector_parts(name, desc.ptr, desc.len, desc.stride, desc.device_id)
}

fn validate_vector_parts(
    name: &str,
    ptr: *const c_void,
    len: i64,
    stride: i64,
    device_id: i32,
) -> Result<(), FlashInferError> {
    if ptr.is_null() {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} pointer must be non-null"
        )));
    }
    if len <= 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} length must be positive"
        )));
    }
    if len > u32::MAX as i64 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} length must fit in u32"
        )));
    }
    if stride != 1 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be contiguous"
        )));
    }
    if device_id < 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} device_id must be non-negative"
        )));
    }
    Ok(())
}

fn validate_f32_param_array(
    name: &str,
    array: Option<SamplingTensor1DF32Desc>,
    input: SamplingTensor2DF32Desc,
) -> Result<(), FlashInferError> {
    if let Some(array) = array {
        validate_f32_vector(name, array)?;
        if array.len != input.rows {
            return Err(FlashInferError::invalid_argument(format!(
                "{name} length must equal input batch ({})",
                input.rows
            )));
        }
        if array.device_id != input.device_id {
            return Err(FlashInferError::invalid_argument(format!(
                "{name} must be on the input CUDA device"
            )));
        }
    }
    Ok(())
}

fn validate_i32_param_array(
    name: &str,
    array: Option<SamplingTensor1DI32Desc>,
    input: SamplingTensor2DF32Desc,
) -> Result<(), FlashInferError> {
    if let Some(array) = array {
        validate_i32_vector(name, array)?;
        if array.len != input.rows {
            return Err(FlashInferError::invalid_argument(format!(
                "{name} length must equal input batch ({})",
                input.rows
            )));
        }
        if array.device_id != input.device_id {
            return Err(FlashInferError::invalid_argument(format!(
                "{name} must be on the input CUDA device"
            )));
        }
    }
    Ok(())
}

fn validate_probability_threshold(name: &str, value: f64) -> Result<(), FlashInferError> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be finite and in (0, 1]"
        )));
    }
    Ok(())
}

fn validate_top_k(top_k: i64, vocab_size: i64) -> Result<(), FlashInferError> {
    if top_k <= 0 || top_k > vocab_size {
        return Err(FlashInferError::invalid_argument(format!(
            "top_k must be in 1..={vocab_size}"
        )));
    }
    Ok(())
}

fn validate_workspace(
    workspace: SamplingWorkspaceDesc,
    device_id: i32,
) -> Result<(), FlashInferError> {
    validate_vector_parts(
        "workspace",
        workspace.ptr,
        workspace.len_bytes,
        1,
        workspace.device_id,
    )?;
    if workspace.device_id != device_id {
        return Err(FlashInferError::invalid_argument(
            "workspace must be on the input CUDA device",
        ));
    }
    if workspace.len_bytes < SAMPLING_WORKSPACE_BYTES as i64 {
        return Err(FlashInferError::invalid_argument(format!(
            "workspace must contain at least {SAMPLING_WORKSPACE_BYTES} bytes"
        )));
    }
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
        // SAFETY: `previous_stream` was returned by TVMFFIEnvSetStream for this device.
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
        // SAFETY: best-effort restoration protects the process-wide TVM-FFI stream state.
        let _ = unsafe {
            self.runtime
                .restore_stream(self.device_id, self.previous_stream)
        };
    }
}

/// `cudarc` RNG options for caller-owned sampling buffers.
#[cfg(feature = "cudarc")]
#[derive(Debug, Clone, Copy)]
pub struct SamplingCudarcRandom<'a> {
    pub deterministic: bool,
    pub seed: u64,
    pub offset: u64,
    /// Optional device-resident U64 seed, length one or output batch.
    pub seed_arr: Option<&'a cudarc::driver::CudaSlice<u64>>,
    /// Optional device-resident U64 offset, length one or output batch.
    pub offset_arr: Option<&'a cudarc::driver::CudaSlice<u64>>,
}

#[cfg(feature = "cudarc")]
impl<'a> SamplingCudarcRandom<'a> {
    pub fn new(seed: u64, offset: u64) -> Self {
        Self {
            deterministic: true,
            seed,
            offset,
            seed_arr: None,
            offset_arr: None,
        }
    }

    pub fn with_arrays(
        mut self,
        seed_arr: &'a cudarc::driver::CudaSlice<u64>,
        offset_arr: &'a cudarc::driver::CudaSlice<u64>,
    ) -> Self {
        self.seed_arr = Some(seed_arr);
        self.offset_arr = Some(offset_arr);
        self
    }
}

#[cfg(feature = "cudarc")]
pub fn sampling_from_probs_cudarc<P, O>(
    stream: &cudarc::driver::CudaStream,
    probs: &P,
    output: &mut O,
    input_batch: usize,
    vocab_size: usize,
    row_indices: Option<&cudarc::driver::CudaSlice<i32>>,
    random: SamplingCudarcRandom<'_>,
) -> Result<(), FlashInferError>
where
    P: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtrMut<i32>,
{
    sampling_cudarc_impl(
        stream,
        probs,
        output,
        input_batch,
        vocab_size,
        row_indices,
        random,
        CudarcSamplingFilter::None,
        SamplingKernel::SamplingFromProbs,
    )
}

#[cfg(feature = "cudarc")]
pub fn sampling_from_logits_cudarc<P, O>(
    stream: &cudarc::driver::CudaStream,
    logits: &P,
    output: &mut O,
    input_batch: usize,
    vocab_size: usize,
    row_indices: Option<&cudarc::driver::CudaSlice<i32>>,
    random: SamplingCudarcRandom<'_>,
) -> Result<(), FlashInferError>
where
    P: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtrMut<i32>,
{
    sampling_cudarc_impl(
        stream,
        logits,
        output,
        input_batch,
        vocab_size,
        row_indices,
        random,
        CudarcSamplingFilter::None,
        SamplingKernel::SamplingFromLogits,
    )
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_p_sampling_from_probs_cudarc<P, O>(
    stream: &cudarc::driver::CudaStream,
    probs: &P,
    output: &mut O,
    input_batch: usize,
    vocab_size: usize,
    row_indices: Option<&cudarc::driver::CudaSlice<i32>>,
    top_p: f64,
    top_p_arr: Option<&cudarc::driver::CudaSlice<f32>>,
    random: SamplingCudarcRandom<'_>,
) -> Result<(), FlashInferError>
where
    P: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtrMut<i32>,
{
    sampling_cudarc_impl(
        stream,
        probs,
        output,
        input_batch,
        vocab_size,
        row_indices,
        random,
        CudarcSamplingFilter::TopP {
            value: top_p,
            array: top_p_arr,
        },
        SamplingKernel::TopPSamplingFromProbs,
    )
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_k_sampling_from_probs_cudarc<P, O>(
    stream: &cudarc::driver::CudaStream,
    probs: &P,
    output: &mut O,
    input_batch: usize,
    vocab_size: usize,
    row_indices: Option<&cudarc::driver::CudaSlice<i32>>,
    top_k: i64,
    top_k_arr: Option<&cudarc::driver::CudaSlice<i32>>,
    random: SamplingCudarcRandom<'_>,
) -> Result<(), FlashInferError>
where
    P: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtrMut<i32>,
{
    sampling_cudarc_impl(
        stream,
        probs,
        output,
        input_batch,
        vocab_size,
        row_indices,
        random,
        CudarcSamplingFilter::TopK {
            value: top_k,
            array: top_k_arr,
        },
        SamplingKernel::TopKSamplingFromProbs,
    )
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn min_p_sampling_from_probs_cudarc<P, O>(
    stream: &cudarc::driver::CudaStream,
    probs: &P,
    output: &mut O,
    input_batch: usize,
    vocab_size: usize,
    row_indices: Option<&cudarc::driver::CudaSlice<i32>>,
    min_p: f64,
    min_p_arr: Option<&cudarc::driver::CudaSlice<f32>>,
    random: SamplingCudarcRandom<'_>,
) -> Result<(), FlashInferError>
where
    P: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtrMut<i32>,
{
    sampling_cudarc_impl(
        stream,
        probs,
        output,
        input_batch,
        vocab_size,
        row_indices,
        random,
        CudarcSamplingFilter::MinP {
            value: min_p,
            array: min_p_arr,
        },
        SamplingKernel::MinPSamplingFromProbs,
    )
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_k_top_p_sampling_from_probs_cudarc<P, O>(
    stream: &cudarc::driver::CudaStream,
    probs: &P,
    output: &mut O,
    input_batch: usize,
    vocab_size: usize,
    row_indices: Option<&cudarc::driver::CudaSlice<i32>>,
    top_k: i64,
    top_k_arr: Option<&cudarc::driver::CudaSlice<i32>>,
    top_p: f64,
    top_p_arr: Option<&cudarc::driver::CudaSlice<f32>>,
    random: SamplingCudarcRandom<'_>,
) -> Result<(), FlashInferError>
where
    P: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtrMut<i32>,
{
    sampling_cudarc_impl(
        stream,
        probs,
        output,
        input_batch,
        vocab_size,
        row_indices,
        random,
        CudarcSamplingFilter::TopKTopP {
            top_k,
            top_k_arr,
            top_p,
            top_p_arr,
        },
        SamplingKernel::TopKTopPSamplingFromProbs,
    )
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn sampling_softmax_cudarc<L, O, W>(
    stream: &cudarc::driver::CudaStream,
    logits: &L,
    output: &mut O,
    workspace: &mut W,
    batch_size: usize,
    vocab_size: usize,
    temperature: f64,
    temperature_arr: Option<&cudarc::driver::CudaSlice<f32>>,
    enable_pdl: bool,
) -> Result<(), FlashInferError>
where
    L: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtrMut<f32>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    validate_flat_matrix_lengths(logits.len(), output.len(), batch_size, vocab_size)?;
    if workspace.len() < SAMPLING_WORKSPACE_BYTES {
        return Err(FlashInferError::invalid_argument(format!(
            "workspace length ({}) must be at least {SAMPLING_WORKSPACE_BYTES}",
            workspace.len()
        )));
    }
    if let Some(array) = temperature_arr
        && array.len() != batch_size
    {
        return Err(FlashInferError::invalid_argument(format!(
            "temperature_arr length ({}) must equal batch_size ({batch_size})",
            array.len()
        )));
    }

    let workspace_len = workspace.len();
    let (logits_ptr, _logits_sync) = logits.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let (workspace_ptr, _workspace_sync) = workspace.device_ptr_mut(stream);
    let (temperature_ptr, _temperature_sync) = match temperature_arr {
        Some(array) => {
            let (ptr, sync) = array.device_ptr(stream);
            (Some(ptr), Some(sync))
        }
        None => (None, None),
    };
    let dims = cudarc_dims(stream, batch_size, vocab_size)?;
    let temperature_desc = temperature_ptr.map(|ptr| SamplingTensor1DF32Desc {
        ptr: ptr as usize as *const c_void,
        len: dims.0,
        stride: 1,
        device_id: dims.2,
    });
    sampling_softmax(&SamplingSoftmaxParams {
        workspace: SamplingWorkspaceDesc {
            ptr: workspace_ptr as usize as *const c_void,
            len_bytes: workspace_len as i64,
            device_id: dims.2,
        },
        logits: contiguous_f32_desc(logits_ptr, dims),
        output: contiguous_f32_desc(output_ptr, dims),
        temperature,
        temperature_arr: temperature_desc,
        enable_pdl,
        stream: stream.cu_stream().cast(),
    })
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_p_renorm_probs_cudarc<P, O>(
    stream: &cudarc::driver::CudaStream,
    probs: &P,
    output: &mut O,
    batch_size: usize,
    vocab_size: usize,
    top_p: f64,
    top_p_arr: Option<&cudarc::driver::CudaSlice<f32>>,
) -> Result<(), FlashInferError>
where
    P: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtrMut<f32>,
{
    validate_flat_matrix_lengths(probs.len(), output.len(), batch_size, vocab_size)?;
    if let Some(array) = top_p_arr
        && array.len() != batch_size
    {
        return Err(FlashInferError::invalid_argument(format!(
            "top_p_arr length ({}) must equal batch_size ({batch_size})",
            array.len()
        )));
    }
    let (probs_ptr, _probs_sync) = probs.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let (array_ptr, _array_sync) = match top_p_arr {
        Some(array) => {
            let (ptr, sync) = array.device_ptr(stream);
            (Some(ptr), Some(sync))
        }
        None => (None, None),
    };
    let dims = cudarc_dims(stream, batch_size, vocab_size)?;
    top_p_renorm_probs(&TopPRenormParams {
        probs: contiguous_f32_desc(probs_ptr, dims),
        output: contiguous_f32_desc(output_ptr, dims),
        top_p,
        top_p_arr: array_ptr.map(|ptr| SamplingTensor1DF32Desc {
            ptr: ptr as usize as *const c_void,
            len: dims.0,
            stride: 1,
            device_id: dims.2,
        }),
        stream: stream.cu_stream().cast(),
    })
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_k_renorm_probs_cudarc<P, O, W>(
    stream: &cudarc::driver::CudaStream,
    probs: &P,
    output: &mut O,
    workspace: &mut W,
    batch_size: usize,
    vocab_size: usize,
    top_k: i64,
    top_k_arr: Option<&cudarc::driver::CudaSlice<i32>>,
) -> Result<(), FlashInferError>
where
    P: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtrMut<f32>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    run_top_k_transform_cudarc(
        stream,
        probs,
        output,
        workspace,
        batch_size,
        vocab_size,
        top_k,
        top_k_arr,
        SamplingKernel::TopKRenormProbs,
    )
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_k_mask_logits_cudarc<L, O, W>(
    stream: &cudarc::driver::CudaStream,
    logits: &L,
    output: &mut O,
    workspace: &mut W,
    batch_size: usize,
    vocab_size: usize,
    top_k: i64,
    top_k_arr: Option<&cudarc::driver::CudaSlice<i32>>,
) -> Result<(), FlashInferError>
where
    L: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtrMut<f32>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    run_top_k_transform_cudarc(
        stream,
        logits,
        output,
        workspace,
        batch_size,
        vocab_size,
        top_k,
        top_k_arr,
        SamplingKernel::TopKMaskLogits,
    )
}

#[cfg(feature = "cudarc")]
#[derive(Clone, Copy)]
enum CudarcSamplingFilter<'a> {
    None,
    TopP {
        value: f64,
        array: Option<&'a cudarc::driver::CudaSlice<f32>>,
    },
    TopK {
        value: i64,
        array: Option<&'a cudarc::driver::CudaSlice<i32>>,
    },
    MinP {
        value: f64,
        array: Option<&'a cudarc::driver::CudaSlice<f32>>,
    },
    TopKTopP {
        top_k: i64,
        top_k_arr: Option<&'a cudarc::driver::CudaSlice<i32>>,
        top_p: f64,
        top_p_arr: Option<&'a cudarc::driver::CudaSlice<f32>>,
    },
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
fn sampling_cudarc_impl<P, O>(
    stream: &cudarc::driver::CudaStream,
    input: &P,
    output: &mut O,
    input_batch: usize,
    vocab_size: usize,
    row_indices: Option<&cudarc::driver::CudaSlice<i32>>,
    random: SamplingCudarcRandom<'_>,
    filter: CudarcSamplingFilter<'_>,
    kernel: SamplingKernel,
) -> Result<(), FlashInferError>
where
    P: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtrMut<i32>,
{
    let expected = input_batch
        .checked_mul(vocab_size)
        .ok_or_else(|| FlashInferError::invalid_argument("input_batch * vocab_size overflow"))?;
    if input.len() != expected {
        return Err(FlashInferError::invalid_argument(format!(
            "input length ({}) must equal input_batch * vocab_size ({expected})",
            input.len()
        )));
    }
    if output.is_empty() {
        return Err(FlashInferError::invalid_argument(
            "output length must be positive",
        ));
    }
    match row_indices {
        Some(indices) if indices.len() != output.len() => {
            return Err(FlashInferError::invalid_argument(format!(
                "row_indices length ({}) must equal output length ({})",
                indices.len(),
                output.len()
            )));
        }
        None if output.len() != input_batch => {
            return Err(FlashInferError::invalid_argument(format!(
                "output length ({}) must equal input_batch ({input_batch}) without row_indices",
                output.len()
            )));
        }
        _ => {}
    }
    match (random.seed_arr, random.offset_arr) {
        (None, None) => {}
        (Some(seed_arr), Some(offset_arr)) => {
            let valid_len = |len: usize| len == 1 || len == output.len();
            if !valid_len(seed_arr.len()) || !valid_len(offset_arr.len()) {
                return Err(FlashInferError::invalid_argument(format!(
                    "seed_arr and offset_arr lengths must be 1 or output batch ({})",
                    output.len()
                )));
            }
            if seed_arr.len() != offset_arr.len() {
                return Err(FlashInferError::invalid_argument(
                    "seed_arr and offset_arr lengths must match",
                ));
            }
        }
        _ => {
            return Err(FlashInferError::invalid_argument(
                "seed_arr and offset_arr must either both be present or both be absent",
            ));
        }
    }
    let validate_param_len = |name: &str, len: usize| {
        if len == input_batch {
            Ok(())
        } else {
            Err(FlashInferError::invalid_argument(format!(
                "{name} length ({len}) must equal input_batch ({input_batch})"
            )))
        }
    };
    match filter {
        CudarcSamplingFilter::TopP { array: Some(a), .. } => {
            validate_param_len("top_p_arr", a.len())?
        }
        CudarcSamplingFilter::TopK { array: Some(a), .. } => {
            validate_param_len("top_k_arr", a.len())?
        }
        CudarcSamplingFilter::MinP { array: Some(a), .. } => {
            validate_param_len("min_p_arr", a.len())?
        }
        CudarcSamplingFilter::TopKTopP {
            top_k_arr,
            top_p_arr,
            ..
        } => {
            if let Some(a) = top_k_arr {
                validate_param_len("top_k_arr", a.len())?;
            }
            if let Some(a) = top_p_arr {
                validate_param_len("top_p_arr", a.len())?;
            }
        }
        _ => {}
    }
    let output_len_usize = output.len();
    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let (indices_ptr, _indices_sync) = match row_indices {
        Some(indices) => {
            let (ptr, sync) = indices.device_ptr(stream);
            (Some(ptr), Some(sync))
        }
        None => (None, None),
    };
    let (seed_ptr, _seed_sync) = match random.seed_arr {
        Some(array) => {
            let (ptr, sync) = array.device_ptr(stream);
            (Some(ptr), Some(sync))
        }
        None => (None, None),
    };
    let (offset_ptr, _offset_sync) = match random.offset_arr {
        Some(array) => {
            let (ptr, sync) = array.device_ptr(stream);
            (Some(ptr), Some(sync))
        }
        None => (None, None),
    };
    let (top_k_ptr, _top_k_sync, top_p_ptr, _top_p_sync) = match filter {
        CudarcSamplingFilter::TopP { array, .. } | CudarcSamplingFilter::MinP { array, .. } => {
            let (ptr, sync) = match array {
                Some(array) => {
                    let (ptr, sync) = array.device_ptr(stream);
                    (Some(ptr), Some(sync))
                }
                None => (None, None),
            };
            (None, None, ptr, sync)
        }
        CudarcSamplingFilter::TopK { array, .. } => {
            let (ptr, sync) = match array {
                Some(array) => {
                    let (ptr, sync) = array.device_ptr(stream);
                    (Some(ptr), Some(sync))
                }
                None => (None, None),
            };
            (ptr, sync, None, None)
        }
        CudarcSamplingFilter::TopKTopP {
            top_k_arr,
            top_p_arr,
            ..
        } => {
            let (k_ptr, k_sync) = match top_k_arr {
                Some(array) => {
                    let (ptr, sync) = array.device_ptr(stream);
                    (Some(ptr), Some(sync))
                }
                None => (None, None),
            };
            let (p_ptr, p_sync) = match top_p_arr {
                Some(array) => {
                    let (ptr, sync) = array.device_ptr(stream);
                    (Some(ptr), Some(sync))
                }
                None => (None, None),
            };
            (k_ptr, k_sync, p_ptr, p_sync)
        }
        CudarcSamplingFilter::None => (None, None, None, None),
    };

    let dims = cudarc_dims(stream, input_batch, vocab_size)?;
    let output_len = i64::try_from(output_len_usize)
        .map_err(|_| FlashInferError::invalid_argument("output length does not fit in i64"))?;
    let input_desc = contiguous_f32_desc(input_ptr, dims);
    let output_desc = SamplingTensor1DI32Desc {
        ptr: output_ptr as usize as *const c_void,
        len: output_len,
        stride: 1,
        device_id: dims.2,
    };
    let indices_desc = indices_ptr.map(|ptr| SamplingTensor1DI32Desc {
        ptr: ptr as usize as *const c_void,
        len: output_len,
        stride: 1,
        device_id: dims.2,
    });
    let random_params = SamplingRandomParams {
        deterministic: random.deterministic,
        seed: random.seed,
        offset: random.offset,
        seed_arr: seed_ptr.map(|ptr| SamplingTensor1DU64Desc {
            ptr: ptr as usize as *const c_void,
            len: random.seed_arr.map_or(0, |array| array.len()) as i64,
            stride: 1,
            device_id: dims.2,
        }),
        offset_arr: offset_ptr.map(|ptr| SamplingTensor1DU64Desc {
            ptr: ptr as usize as *const c_void,
            len: random.offset_arr.map_or(0, |array| array.len()) as i64,
            stride: 1,
            device_id: dims.2,
        }),
    };
    let sampling = SamplingParams {
        input: input_desc,
        output: output_desc,
        row_indices: indices_desc,
        random: random_params,
        stream: stream.cu_stream().cast(),
    };

    match (kernel, filter) {
        (SamplingKernel::SamplingFromProbs, CudarcSamplingFilter::None) => {
            sampling_from_probs(&sampling)
        }
        (SamplingKernel::SamplingFromLogits, CudarcSamplingFilter::None) => {
            sampling_from_logits(&sampling)
        }
        (SamplingKernel::TopPSamplingFromProbs, CudarcSamplingFilter::TopP { value, array }) => {
            top_p_sampling_from_probs(&TopPSamplingParams {
                sampling,
                top_p: value,
                top_p_arr: top_p_ptr.map(|ptr| SamplingTensor1DF32Desc {
                    ptr: ptr as usize as *const c_void,
                    len: array.map_or(0, |array| array.len()) as i64,
                    stride: 1,
                    device_id: dims.2,
                }),
            })
        }
        (SamplingKernel::TopKSamplingFromProbs, CudarcSamplingFilter::TopK { value, array }) => {
            top_k_sampling_from_probs(&TopKSamplingParams {
                sampling,
                top_k: value,
                top_k_arr: top_k_ptr.map(|ptr| SamplingTensor1DI32Desc {
                    ptr: ptr as usize as *const c_void,
                    len: array.map_or(0, |array| array.len()) as i64,
                    stride: 1,
                    device_id: dims.2,
                }),
            })
        }
        (SamplingKernel::MinPSamplingFromProbs, CudarcSamplingFilter::MinP { value, array }) => {
            min_p_sampling_from_probs(&MinPSamplingParams {
                sampling,
                min_p: value,
                min_p_arr: top_p_ptr.map(|ptr| SamplingTensor1DF32Desc {
                    ptr: ptr as usize as *const c_void,
                    len: array.map_or(0, |array| array.len()) as i64,
                    stride: 1,
                    device_id: dims.2,
                }),
            })
        }
        (
            SamplingKernel::TopKTopPSamplingFromProbs,
            CudarcSamplingFilter::TopKTopP {
                top_k,
                top_k_arr,
                top_p,
                top_p_arr,
            },
        ) => top_k_top_p_sampling_from_probs(&TopKTopPSamplingParams {
            sampling,
            top_k,
            top_k_arr: top_k_ptr.map(|ptr| SamplingTensor1DI32Desc {
                ptr: ptr as usize as *const c_void,
                len: top_k_arr.map_or(0, |array| array.len()) as i64,
                stride: 1,
                device_id: dims.2,
            }),
            top_p,
            top_p_arr: top_p_ptr.map(|ptr| SamplingTensor1DF32Desc {
                ptr: ptr as usize as *const c_void,
                len: top_p_arr.map_or(0, |array| array.len()) as i64,
                stride: 1,
                device_id: dims.2,
            }),
        }),
        _ => Err(FlashInferError::invalid_argument(
            "internal cudarc sampling operation mismatch",
        )),
    }
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
fn run_top_k_transform_cudarc<I, O, W>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    output: &mut O,
    workspace: &mut W,
    batch_size: usize,
    vocab_size: usize,
    top_k: i64,
    top_k_arr: Option<&cudarc::driver::CudaSlice<i32>>,
    kernel: SamplingKernel,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtrMut<f32>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    validate_flat_matrix_lengths(input.len(), output.len(), batch_size, vocab_size)?;
    if workspace.len() < SAMPLING_WORKSPACE_BYTES {
        return Err(FlashInferError::invalid_argument(format!(
            "workspace length ({}) must be at least {SAMPLING_WORKSPACE_BYTES}",
            workspace.len()
        )));
    }
    if let Some(array) = top_k_arr
        && array.len() != batch_size
    {
        return Err(FlashInferError::invalid_argument(format!(
            "top_k_arr length ({}) must equal batch_size ({batch_size})",
            array.len()
        )));
    }
    let workspace_len = workspace.len();
    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let (workspace_ptr, _workspace_sync) = workspace.device_ptr_mut(stream);
    let (array_ptr, _array_sync) = match top_k_arr {
        Some(array) => {
            let (ptr, sync) = array.device_ptr(stream);
            (Some(ptr), Some(sync))
        }
        None => (None, None),
    };
    let dims = cudarc_dims(stream, batch_size, vocab_size)?;
    let input_desc = contiguous_f32_desc(input_ptr, dims);
    let output_desc = contiguous_f32_desc(output_ptr, dims);
    let top_k_desc = array_ptr.map(|ptr| SamplingTensor1DI32Desc {
        ptr: ptr as usize as *const c_void,
        len: dims.0,
        stride: 1,
        device_id: dims.2,
    });
    let workspace_desc = SamplingWorkspaceDesc {
        ptr: workspace_ptr as usize as *const c_void,
        len_bytes: workspace_len as i64,
        device_id: dims.2,
    };
    match kernel {
        SamplingKernel::TopKRenormProbs => top_k_renorm_probs(&TopKRenormParams {
            probs: input_desc,
            output: output_desc,
            top_k,
            top_k_arr: top_k_desc,
            workspace: workspace_desc,
            stream: stream.cu_stream().cast(),
        }),
        SamplingKernel::TopKMaskLogits => top_k_mask_logits(&TopKMaskLogitsParams {
            logits: input_desc,
            output: output_desc,
            top_k,
            top_k_arr: top_k_desc,
            workspace: workspace_desc,
            stream: stream.cu_stream().cast(),
        }),
        _ => Err(FlashInferError::invalid_argument(
            "internal cudarc top-k transform mismatch",
        )),
    }
}

#[cfg(feature = "cudarc")]
fn validate_flat_matrix_lengths(
    input_len: usize,
    output_len: usize,
    rows: usize,
    cols: usize,
) -> Result<(), FlashInferError> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| FlashInferError::invalid_argument("rows * cols overflow"))?;
    if input_len != expected || output_len != expected {
        return Err(FlashInferError::invalid_argument(format!(
            "input and output lengths must both equal rows * cols ({expected}); got {input_len} and {output_len}"
        )));
    }
    Ok(())
}

#[cfg(feature = "cudarc")]
fn cudarc_dims(
    stream: &cudarc::driver::CudaStream,
    rows: usize,
    cols: usize,
) -> Result<(i64, i64, i32), FlashInferError> {
    let rows = i64::try_from(rows)
        .map_err(|_| FlashInferError::invalid_argument("rows does not fit in i64"))?;
    let cols = i64::try_from(cols)
        .map_err(|_| FlashInferError::invalid_argument("cols does not fit in i64"))?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;
    Ok((rows, cols, device_id))
}

#[cfg(feature = "cudarc")]
fn contiguous_f32_desc(
    ptr: cudarc::driver::sys::CUdeviceptr,
    dims: (i64, i64, i32),
) -> SamplingTensor2DF32Desc {
    SamplingTensor2DF32Desc {
        ptr: ptr as usize as *const c_void,
        rows: dims.0,
        cols: dims.1,
        stride_row: dims.1,
        stride_col: 1,
        device_id: dims.2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{
        KTVM_FFI_BOOL, KTVM_FFI_DL_TENSOR_PTR, KTVM_FFI_FLOAT, KTVM_FFI_INT, KTVM_FFI_NONE,
    };

    fn ptr() -> *const c_void {
        std::ptr::NonNull::<u8>::dangling().as_ptr().cast()
    }

    fn matrix(rows: i64, cols: i64) -> SamplingTensor2DF32Desc {
        SamplingTensor2DF32Desc {
            ptr: ptr(),
            rows,
            cols,
            stride_row: cols,
            stride_col: 1,
            device_id: 0,
        }
    }

    fn i32_vector(len: i64) -> SamplingTensor1DI32Desc {
        SamplingTensor1DI32Desc {
            ptr: ptr(),
            len,
            stride: 1,
            device_id: 0,
        }
    }

    fn u64_vector(len: i64) -> SamplingTensor1DU64Desc {
        SamplingTensor1DU64Desc {
            ptr: ptr(),
            len,
            stride: 1,
            device_id: 0,
        }
    }

    fn valid_sampling() -> SamplingParams {
        SamplingParams::new(
            matrix(2, 4),
            i32_vector(2),
            SamplingRandomParams::new(7, 11),
            std::ptr::null_mut(),
        )
    }

    #[test]
    fn validation_accepts_contiguous_fp32_i32_contract() {
        valid_sampling().validate().unwrap();
    }

    #[test]
    fn validation_rejects_non_contiguous_input() {
        let mut params = valid_sampling();
        params.input.stride_row = 8;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validation_requires_row_indices_for_expanded_output() {
        let mut params = valid_sampling();
        params.output.len = 3;
        assert!(params.validate().is_err());
        params.row_indices = Some(i32_vector(3));
        params.validate().unwrap();
    }

    #[test]
    fn validation_checks_parameter_array_batch() {
        let params = TopPSamplingParams::new(valid_sampling(), 0.9).with_top_p_arr(
            SamplingTensor1DF32Desc {
                ptr: ptr(),
                len: 1,
                stride: 1,
                device_id: 0,
            },
        );
        assert!(params.validate().is_err());
    }

    #[test]
    fn validation_checks_device_rng_array_pair_and_length() {
        let mut params = valid_sampling();
        params.random.seed_arr = Some(u64_vector(1));
        assert!(params.validate().is_err());

        params.random.offset_arr = Some(u64_vector(2));
        assert!(params.validate().is_err());

        params.random.seed_arr = Some(u64_vector(2));
        params.validate().unwrap();
    }

    #[test]
    fn validation_checks_workspace_size_and_device() {
        let params = SamplingSoftmaxParams {
            workspace: SamplingWorkspaceDesc {
                ptr: ptr(),
                len_bytes: (SAMPLING_WORKSPACE_BYTES - 1) as i64,
                device_id: 0,
            },
            logits: matrix(2, 4),
            output: matrix(2, 4),
            temperature: 1.0,
            temperature_arr: None,
            enable_pdl: false,
            stream: std::ptr::null_mut(),
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn basic_packer_matches_pinned_argument_order() {
        let mut matrix_shape = [2, 4];
        let mut matrix_strides = [4, 1];
        let input = tensor_2d_f32(matrix(2, 4), &mut matrix_shape, &mut matrix_strides);
        let mut output_shape = [2];
        let mut output_strides = [1];
        let output = tensor_1d(ptr(), 0, &mut output_shape, &mut output_strides, dl_i32());
        let args = pack_basic_sampling_args(
            &input,
            &output,
            None,
            None,
            None,
            SamplingRandomParams::new(7, 11),
        );
        let types: Vec<i32> = args.iter().map(|arg| arg.type_index).collect();
        assert_eq!(
            types,
            vec![
                KTVM_FFI_DL_TENSOR_PTR,
                KTVM_FFI_DL_TENSOR_PTR,
                KTVM_FFI_NONE,
                KTVM_FFI_BOOL,
                KTVM_FFI_NONE,
                KTVM_FFI_INT,
                KTVM_FFI_NONE,
                KTVM_FFI_INT,
            ]
        );
        // SAFETY: the packer wrote integer scalar payloads for these slots.
        assert_eq!(unsafe { args[5].value.v_int64 }, 7);
        // SAFETY: the packer wrote integer scalar payloads for these slots.
        assert_eq!(unsafe { args[7].value.v_int64 }, 11);
    }

    #[test]
    fn fused_top_k_top_p_packer_uses_double_top_k_abi() {
        let mut matrix_shape = [2, 4];
        let mut matrix_strides = [4, 1];
        let input = tensor_2d_f32(matrix(2, 4), &mut matrix_shape, &mut matrix_strides);
        let mut output_shape = [2];
        let mut output_strides = [1];
        let output = tensor_1d(ptr(), 0, &mut output_shape, &mut output_strides, dl_i32());
        let args = pack_filtered_sampling_args(
            SamplingKernel::TopKTopPSamplingFromProbs,
            &input,
            &output,
            None,
            None,
            None,
            3,
            0.8,
            None,
            None,
            SamplingRandomParams::new(1, 2),
        )
        .unwrap();
        assert_eq!(args.len(), 12);
        assert_eq!(args[4].type_index, KTVM_FFI_FLOAT);
        assert_eq!(args[6].type_index, KTVM_FFI_FLOAT);
        // SAFETY: the packer wrote floating-point payloads for these slots.
        assert_eq!(unsafe { args[4].value.v_float64 }, 3.0);
    }
}
