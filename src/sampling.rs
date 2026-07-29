//! Typed wrappers for the fixed FlashInfer sampling artifact.

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_CUDA, KDL_FLOAT, KDL_INT, KDL_UINT, TVMFFIAny, any_bool,
    any_dltensor_ptr, any_f64, any_i64, any_none, any_u64,
};
use crate::runtime::FlashInferRuntime;

/// Default workspace size used by FlashInfer's Python online-softmax wrapper.
pub const DEFAULT_SAMPLING_WORKSPACE_BYTES: usize = 1 << 20;

/// Contiguous rank-2 FP32 CUDA tensor descriptor with shape `[rows, cols]`.
#[derive(Debug, Clone, Copy)]
pub struct SamplingTensor2DF32Desc {
    /// Device pointer to the first element.
    pub ptr: *const c_void,
    /// Number of rows.
    pub rows: i64,
    /// Number of columns.
    pub cols: i64,
    /// Row stride in FP32 elements; must equal `cols`.
    pub stride_row: i64,
    /// Column stride in FP32 elements; must equal one.
    pub stride_col: i64,
    /// CUDA device ordinal.
    pub device_id: i32,
}

macro_rules! sampling_tensor_1d_desc {
    ($name:ident, $dtype:literal) => {
        #[doc = concat!("Contiguous rank-1 ", $dtype, " CUDA tensor descriptor with shape `[len]`.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            /// Device pointer to the first element.
            pub ptr: *const c_void,
            /// Number of elements.
            pub len: i64,
            /// Element stride; must equal one.
            pub stride: i64,
            /// CUDA device ordinal.
            pub device_id: i32,
        }
    };
}

sampling_tensor_1d_desc!(SamplingTensor1DF32Desc, "FP32");
sampling_tensor_1d_desc!(SamplingTensor1DI32Desc, "I32");
sampling_tensor_1d_desc!(SamplingTensor1DU64Desc, "U64");
sampling_tensor_1d_desc!(SamplingTensor1DU8Desc, "U8");

/// Arguments shared by the logits and probability sampling kernels.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    /// Input logits or probabilities, contiguous FP32 `[source_rows, vocab_size]`.
    pub input: SamplingTensor2DF32Desc,
    /// Sampled token IDs, contiguous I32 `[sample_rows]`.
    pub output: SamplingTensor1DI32Desc,
    /// Optional source-row mapping, contiguous I32 `[sample_rows]`.
    pub indices: Option<SamplingTensor1DI32Desc>,
    /// Whether to use FlashInfer's deterministic reduction.
    pub deterministic: bool,
    /// Optional Philox seeds.
    ///
    /// The stock 0.6.3 artifact consumes element zero. A subsequent kernel
    /// revision may consume one value per output row.
    pub seed_arr: Option<SamplingTensor1DU64Desc>,
    /// Scalar Philox seed used when `seed_arr` is absent.
    pub seed_val: u64,
    /// Optional Philox offsets, with the same shape contract as `seed_arr`.
    pub offset_arr: Option<SamplingTensor1DU64Desc>,
    /// Scalar Philox offset used when `offset_arr` is absent.
    pub offset_val: u64,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    pub stream: *mut c_void,
}

impl SamplingParams {
    /// Constructs sampling parameters with scalar Philox state and no row mapping.
    pub fn new(
        input: SamplingTensor2DF32Desc,
        output: SamplingTensor1DI32Desc,
        seed: u64,
        offset: u64,
        stream: *mut c_void,
    ) -> Self {
        Self {
            input,
            output,
            indices: None,
            deterministic: true,
            seed_arr: None,
            seed_val: seed,
            offset_arr: None,
            offset_val: offset,
            stream,
        }
    }

    /// Sets the optional source-row mapping.
    pub fn with_indices(mut self, indices: SamplingTensor1DI32Desc) -> Self {
        self.indices = Some(indices);
        self
    }

    /// Sets optional device-resident Philox seed and offset arrays.
    pub fn with_rng_arrays(
        mut self,
        seeds: SamplingTensor1DU64Desc,
        offsets: SamplingTensor1DU64Desc,
    ) -> Self {
        self.seed_arr = Some(seeds);
        self.offset_arr = Some(offsets);
        self
    }

    /// Selects deterministic or non-deterministic FlashInfer reductions.
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    /// Validates shapes, layouts, devices, and optional-array capacities.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_2d(self.input, "input")?;
        validate_i32_1d(self.output, "output")?;
        if self.output.len > self.input.rows {
            return invalid("output length must not exceed input rows");
        }
        if let Some(indices) = self.indices {
            validate_i32_1d(indices, "indices")?;
            require_len(indices.len, self.output.len, "indices")?;
            require_device(indices.device_id, self.input.device_id, "indices")?;
        } else {
            require_len(self.output.len, self.input.rows, "output")?;
        }
        require_device(self.output.device_id, self.input.device_id, "output")?;
        validate_rng_array(
            self.seed_arr,
            self.output.len,
            self.input.device_id,
            "seed_arr",
        )?;
        validate_rng_array(
            self.offset_arr,
            self.output.len,
            self.input.device_id,
            "offset_arr",
        )?;
        Ok(())
    }
}

/// Arguments for FlashInfer online softmax.
#[derive(Debug, Clone, Copy)]
pub struct SamplingSoftmaxParams {
    /// Caller-owned contiguous U8 workspace `[workspace_bytes]`.
    pub workspace: SamplingTensor1DU8Desc,
    /// Input logits, contiguous FP32 `[batch_size, vocab_size]`.
    pub logits: SamplingTensor2DF32Desc,
    /// Output probabilities, contiguous FP32 `[batch_size, vocab_size]`.
    pub output: SamplingTensor2DF32Desc,
    /// Optional temperatures, contiguous FP32 `[batch_size]`.
    pub temperatures: Option<SamplingTensor1DF32Desc>,
    /// Scalar temperature used when `temperatures` is absent.
    pub temperature: f64,
    /// Whether to enable programmatic dependent launch.
    pub enable_pdl: bool,
    /// CUDA stream (`cudaStream_t`) used for the asynchronous launch.
    pub stream: *mut c_void,
}

impl SamplingSoftmaxParams {
    /// Constructs online-softmax parameters with a scalar temperature.
    pub fn new(
        workspace: SamplingTensor1DU8Desc,
        logits: SamplingTensor2DF32Desc,
        output: SamplingTensor2DF32Desc,
        temperature: f64,
        stream: *mut c_void,
    ) -> Self {
        Self {
            workspace,
            logits,
            output,
            temperatures: None,
            temperature,
            enable_pdl: false,
            stream,
        }
    }

    /// Sets per-row temperatures.
    pub fn with_temperatures(mut self, temperatures: SamplingTensor1DF32Desc) -> Self {
        self.temperatures = Some(temperatures);
        self
    }

    /// Enables or disables programmatic dependent launch.
    pub fn with_enable_pdl(mut self, enable_pdl: bool) -> Self {
        self.enable_pdl = enable_pdl;
        self
    }

    /// Validates shapes, layouts, devices, workspace, and temperature values.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_u8_1d(self.workspace, "workspace")?;
        validate_2d(self.logits, "logits")?;
        validate_2d(self.output, "output")?;
        if self.output.rows != self.logits.rows || self.output.cols != self.logits.cols {
            return invalid("output shape must match logits shape");
        }
        require_device(self.output.device_id, self.logits.device_id, "output")?;
        require_device(self.workspace.device_id, self.logits.device_id, "workspace")?;
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return invalid("temperature must be finite and positive");
        }
        if let Some(temperatures) = self.temperatures {
            validate_f32_1d(temperatures, "temperatures")?;
            require_len(temperatures.len, self.logits.rows, "temperatures")?;
            require_device(
                temperatures.device_id,
                self.logits.device_id,
                "temperatures",
            )?;
        }
        Ok(())
    }
}

/// Arguments for top-k probability sampling.
#[derive(Debug, Clone, Copy)]
pub struct TopKSamplingParams {
    /// Shared probability-sampling arguments.
    pub sampling: SamplingParams,
    /// Optional per-source-row top-k values, contiguous I32 `[source_rows]`.
    pub top_k: Option<SamplingTensor1DI32Desc>,
    /// Scalar top-k value used when `top_k` is absent.
    pub top_k_val: i64,
}

impl TopKSamplingParams {
    /// Constructs scalar top-k sampling parameters.
    pub fn new(sampling: SamplingParams, top_k: i64) -> Self {
        Self {
            sampling,
            top_k: None,
            top_k_val: top_k,
        }
    }

    /// Sets per-source-row top-k values.
    pub fn with_top_k(mut self, top_k: SamplingTensor1DI32Desc) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Validates shared and top-k-specific arguments.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        self.sampling.validate()?;
        if self.top_k_val <= 0 || self.top_k_val > self.sampling.input.cols {
            return invalid("top_k_val must be in 1..=vocab_size");
        }
        validate_source_i32(self.top_k, &self.sampling, "top_k")
    }
}

/// Arguments for top-p probability sampling.
#[derive(Debug, Clone, Copy)]
pub struct TopPSamplingParams {
    /// Shared probability-sampling arguments.
    pub sampling: SamplingParams,
    /// Optional per-source-row top-p values, contiguous FP32 `[source_rows]`.
    pub top_p: Option<SamplingTensor1DF32Desc>,
    /// Scalar top-p value used when `top_p` is absent.
    pub top_p_val: f64,
}

impl TopPSamplingParams {
    /// Constructs scalar top-p sampling parameters.
    pub fn new(sampling: SamplingParams, top_p: f64) -> Self {
        Self {
            sampling,
            top_p: None,
            top_p_val: top_p,
        }
    }

    /// Sets per-source-row top-p values.
    pub fn with_top_p(mut self, top_p: SamplingTensor1DF32Desc) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Validates shared and top-p-specific arguments.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        self.sampling.validate()?;
        validate_probability(self.top_p_val, "top_p_val")?;
        validate_source_f32(self.top_p, &self.sampling, "top_p")
    }
}

/// Arguments for joint top-k/top-p probability sampling.
#[derive(Debug, Clone, Copy)]
pub struct TopKTopPSamplingParams {
    /// Shared probability-sampling arguments.
    pub sampling: SamplingParams,
    /// Optional per-source-row top-k values, contiguous I32 `[source_rows]`.
    pub top_k: Option<SamplingTensor1DI32Desc>,
    /// Scalar top-k value used when `top_k` is absent.
    pub top_k_val: i64,
    /// Optional per-source-row top-p values, contiguous FP32 `[source_rows]`.
    pub top_p: Option<SamplingTensor1DF32Desc>,
    /// Scalar top-p value used when `top_p` is absent.
    pub top_p_val: f64,
}

impl TopKTopPSamplingParams {
    /// Constructs scalar joint top-k/top-p sampling parameters.
    pub fn new(sampling: SamplingParams, top_k: i64, top_p: f64) -> Self {
        Self {
            sampling,
            top_k: None,
            top_k_val: top_k,
            top_p: None,
            top_p_val: top_p,
        }
    }

    /// Sets per-source-row top-k values.
    pub fn with_top_k(mut self, top_k: SamplingTensor1DI32Desc) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Sets per-source-row top-p values.
    pub fn with_top_p(mut self, top_p: SamplingTensor1DF32Desc) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Validates shared and joint-filter arguments.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        self.sampling.validate()?;
        if self.top_k_val <= 0 || self.top_k_val > self.sampling.input.cols {
            return invalid("top_k_val must be in 1..=vocab_size");
        }
        validate_probability(self.top_p_val, "top_p_val")?;
        validate_source_i32(self.top_k, &self.sampling, "top_k")?;
        validate_source_f32(self.top_p, &self.sampling, "top_p")
    }
}

/// Computes online softmax from FP32 `[batch, vocab]` logits into caller-owned output.
pub fn sampling_softmax(params: &SamplingSoftmaxParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: validation establishes the exported TVM-FFI tensor and stream contracts.
    unsafe { sampling_softmax_with_runtime(runtime, params) }
}

/// Samples I32 token IDs directly from FP32 `[batch, vocab]` logits.
pub fn sampling_from_logits(params: &SamplingParams) -> Result<(), FlashInferError> {
    params.validate()?;
    invoke_sampling(params, SamplingCall::Logits)
}

/// Samples I32 token IDs from FP32 `[batch, vocab]` probabilities.
pub fn sampling_from_probs(params: &SamplingParams) -> Result<(), FlashInferError> {
    params.validate()?;
    invoke_sampling(params, SamplingCall::Probs)
}

/// Samples I32 token IDs from probabilities after top-k filtering.
pub fn top_k_sampling_from_probs(params: &TopKSamplingParams) -> Result<(), FlashInferError> {
    params.validate()?;
    invoke_sampling(
        &params.sampling,
        SamplingCall::TopK {
            array: params.top_k,
            value: params.top_k_val,
        },
    )
}

/// Samples I32 token IDs from probabilities after top-p filtering.
pub fn top_p_sampling_from_probs(params: &TopPSamplingParams) -> Result<(), FlashInferError> {
    params.validate()?;
    invoke_sampling(
        &params.sampling,
        SamplingCall::TopP {
            array: params.top_p,
            value: params.top_p_val,
        },
    )
}

/// Samples I32 token IDs from probabilities with joint top-k/top-p filtering.
pub fn top_k_top_p_sampling_from_probs(
    params: &TopKTopPSamplingParams,
) -> Result<(), FlashInferError> {
    params.validate()?;
    invoke_sampling(
        &params.sampling,
        SamplingCall::TopKTopP {
            top_k: params.top_k,
            top_k_value: params.top_k_val,
            top_p: params.top_p,
            top_p_value: params.top_p_val,
        },
    )
}

/// Cudarc wrapper for online softmax over contiguous FP32 `[rows, cols]` tensors.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn sampling_softmax_cudarc(
    stream: &cudarc::driver::CudaStream,
    workspace: &mut cudarc::driver::CudaSlice<u8>,
    logits: &cudarc::driver::CudaSlice<f32>,
    output: &mut cudarc::driver::CudaSlice<f32>,
    rows: usize,
    cols: usize,
    temperatures: Option<&cudarc::driver::CudaSlice<f32>>,
    temperature: f64,
    enable_pdl: bool,
) -> Result<(), FlashInferError> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let elements = checked_elements(rows, cols)?;
    require_usize_len(logits.len(), elements, "logits")?;
    require_usize_len(output.len(), elements, "output")?;
    if let Some(temperatures) = temperatures {
        require_usize_len(temperatures.len(), rows, "temperatures")?;
    }
    if workspace.is_empty() {
        return invalid("workspace length must be positive");
    }

    let workspace_len = to_i64(workspace.len(), "workspace length")?;
    let (workspace_ptr, _workspace_sync) = workspace.device_ptr_mut(stream);
    let (logits_ptr, _logits_sync) = logits.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let temperature_ptr = temperatures.map(|value| value.device_ptr(stream));
    let device_id = cudarc_device_id(stream)?;
    let rows = to_i64(rows, "rows")?;
    let cols = to_i64(cols, "cols")?;

    let mut params = SamplingSoftmaxParams::new(
        SamplingTensor1DU8Desc {
            ptr: raw_ptr(workspace_ptr),
            len: workspace_len,
            stride: 1,
            device_id,
        },
        SamplingTensor2DF32Desc {
            ptr: raw_ptr(logits_ptr),
            rows,
            cols,
            stride_row: cols,
            stride_col: 1,
            device_id,
        },
        SamplingTensor2DF32Desc {
            ptr: raw_ptr(output_ptr),
            rows,
            cols,
            stride_row: cols,
            stride_col: 1,
            device_id,
        },
        temperature,
        stream.cu_stream().cast(),
    )
    .with_enable_pdl(enable_pdl);
    if let Some((ptr, _sync)) = temperature_ptr.as_ref() {
        params = params.with_temperatures(SamplingTensor1DF32Desc {
            ptr: raw_ptr(*ptr),
            len: rows,
            stride: 1,
            device_id,
        });
    }
    sampling_softmax(&params)
}

/// Cudarc wrapper that samples I32 token IDs directly from FP32 logits.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn sampling_from_logits_cudarc(
    stream: &cudarc::driver::CudaStream,
    input: &cudarc::driver::CudaSlice<f32>,
    output: &mut cudarc::driver::CudaSlice<i32>,
    rows: usize,
    cols: usize,
    indices: Option<&cudarc::driver::CudaSlice<i32>>,
    seeds: Option<&cudarc::driver::CudaSlice<u64>>,
    seed: u64,
    offsets: Option<&cudarc::driver::CudaSlice<u64>>,
    offset: u64,
    deterministic: bool,
) -> Result<(), FlashInferError> {
    sampling_cudarc(
        stream,
        input,
        output,
        rows,
        cols,
        indices,
        seeds,
        seed,
        offsets,
        offset,
        deterministic,
        CudarcSamplingCall::Logits,
    )
}

/// Cudarc wrapper that samples I32 token IDs from FP32 probabilities.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn sampling_from_probs_cudarc(
    stream: &cudarc::driver::CudaStream,
    input: &cudarc::driver::CudaSlice<f32>,
    output: &mut cudarc::driver::CudaSlice<i32>,
    rows: usize,
    cols: usize,
    indices: Option<&cudarc::driver::CudaSlice<i32>>,
    seeds: Option<&cudarc::driver::CudaSlice<u64>>,
    seed: u64,
    offsets: Option<&cudarc::driver::CudaSlice<u64>>,
    offset: u64,
    deterministic: bool,
) -> Result<(), FlashInferError> {
    sampling_cudarc(
        stream,
        input,
        output,
        rows,
        cols,
        indices,
        seeds,
        seed,
        offsets,
        offset,
        deterministic,
        CudarcSamplingCall::Probs,
    )
}

/// Cudarc wrapper for top-k probability sampling.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_k_sampling_from_probs_cudarc(
    stream: &cudarc::driver::CudaStream,
    input: &cudarc::driver::CudaSlice<f32>,
    output: &mut cudarc::driver::CudaSlice<i32>,
    rows: usize,
    cols: usize,
    indices: Option<&cudarc::driver::CudaSlice<i32>>,
    top_k: Option<&cudarc::driver::CudaSlice<i32>>,
    top_k_val: i64,
    seeds: Option<&cudarc::driver::CudaSlice<u64>>,
    seed: u64,
    offsets: Option<&cudarc::driver::CudaSlice<u64>>,
    offset: u64,
    deterministic: bool,
) -> Result<(), FlashInferError> {
    sampling_cudarc(
        stream,
        input,
        output,
        rows,
        cols,
        indices,
        seeds,
        seed,
        offsets,
        offset,
        deterministic,
        CudarcSamplingCall::TopK { top_k, top_k_val },
    )
}

/// Cudarc wrapper for top-p probability sampling.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_p_sampling_from_probs_cudarc(
    stream: &cudarc::driver::CudaStream,
    input: &cudarc::driver::CudaSlice<f32>,
    output: &mut cudarc::driver::CudaSlice<i32>,
    rows: usize,
    cols: usize,
    indices: Option<&cudarc::driver::CudaSlice<i32>>,
    top_p: Option<&cudarc::driver::CudaSlice<f32>>,
    top_p_val: f64,
    seeds: Option<&cudarc::driver::CudaSlice<u64>>,
    seed: u64,
    offsets: Option<&cudarc::driver::CudaSlice<u64>>,
    offset: u64,
    deterministic: bool,
) -> Result<(), FlashInferError> {
    sampling_cudarc(
        stream,
        input,
        output,
        rows,
        cols,
        indices,
        seeds,
        seed,
        offsets,
        offset,
        deterministic,
        CudarcSamplingCall::TopP { top_p, top_p_val },
    )
}

/// Cudarc wrapper for joint top-k/top-p probability sampling.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn top_k_top_p_sampling_from_probs_cudarc(
    stream: &cudarc::driver::CudaStream,
    input: &cudarc::driver::CudaSlice<f32>,
    output: &mut cudarc::driver::CudaSlice<i32>,
    rows: usize,
    cols: usize,
    indices: Option<&cudarc::driver::CudaSlice<i32>>,
    top_k: Option<&cudarc::driver::CudaSlice<i32>>,
    top_k_val: i64,
    top_p: Option<&cudarc::driver::CudaSlice<f32>>,
    top_p_val: f64,
    seeds: Option<&cudarc::driver::CudaSlice<u64>>,
    seed: u64,
    offsets: Option<&cudarc::driver::CudaSlice<u64>>,
    offset: u64,
    deterministic: bool,
) -> Result<(), FlashInferError> {
    sampling_cudarc(
        stream,
        input,
        output,
        rows,
        cols,
        indices,
        seeds,
        seed,
        offsets,
        offset,
        deterministic,
        CudarcSamplingCall::TopKTopP {
            top_k,
            top_k_val,
            top_p,
            top_p_val,
        },
    )
}

#[cfg(feature = "cudarc")]
#[derive(Clone, Copy)]
enum CudarcSamplingCall<'a> {
    Logits,
    Probs,
    TopK {
        top_k: Option<&'a cudarc::driver::CudaSlice<i32>>,
        top_k_val: i64,
    },
    TopP {
        top_p: Option<&'a cudarc::driver::CudaSlice<f32>>,
        top_p_val: f64,
    },
    TopKTopP {
        top_k: Option<&'a cudarc::driver::CudaSlice<i32>>,
        top_k_val: i64,
        top_p: Option<&'a cudarc::driver::CudaSlice<f32>>,
        top_p_val: f64,
    },
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
fn sampling_cudarc(
    stream: &cudarc::driver::CudaStream,
    input: &cudarc::driver::CudaSlice<f32>,
    output: &mut cudarc::driver::CudaSlice<i32>,
    rows: usize,
    cols: usize,
    indices: Option<&cudarc::driver::CudaSlice<i32>>,
    seeds: Option<&cudarc::driver::CudaSlice<u64>>,
    seed: u64,
    offsets: Option<&cudarc::driver::CudaSlice<u64>>,
    offset: u64,
    deterministic: bool,
    call: CudarcSamplingCall<'_>,
) -> Result<(), FlashInferError> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    require_usize_len(input.len(), checked_elements(rows, cols)?, "input")?;
    if output.is_empty() || output.len() > rows {
        return invalid("output length must be in 1..=rows");
    }
    if let Some(indices) = indices {
        require_usize_len(indices.len(), output.len(), "indices")?;
    } else {
        require_usize_len(output.len(), rows, "output")?;
    }
    if let Some(top_k) = match call {
        CudarcSamplingCall::TopK { top_k, .. } | CudarcSamplingCall::TopKTopP { top_k, .. } => {
            top_k
        }
        _ => None,
    } {
        require_usize_len(top_k.len(), rows, "top_k")?;
    }
    if let Some(top_p) = match call {
        CudarcSamplingCall::TopP { top_p, .. } | CudarcSamplingCall::TopKTopP { top_p, .. } => {
            top_p
        }
        _ => None,
    } {
        require_usize_len(top_p.len(), rows, "top_p")?;
    }

    let output_len = to_i64(output.len(), "output length")?;
    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let indices_ptr = indices.map(|value| value.device_ptr(stream));
    let seeds_ptr = seeds.map(|value| value.device_ptr(stream));
    let offsets_ptr = offsets.map(|value| value.device_ptr(stream));
    let top_k_ptr = match call {
        CudarcSamplingCall::TopK { top_k, .. } | CudarcSamplingCall::TopKTopP { top_k, .. } => {
            top_k.map(|value| value.device_ptr(stream))
        }
        _ => None,
    };
    let top_p_ptr = match call {
        CudarcSamplingCall::TopP { top_p, .. } | CudarcSamplingCall::TopKTopP { top_p, .. } => {
            top_p.map(|value| value.device_ptr(stream))
        }
        _ => None,
    };

    let device_id = cudarc_device_id(stream)?;
    let rows_i64 = to_i64(rows, "rows")?;
    let cols_i64 = to_i64(cols, "cols")?;
    let mut params = SamplingParams::new(
        SamplingTensor2DF32Desc {
            ptr: raw_ptr(input_ptr),
            rows: rows_i64,
            cols: cols_i64,
            stride_row: cols_i64,
            stride_col: 1,
            device_id,
        },
        SamplingTensor1DI32Desc {
            ptr: raw_ptr(output_ptr),
            len: output_len,
            stride: 1,
            device_id,
        },
        seed,
        offset,
        stream.cu_stream().cast(),
    )
    .with_deterministic(deterministic);
    if let Some((ptr, _sync)) = indices_ptr.as_ref() {
        params = params.with_indices(SamplingTensor1DI32Desc {
            ptr: raw_ptr(*ptr),
            len: output_len,
            stride: 1,
            device_id,
        });
    }
    if let (Some((seed_ptr, _)), Some((offset_ptr, _))) = (seeds_ptr.as_ref(), offsets_ptr.as_ref())
    {
        params = params.with_rng_arrays(
            SamplingTensor1DU64Desc {
                ptr: raw_ptr(*seed_ptr),
                len: to_i64(seeds.expect("checked Some").len(), "seed length")?,
                stride: 1,
                device_id,
            },
            SamplingTensor1DU64Desc {
                ptr: raw_ptr(*offset_ptr),
                len: to_i64(offsets.expect("checked Some").len(), "offset length")?,
                stride: 1,
                device_id,
            },
        );
    } else if seeds.is_some() || offsets.is_some() {
        return invalid("seeds and offsets must either both be present or both be absent");
    }

    match call {
        CudarcSamplingCall::Logits => sampling_from_logits(&params),
        CudarcSamplingCall::Probs => sampling_from_probs(&params),
        CudarcSamplingCall::TopK { top_k_val, .. } => {
            let mut filtered = TopKSamplingParams::new(params, top_k_val);
            if let Some((ptr, _sync)) = top_k_ptr.as_ref() {
                filtered = filtered.with_top_k(SamplingTensor1DI32Desc {
                    ptr: raw_ptr(*ptr),
                    len: rows_i64,
                    stride: 1,
                    device_id,
                });
            }
            top_k_sampling_from_probs(&filtered)
        }
        CudarcSamplingCall::TopP { top_p_val, .. } => {
            let mut filtered = TopPSamplingParams::new(params, top_p_val);
            if let Some((ptr, _sync)) = top_p_ptr.as_ref() {
                filtered = filtered.with_top_p(SamplingTensor1DF32Desc {
                    ptr: raw_ptr(*ptr),
                    len: rows_i64,
                    stride: 1,
                    device_id,
                });
            }
            top_p_sampling_from_probs(&filtered)
        }
        CudarcSamplingCall::TopKTopP {
            top_k_val,
            top_p_val,
            ..
        } => {
            let mut filtered = TopKTopPSamplingParams::new(params, top_k_val, top_p_val);
            if let Some((ptr, _sync)) = top_k_ptr.as_ref() {
                filtered = filtered.with_top_k(SamplingTensor1DI32Desc {
                    ptr: raw_ptr(*ptr),
                    len: rows_i64,
                    stride: 1,
                    device_id,
                });
            }
            if let Some((ptr, _sync)) = top_p_ptr.as_ref() {
                filtered = filtered.with_top_p(SamplingTensor1DF32Desc {
                    ptr: raw_ptr(*ptr),
                    len: rows_i64,
                    stride: 1,
                    device_id,
                });
            }
            top_k_top_p_sampling_from_probs(&filtered)
        }
    }
}

#[derive(Clone, Copy)]
enum SamplingCall {
    Logits,
    Probs,
    TopK {
        array: Option<SamplingTensor1DI32Desc>,
        value: i64,
    },
    TopP {
        array: Option<SamplingTensor1DF32Desc>,
        value: f64,
    },
    TopKTopP {
        top_k: Option<SamplingTensor1DI32Desc>,
        top_k_value: i64,
        top_p: Option<SamplingTensor1DF32Desc>,
        top_p_value: f64,
    },
}

fn invoke_sampling(params: &SamplingParams, call: SamplingCall) -> Result<(), FlashInferError> {
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: public entry points validate descriptors before reaching this call.
    unsafe { sampling_with_runtime(runtime, params, call) }
}

unsafe fn sampling_softmax_with_runtime(
    runtime: &FlashInferRuntime,
    params: &SamplingSoftmaxParams,
) -> Result<(), FlashInferError> {
    let mut workspace_shape = [params.workspace.len];
    let workspace = dl_tensor_1d(
        params.workspace.ptr,
        params.workspace.device_id,
        dl_dtype(KDL_UINT, 8),
        &mut workspace_shape,
    );
    let mut logits_shape = [params.logits.rows, params.logits.cols];
    let logits = dl_tensor_2d(
        params.logits.ptr,
        params.logits.device_id,
        dl_dtype(KDL_FLOAT, 32),
        &mut logits_shape,
    );
    let mut output_shape = [params.output.rows, params.output.cols];
    let output = dl_tensor_2d(
        params.output.ptr,
        params.output.device_id,
        dl_dtype(KDL_FLOAT, 32),
        &mut output_shape,
    );
    let mut temperature_shape = [params.logits.rows];
    let temperature = params.temperatures.map(|desc| {
        dl_tensor_1d(
            desc.ptr,
            desc.device_id,
            dl_dtype(KDL_FLOAT, 32),
            &mut temperature_shape,
        )
    });
    let args = [
        any_dltensor_ptr(&workspace),
        any_dltensor_ptr(&logits),
        any_dltensor_ptr(&output),
        optional_tensor(temperature.as_ref()),
        any_f64(params.temperature),
        any_bool(params.enable_pdl),
    ];
    let mut result = any_none();
    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous = unsafe { runtime.set_stream(params.logits.device_id, params.stream)? };
    let mut guard = StreamRestoreGuard::new(runtime, params.logits.device_id, previous);
    // SAFETY: argument order and scalar tags match `flashinfer_sampling_binding.cu::softmax`.
    let call_result =
        unsafe { runtime.call_sampling_softmax(args.as_ptr(), args.len() as i32, &mut result) };
    combine_restore(call_result, guard.restore_now())
}

unsafe fn sampling_with_runtime(
    runtime: &FlashInferRuntime,
    params: &SamplingParams,
    call: SamplingCall,
) -> Result<(), FlashInferError> {
    let mut input_shape = [params.input.rows, params.input.cols];
    let input = dl_tensor_2d(
        params.input.ptr,
        params.input.device_id,
        dl_dtype(KDL_FLOAT, 32),
        &mut input_shape,
    );
    let mut output_shape = [params.output.len];
    let output = dl_tensor_1d(
        params.output.ptr,
        params.output.device_id,
        dl_dtype(KDL_INT, 32),
        &mut output_shape,
    );
    let mut indices_shape = [params.output.len];
    let indices = params.indices.map(|desc| {
        dl_tensor_1d(
            desc.ptr,
            desc.device_id,
            dl_dtype(KDL_INT, 32),
            &mut indices_shape,
        )
    });
    let mut seed_shape = [params.seed_arr.map_or(1, |desc| desc.len)];
    let seeds = params.seed_arr.map(|desc| {
        dl_tensor_1d(
            desc.ptr,
            desc.device_id,
            dl_dtype(KDL_UINT, 64),
            &mut seed_shape,
        )
    });
    let mut offset_shape = [params.offset_arr.map_or(1, |desc| desc.len)];
    let offsets = params.offset_arr.map(|desc| {
        dl_tensor_1d(
            desc.ptr,
            desc.device_id,
            dl_dtype(KDL_UINT, 64),
            &mut offset_shape,
        )
    });
    let mut top_k_shape = [params.input.rows];
    let mut top_p_shape = [params.input.rows];
    let top_k_tensor = match call {
        SamplingCall::TopK { array, .. } | SamplingCall::TopKTopP { top_k: array, .. } => array
            .map(|desc| {
                dl_tensor_1d(
                    desc.ptr,
                    desc.device_id,
                    dl_dtype(KDL_INT, 32),
                    &mut top_k_shape,
                )
            }),
        _ => None,
    };
    let top_p_tensor = match call {
        SamplingCall::TopP { array, .. } | SamplingCall::TopKTopP { top_p: array, .. } => array
            .map(|desc| {
                dl_tensor_1d(
                    desc.ptr,
                    desc.device_id,
                    dl_dtype(KDL_FLOAT, 32),
                    &mut top_p_shape,
                )
            }),
        _ => None,
    };
    let common_tail = [
        any_bool(params.deterministic),
        optional_tensor(seeds.as_ref()),
        any_u64(params.seed_val),
        optional_tensor(offsets.as_ref()),
        any_u64(params.offset_val),
    ];
    let mut args = Vec::with_capacity(13);
    args.extend([
        any_dltensor_ptr(&input),
        any_dltensor_ptr(&output),
        optional_tensor(indices.as_ref()),
    ]);
    match call {
        SamplingCall::Logits | SamplingCall::Probs => {}
        SamplingCall::TopK { value, .. } => {
            args.push(optional_tensor(top_k_tensor.as_ref()));
            args.push(any_i64(value));
        }
        SamplingCall::TopP { value, .. } => {
            args.push(optional_tensor(top_p_tensor.as_ref()));
            args.push(any_f64(value));
        }
        SamplingCall::TopKTopP {
            top_k_value,
            top_p_value,
            ..
        } => {
            args.push(optional_tensor(top_k_tensor.as_ref()));
            // The exported 0.6.3 binding declares this scalar as `double`.
            args.push(any_f64(top_k_value as f64));
            args.push(optional_tensor(top_p_tensor.as_ref()));
            args.push(any_f64(top_p_value));
        }
    }
    args.extend(common_tail);

    let mut result = any_none();
    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous = unsafe { runtime.set_stream(params.input.device_id, params.stream)? };
    let mut guard = StreamRestoreGuard::new(runtime, params.input.device_id, previous);
    // SAFETY: argument order and scalar tags match the selected fixed sampling export.
    let call_result = unsafe {
        match call {
            SamplingCall::Logits => {
                runtime.call_sampling_from_logits(args.as_ptr(), args.len() as i32, &mut result)
            }
            SamplingCall::Probs => {
                runtime.call_sampling_from_probs(args.as_ptr(), args.len() as i32, &mut result)
            }
            SamplingCall::TopK { .. } => runtime.call_top_k_sampling_from_probs(
                args.as_ptr(),
                args.len() as i32,
                &mut result,
            ),
            SamplingCall::TopP { .. } => runtime.call_top_p_sampling_from_probs(
                args.as_ptr(),
                args.len() as i32,
                &mut result,
            ),
            SamplingCall::TopKTopP { .. } => runtime.call_top_k_top_p_sampling_from_probs(
                args.as_ptr(),
                args.len() as i32,
                &mut result,
            ),
        }
    };
    combine_restore(call_result, guard.restore_now())
}

fn dl_dtype(code: u8, bits: u8) -> DLDataType {
    DLDataType {
        code,
        bits,
        lanes: 1,
    }
}

fn dl_tensor_1d(
    ptr: *const c_void,
    device_id: i32,
    dtype: DLDataType,
    shape: &mut [i64; 1],
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
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    }
}

fn dl_tensor_2d(
    ptr: *const c_void,
    device_id: i32,
    dtype: DLDataType,
    shape: &mut [i64; 2],
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
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    }
}

fn optional_tensor(tensor: Option<&DLTensor>) -> TVMFFIAny {
    tensor.map_or_else(any_none, |tensor| any_dltensor_ptr(tensor))
}

fn validate_2d(desc: SamplingTensor2DF32Desc, name: &str) -> Result<(), FlashInferError> {
    check_ptr(desc.ptr, name)?;
    if desc.rows <= 0 || desc.cols <= 0 {
        return invalid(format!("{name} shape must be positive"));
    }
    if desc.stride_col != 1 || desc.stride_row != desc.cols {
        return invalid(format!("{name} must be contiguous"));
    }
    Ok(())
}

fn validate_f32_1d(desc: SamplingTensor1DF32Desc, name: &str) -> Result<(), FlashInferError> {
    validate_1d(desc.ptr, desc.len, desc.stride, name)
}

fn validate_i32_1d(desc: SamplingTensor1DI32Desc, name: &str) -> Result<(), FlashInferError> {
    validate_1d(desc.ptr, desc.len, desc.stride, name)
}

fn validate_u8_1d(desc: SamplingTensor1DU8Desc, name: &str) -> Result<(), FlashInferError> {
    validate_1d(desc.ptr, desc.len, desc.stride, name)
}

fn validate_1d(
    ptr: *const c_void,
    len: i64,
    stride: i64,
    name: &str,
) -> Result<(), FlashInferError> {
    check_ptr(ptr, name)?;
    if len <= 0 {
        return invalid(format!("{name} length must be positive"));
    }
    if stride != 1 {
        return invalid(format!("{name} must be contiguous"));
    }
    Ok(())
}

fn validate_rng_array(
    desc: Option<SamplingTensor1DU64Desc>,
    output_rows: i64,
    device_id: i32,
    name: &str,
) -> Result<(), FlashInferError> {
    if let Some(desc) = desc {
        validate_1d(desc.ptr, desc.len, desc.stride, name)?;
        if desc.len != 1 && desc.len != output_rows {
            return invalid(format!("{name} length must be one or output length"));
        }
        require_device(desc.device_id, device_id, name)?;
    }
    Ok(())
}

fn validate_source_i32(
    desc: Option<SamplingTensor1DI32Desc>,
    sampling: &SamplingParams,
    name: &str,
) -> Result<(), FlashInferError> {
    if let Some(desc) = desc {
        validate_i32_1d(desc, name)?;
        require_len(desc.len, sampling.input.rows, name)?;
        require_device(desc.device_id, sampling.input.device_id, name)?;
    }
    Ok(())
}

fn validate_source_f32(
    desc: Option<SamplingTensor1DF32Desc>,
    sampling: &SamplingParams,
    name: &str,
) -> Result<(), FlashInferError> {
    if let Some(desc) = desc {
        validate_f32_1d(desc, name)?;
        require_len(desc.len, sampling.input.rows, name)?;
        require_device(desc.device_id, sampling.input.device_id, name)?;
    }
    Ok(())
}

fn validate_probability(value: f64, name: &str) -> Result<(), FlashInferError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0 {
        return invalid(format!("{name} must be in (0, 1]"));
    }
    Ok(())
}

fn check_ptr(ptr: *const c_void, name: &str) -> Result<(), FlashInferError> {
    if ptr.is_null() {
        return invalid(format!("{name} pointer is null"));
    }
    Ok(())
}

fn require_len(actual: i64, expected: i64, name: &str) -> Result<(), FlashInferError> {
    if actual != expected {
        return invalid(format!("{name} length ({actual}) must equal {expected}"));
    }
    Ok(())
}

fn require_device(actual: i32, expected: i32, name: &str) -> Result<(), FlashInferError> {
    if actual != expected {
        return invalid(format!("{name} must be on CUDA device {expected}"));
    }
    Ok(())
}

#[cfg(feature = "cudarc")]
fn checked_elements(rows: usize, cols: usize) -> Result<usize, FlashInferError> {
    if rows == 0 || cols == 0 {
        return invalid("rows and cols must be positive");
    }
    rows.checked_mul(cols)
        .ok_or_else(|| FlashInferError::invalid_argument("rows * cols overflow"))
}

#[cfg(feature = "cudarc")]
fn require_usize_len(actual: usize, expected: usize, name: &str) -> Result<(), FlashInferError> {
    if actual != expected {
        return invalid(format!("{name} length ({actual}) must equal {expected}"));
    }
    Ok(())
}

#[cfg(feature = "cudarc")]
fn to_i64(value: usize, name: &str) -> Result<i64, FlashInferError> {
    i64::try_from(value)
        .map_err(|_| FlashInferError::invalid_argument(format!("{name} does not fit in i64")))
}

#[cfg(feature = "cudarc")]
fn cudarc_device_id(stream: &cudarc::driver::CudaStream) -> Result<i32, FlashInferError> {
    i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))
}

#[cfg(feature = "cudarc")]
fn raw_ptr(ptr: u64) -> *const c_void {
    ptr as usize as *const c_void
}

fn invalid<T>(message: impl Into<String>) -> Result<T, FlashInferError> {
    Err(FlashInferError::invalid_argument(message))
}

fn combine_restore(
    call: Result<(), FlashInferError>,
    restore: Result<(), FlashInferError>,
) -> Result<(), FlashInferError> {
    match (call, restore) {
        (Err(error), _) | (Ok(()), Err(error)) => Err(error),
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
        // SAFETY: previous_stream came from TVMFFIEnvSetStream for this device.
        unsafe {
            self.runtime
                .restore_stream(self.device_id, self.previous_stream)
        }
    }
}

impl Drop for StreamRestoreGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            // SAFETY: best-effort restore for an early-return path.
            let _ = unsafe {
                self.runtime
                    .restore_stream(self.device_id, self.previous_stream)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr() -> *const c_void {
        1usize as *const c_void
    }

    fn sampling() -> SamplingParams {
        SamplingParams::new(
            SamplingTensor2DF32Desc {
                ptr: ptr(),
                rows: 4,
                cols: 16,
                stride_row: 16,
                stride_col: 1,
                device_id: 0,
            },
            SamplingTensor1DI32Desc {
                ptr: ptr(),
                len: 4,
                stride: 1,
                device_id: 0,
            },
            7,
            0,
            std::ptr::null_mut(),
        )
    }

    #[test]
    fn sampling_validation_supports_indexed_subsets() {
        let mut params = sampling();
        params.output.len = 2;
        params.indices = Some(SamplingTensor1DI32Desc {
            ptr: ptr(),
            len: 2,
            stride: 1,
            device_id: 0,
        });
        assert!(params.validate().is_ok());
    }

    #[test]
    fn sampling_validation_rejects_non_contiguous_input() {
        let mut params = sampling();
        params.input.stride_row = 17;
        assert!(params.validate().is_err());
    }

    #[test]
    fn filter_validation_checks_ranges_and_source_shapes() {
        assert!(TopKSamplingParams::new(sampling(), 17).validate().is_err());
        assert!(TopPSamplingParams::new(sampling(), 0.0).validate().is_err());

        let mut top_k = TopKSamplingParams::new(sampling(), 4);
        top_k.top_k = Some(SamplingTensor1DI32Desc {
            ptr: ptr(),
            len: 2,
            stride: 1,
            device_id: 0,
        });
        assert!(top_k.validate().is_err());
    }
}
