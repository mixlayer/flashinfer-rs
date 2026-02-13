use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_CUDA, KDL_FLOAT, KDL_UINT, TVMFFIAny, any_dltensor_ptr,
    any_f64, any_i64, any_none,
};
use crate::norm::DType;
use crate::runtime::FlashInferRuntime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MhaMaskMode {
    NonCausal,
    Causal,
    Custom,
    MultiItemScoring,
}

impl MhaMaskMode {
    fn as_i64(self) -> i64 {
        match self {
            MhaMaskMode::NonCausal => 0,
            MhaMaskMode::Causal => 1,
            MhaMaskMode::Custom => 2,
            MhaMaskMode::MultiItemScoring => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MhaQkvLayout {
    Nhd,
    Hnd,
}

impl MhaQkvLayout {
    fn as_i64(self) -> i64 {
        match self {
            MhaQkvLayout::Nhd => 0,
            MhaQkvLayout::Hnd => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MhaPosEncodingMode {
    None,
    RoPELlama,
    ALiBi,
}

impl MhaPosEncodingMode {
    fn as_i64(self) -> i64 {
        match self {
            MhaPosEncodingMode::None => 0,
            MhaPosEncodingMode::RoPELlama => 1,
            MhaPosEncodingMode::ALiBi => 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MhaTensor1DU8Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct MhaTensor1DF32Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct MhaTensor2DF32Desc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct MhaTensor3DDesc {
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
pub struct MhaSinglePrefillParams {
    /// Query tensor, rank-3: `[qo_len, num_qo_heads, head_dim_qk]`.
    ///
    /// Cross-reference:
    /// `flashinfer/flashinfer/prefill.py::single_prefill_with_kv_cache`.
    pub q: MhaTensor3DDesc,
    /// Key tensor, rank-3:
    /// - `NHD` layout: `[kv_len, num_kv_heads, head_dim_qk]`
    /// - `HND` layout: `[num_kv_heads, kv_len, head_dim_qk]`.
    pub k: MhaTensor3DDesc,
    /// Value tensor, rank-3:
    /// - `NHD` layout: `[kv_len, num_kv_heads, head_dim_vo]`
    /// - `HND` layout: `[num_kv_heads, kv_len, head_dim_vo]`.
    pub v: MhaTensor3DDesc,
    /// Temporary workspace buffer in bytes, rank-1 u8: `[workspace_bytes]`.
    pub tmp: MhaTensor1DU8Desc,
    /// Output tensor, rank-3: `[qo_len, num_qo_heads, head_dim_vo]`.
    pub out: MhaTensor3DDesc,
    /// Optional log-sum-exp output, rank-2 f32: `[qo_len, num_qo_heads]`.
    pub lse: Option<MhaTensor2DF32Desc>,
    /// Optional packed custom mask, rank-1 u8.
    ///
    /// This follows FlashInfer's packed-mask path (`packed_custom_mask` in Python API).
    pub custom_mask: Option<MhaTensor1DU8Desc>,
    /// Optional ALiBi slopes, rank-1 f32: `[num_qo_heads]`.
    pub alibi_slopes: Option<MhaTensor1DF32Desc>,
    /// Mask mode enum (`NON_CAUSAL`, `CAUSAL`, `CUSTOM`, `MULTI_ITEM_SCORING`).
    pub mask_mode: MhaMaskMode,
    /// KV tensor layout enum (`NHD` or `HND`).
    pub kv_layout: MhaQkvLayout,
    /// Positional encoding mode enum (`NONE`, `ROPE_LLAMA`, `ALIBI`).
    pub pos_encoding_mode: MhaPosEncodingMode,
    /// Left sliding-window size; `-1` disables sliding window.
    pub window_left: i64,
    /// Logits soft cap value; `> 0` enables capping.
    pub logits_soft_cap: f64,
    /// Softmax scale; default is `1 / sqrt(head_dim_qk)`.
    pub sm_scale: f64,
    /// RoPE interpolation scale (`1.0` default).
    pub rope_scale: f64,
    /// RoPE theta base (`1e4` default).
    pub rope_theta: f64,
    /// Whether to enable fp16 QK reduction in compatible kernels.
    pub use_fp16_qk_reduction: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl MhaSinglePrefillParams {
    pub fn new(
        q: MhaTensor3DDesc,
        k: MhaTensor3DDesc,
        v: MhaTensor3DDesc,
        tmp: MhaTensor1DU8Desc,
        out: MhaTensor3DDesc,
        stream: *mut c_void,
    ) -> Self {
        let sm_scale = if q.dim2 > 0 {
            1.0 / (q.dim2 as f64).sqrt()
        } else {
            1.0
        };
        Self {
            q,
            k,
            v,
            tmp,
            out,
            lse: None,
            custom_mask: None,
            alibi_slopes: None,
            mask_mode: MhaMaskMode::NonCausal,
            kv_layout: MhaQkvLayout::Nhd,
            pos_encoding_mode: MhaPosEncodingMode::None,
            window_left: -1,
            logits_soft_cap: 0.0,
            sm_scale,
            rope_scale: 1.0,
            rope_theta: 1e4,
            use_fp16_qk_reduction: false,
            stream,
        }
    }

    pub fn with_lse(mut self, lse: MhaTensor2DF32Desc) -> Self {
        self.lse = Some(lse);
        self
    }

    pub fn with_custom_mask(mut self, custom_mask: MhaTensor1DU8Desc) -> Self {
        self.custom_mask = Some(custom_mask);
        self
    }

    pub fn with_alibi_slopes(mut self, alibi_slopes: MhaTensor1DF32Desc) -> Self {
        self.alibi_slopes = Some(alibi_slopes);
        self
    }

    pub fn with_mask_mode(mut self, mask_mode: MhaMaskMode) -> Self {
        self.mask_mode = mask_mode;
        self
    }

    pub fn with_kv_layout(mut self, kv_layout: MhaQkvLayout) -> Self {
        self.kv_layout = kv_layout;
        self
    }

    pub fn with_pos_encoding_mode(mut self, pos_encoding_mode: MhaPosEncodingMode) -> Self {
        self.pos_encoding_mode = pos_encoding_mode;
        self
    }

    pub fn with_window_left(mut self, window_left: i64) -> Self {
        self.window_left = window_left;
        self
    }

    pub fn with_logits_soft_cap(mut self, logits_soft_cap: f64) -> Self {
        self.logits_soft_cap = logits_soft_cap;
        self
    }

    pub fn with_sm_scale(mut self, sm_scale: f64) -> Self {
        self.sm_scale = sm_scale;
        self
    }

    pub fn with_rope_scale(mut self, rope_scale: f64) -> Self {
        self.rope_scale = rope_scale;
        self
    }

    pub fn with_rope_theta(mut self, rope_theta: f64) -> Self {
        self.rope_theta = rope_theta;
        self
    }

    pub fn with_fp16_qk_reduction(mut self, use_fp16_qk_reduction: bool) -> Self {
        self.use_fp16_qk_reduction = use_fp16_qk_reduction;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.q.ptr, "q")?;
        check_non_null(self.k.ptr, "k")?;
        check_non_null(self.v.ptr, "v")?;
        check_non_null(self.tmp.ptr, "tmp")?;
        check_non_null(self.out.ptr, "out")?;

        check_positive("q.dim0", self.q.dim0)?;
        check_positive("q.dim1", self.q.dim1)?;
        check_positive("q.dim2", self.q.dim2)?;
        check_positive("k.dim0", self.k.dim0)?;
        check_positive("k.dim1", self.k.dim1)?;
        check_positive("k.dim2", self.k.dim2)?;
        check_positive("v.dim0", self.v.dim0)?;
        check_positive("v.dim1", self.v.dim1)?;
        check_positive("v.dim2", self.v.dim2)?;
        check_positive("out.dim0", self.out.dim0)?;
        check_positive("out.dim1", self.out.dim1)?;
        check_positive("out.dim2", self.out.dim2)?;
        check_positive("tmp.len", self.tmp.len)?;

        if self.tmp.stride != 1 {
            return Err(FlashInferError::invalid_argument("tmp stride must be 1"));
        }
        if self.q.stride2 != 1
            || self.k.stride2 != 1
            || self.v.stride2 != 1
            || self.out.stride2 != 1
        {
            return Err(FlashInferError::invalid_argument(
                "q/k/v/out last-dimension stride must be 1",
            ));
        }

        if self.q.dtype != self.k.dtype
            || self.k.dtype != self.v.dtype
            || self.q.dtype != self.out.dtype
        {
            return Err(FlashInferError::invalid_argument(
                "q/k/v/out dtype mismatch",
            ));
        }
        if self.use_fp16_qk_reduction && self.q.dtype != DType::F16 {
            return Err(FlashInferError::invalid_argument(
                "use_fp16_qk_reduction requires dtype F16",
            ));
        }

        let qo_len = self.q.dim0;
        let num_qo_heads = self.q.dim1;
        let head_dim_qk = self.q.dim2;
        let head_dim_vo = self.v.dim2;

        if self.k.dim2 != head_dim_qk {
            return Err(FlashInferError::invalid_argument(
                "k head_dim_qk must match q",
            ));
        }
        if self.out.dim2 != head_dim_vo {
            return Err(FlashInferError::invalid_argument(
                "out head_dim_vo must match v",
            ));
        }
        if self.out.dim0 != qo_len || self.out.dim1 != num_qo_heads {
            return Err(FlashInferError::invalid_argument(
                "out shape must be [qo_len, num_qo_heads, head_dim_vo]",
            ));
        }

        let (kv_len_k, num_kv_heads_k) = match self.kv_layout {
            MhaQkvLayout::Nhd => (self.k.dim0, self.k.dim1),
            MhaQkvLayout::Hnd => (self.k.dim1, self.k.dim0),
        };
        let (kv_len_v, num_kv_heads_v) = match self.kv_layout {
            MhaQkvLayout::Nhd => (self.v.dim0, self.v.dim1),
            MhaQkvLayout::Hnd => (self.v.dim1, self.v.dim0),
        };
        if kv_len_k != kv_len_v || num_kv_heads_k != num_kv_heads_v {
            return Err(FlashInferError::invalid_argument(
                "k and v layout dimensions must agree",
            ));
        }
        if num_qo_heads % num_kv_heads_k != 0 {
            return Err(FlashInferError::invalid_argument(
                "num_qo_heads must be divisible by num_kv_heads",
            ));
        }

        if self.q.device_id != self.k.device_id
            || self.q.device_id != self.v.device_id
            || self.q.device_id != self.out.device_id
            || self.q.device_id != self.tmp.device_id
        {
            return Err(FlashInferError::invalid_argument(
                "all required tensors must be on the same CUDA device",
            ));
        }

        if self.window_left < -1 {
            return Err(FlashInferError::invalid_argument(
                "window_left must be -1 or >= 0",
            ));
        }
        if !self.logits_soft_cap.is_finite() {
            return Err(FlashInferError::invalid_argument(
                "logits_soft_cap must be finite",
            ));
        }
        if !self.sm_scale.is_finite() || self.sm_scale <= 0.0 {
            return Err(FlashInferError::invalid_argument(
                "sm_scale must be finite and > 0",
            ));
        }
        if !self.rope_scale.is_finite() || self.rope_scale <= 0.0 {
            return Err(FlashInferError::invalid_argument(
                "rope_scale must be finite and > 0",
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(FlashInferError::invalid_argument(
                "rope_theta must be finite and > 0",
            ));
        }

        if let Some(lse) = self.lse {
            check_non_null(lse.ptr, "lse")?;
            check_positive("lse.rows", lse.rows)?;
            check_positive("lse.cols", lse.cols)?;
            if lse.rows != qo_len || lse.cols != num_qo_heads {
                return Err(FlashInferError::invalid_argument(
                    "lse shape must be [qo_len, num_qo_heads]",
                ));
            }
            if lse.stride_col != 1 || lse.stride_row != lse.cols {
                return Err(FlashInferError::invalid_argument(
                    "lse must be contiguous row-major",
                ));
            }
            if lse.device_id != self.q.device_id {
                return Err(FlashInferError::invalid_argument(
                    "lse must be on the same CUDA device",
                ));
            }
        }

        if let Some(custom_mask) = self.custom_mask {
            check_non_null(custom_mask.ptr, "custom_mask")?;
            check_positive("custom_mask.len", custom_mask.len)?;
            if custom_mask.stride != 1 {
                return Err(FlashInferError::invalid_argument(
                    "custom_mask stride must be 1",
                ));
            }
            if custom_mask.device_id != self.q.device_id {
                return Err(FlashInferError::invalid_argument(
                    "custom_mask must be on the same CUDA device",
                ));
            }
        }
        if self.mask_mode == MhaMaskMode::Custom && self.custom_mask.is_none() {
            return Err(FlashInferError::invalid_argument(
                "custom_mask is required when mask_mode is Custom",
            ));
        }

        if let Some(alibi_slopes) = self.alibi_slopes {
            check_non_null(alibi_slopes.ptr, "alibi_slopes")?;
            check_positive("alibi_slopes.len", alibi_slopes.len)?;
            if alibi_slopes.len != num_qo_heads {
                return Err(FlashInferError::invalid_argument(
                    "alibi_slopes length must equal num_qo_heads",
                ));
            }
            if alibi_slopes.stride != 1 {
                return Err(FlashInferError::invalid_argument(
                    "alibi_slopes stride must be 1",
                ));
            }
            if alibi_slopes.device_id != self.q.device_id {
                return Err(FlashInferError::invalid_argument(
                    "alibi_slopes must be on the same CUDA device",
                ));
            }
        }
        if self.pos_encoding_mode == MhaPosEncodingMode::ALiBi && self.alibi_slopes.is_none() {
            return Err(FlashInferError::invalid_argument(
                "alibi_slopes is required when pos_encoding_mode is ALiBi",
            ));
        }

        Ok(())
    }

    fn kernel_uri(&self) -> String {
        format!(
            "single_prefill_with_kv_cache_dtype_q_{}_dtype_kv_{}_dtype_o_{}_head_dim_qk_{}_head_dim_vo_{}_posenc_{}_use_swa_{}_use_logits_cap_{}_f16qk_{}",
            dtype_filename(self.q.dtype),
            dtype_filename(self.k.dtype),
            dtype_filename(self.out.dtype),
            self.q.dim2,
            self.v.dim2,
            self.pos_encoding_mode.as_i64(),
            bool_name(self.window_left >= 0),
            bool_name(self.logits_soft_cap > 0.0),
            bool_name(self.use_fp16_qk_reduction),
        )
    }
}

pub fn mha_single_prefill(params: &MhaSinglePrefillParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { mha_single_prefill_with_runtime(runtime, params) }
}

unsafe fn mha_single_prefill_with_runtime(
    runtime: &FlashInferRuntime,
    params: &MhaSinglePrefillParams,
) -> Result<(), FlashInferError> {
    let kernel_uri = params.kernel_uri();

    let mut q_shape = [params.q.dim0, params.q.dim1, params.q.dim2];
    let mut q_strides = [params.q.stride0, params.q.stride1, params.q.stride2];
    let q_tensor = DLTensor {
        data: params.q.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.q.device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_norm_dtype(params.q.dtype),
        shape: q_shape.as_mut_ptr(),
        strides: q_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut k_shape = [params.k.dim0, params.k.dim1, params.k.dim2];
    let mut k_strides = [params.k.stride0, params.k.stride1, params.k.stride2];
    let k_tensor = DLTensor {
        data: params.k.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.k.device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_norm_dtype(params.k.dtype),
        shape: k_shape.as_mut_ptr(),
        strides: k_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut v_shape = [params.v.dim0, params.v.dim1, params.v.dim2];
    let mut v_strides = [params.v.stride0, params.v.stride1, params.v.stride2];
    let v_tensor = DLTensor {
        data: params.v.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.v.device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_norm_dtype(params.v.dtype),
        shape: v_shape.as_mut_ptr(),
        strides: v_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut tmp_shape = [params.tmp.len];
    let mut tmp_strides = [params.tmp.stride];
    let tmp_tensor = DLTensor {
        data: params.tmp.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.tmp.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_u8(),
        shape: tmp_shape.as_mut_ptr(),
        strides: tmp_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut out_shape = [params.out.dim0, params.out.dim1, params.out.dim2];
    let mut out_strides = [params.out.stride0, params.out.stride1, params.out.stride2];
    let out_tensor = DLTensor {
        data: params.out.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.out.device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_norm_dtype(params.out.dtype),
        shape: out_shape.as_mut_ptr(),
        strides: out_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut lse_shape = [0_i64; 2];
    let mut lse_strides = [0_i64; 2];
    let lse_tensor = params.lse.map(|lse| {
        lse_shape = [lse.rows, lse.cols];
        lse_strides = [lse.stride_row, lse.stride_col];
        DLTensor {
            data: lse.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: lse.device_id,
            },
            ndim: 2,
            dtype: dl_dtype_f32(),
            shape: lse_shape.as_mut_ptr(),
            strides: lse_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    });

    let mut custom_mask_shape = [0_i64; 1];
    let mut custom_mask_strides = [0_i64; 1];
    let custom_mask_tensor = params.custom_mask.map(|custom_mask| {
        custom_mask_shape = [custom_mask.len];
        custom_mask_strides = [custom_mask.stride];
        DLTensor {
            data: custom_mask.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: custom_mask.device_id,
            },
            ndim: 1,
            dtype: dl_dtype_u8(),
            shape: custom_mask_shape.as_mut_ptr(),
            strides: custom_mask_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    });

    let mut alibi_shape = [0_i64; 1];
    let mut alibi_strides = [0_i64; 1];
    let alibi_tensor = params.alibi_slopes.map(|alibi| {
        alibi_shape = [alibi.len];
        alibi_strides = [alibi.stride];
        DLTensor {
            data: alibi.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: alibi.device_id,
            },
            ndim: 1,
            dtype: dl_dtype_f32(),
            shape: alibi_shape.as_mut_ptr(),
            strides: alibi_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    });

    let lse_any = lse_tensor
        .as_ref()
        .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));
    let custom_mask_any = custom_mask_tensor
        .as_ref()
        .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));
    let alibi_any = alibi_tensor
        .as_ref()
        .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));

    let args: [TVMFFIAny; 15] = [
        any_dltensor_ptr(&q_tensor),
        any_dltensor_ptr(&k_tensor),
        any_dltensor_ptr(&v_tensor),
        any_dltensor_ptr(&tmp_tensor),
        any_dltensor_ptr(&out_tensor),
        lse_any,
        any_i64(params.mask_mode.as_i64()),
        any_i64(params.kv_layout.as_i64()),
        any_i64(params.window_left),
        custom_mask_any,
        alibi_any,
        any_f64(params.logits_soft_cap),
        any_f64(params.sm_scale),
        any_f64(1.0 / params.rope_scale),
        any_f64(1.0 / params.rope_theta),
    ];
    let mut result = any_none();

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.q.device_id, params.stream)? };
    let mut restore_guard = StreamRestoreGuard::new(runtime, params.q.device_id, previous_stream);
    // SAFETY: argument packing follows TVMFFIAny ABI.
    let call_result = unsafe {
        runtime.call_single_prefill(
            &kernel_uri,
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

fn dtype_filename(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
    }
}

fn bool_name(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn dl_dtype_from_norm_dtype(dtype: DType) -> DLDataType {
    match dtype {
        DType::F16 => DLDataType {
            code: KDL_FLOAT,
            bits: 16,
            lanes: 1,
        },
        DType::BF16 => DLDataType {
            code: crate::ffi::KDL_BFLOAT,
            bits: 16,
            lanes: 1,
        },
    }
}

fn dl_dtype_f32() -> DLDataType {
    DLDataType {
        code: KDL_FLOAT,
        bits: 32,
        lanes: 1,
    }
}

fn dl_dtype_u8() -> DLDataType {
    DLDataType {
        code: KDL_UINT,
        bits: 8,
        lanes: 1,
    }
}

#[cfg(feature = "cudarc")]
#[derive(Debug, Clone, Copy)]
pub struct MhaSinglePrefillCudarcOptions {
    pub mask_mode: MhaMaskMode,
    pub kv_layout: MhaQkvLayout,
    pub pos_encoding_mode: MhaPosEncodingMode,
    pub window_left: i64,
    pub logits_soft_cap: f64,
    pub sm_scale: Option<f64>,
    pub rope_scale: f64,
    pub rope_theta: f64,
    pub use_fp16_qk_reduction: bool,
}

#[cfg(feature = "cudarc")]
impl Default for MhaSinglePrefillCudarcOptions {
    fn default() -> Self {
        Self {
            mask_mode: MhaMaskMode::NonCausal,
            kv_layout: MhaQkvLayout::Nhd,
            pos_encoding_mode: MhaPosEncodingMode::None,
            window_left: -1,
            logits_soft_cap: 0.0,
            sm_scale: None,
            rope_scale: 1.0,
            rope_theta: 1e4,
            use_fp16_qk_reduction: false,
        }
    }
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn mha_single_prefill_cudarc<T, Q, K, V, W, O>(
    stream: &cudarc::driver::CudaStream,
    q: &Q,
    k: &K,
    v: &V,
    tmp: &mut W,
    out: &mut O,
    qo_len: usize,
    kv_len: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim_qk: usize,
    head_dim_vo: usize,
    dtype: DType,
    options: MhaSinglePrefillCudarcOptions,
) -> Result<(), FlashInferError>
where
    Q: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    K: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    V: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    if qo_len == 0
        || kv_len == 0
        || num_qo_heads == 0
        || num_kv_heads == 0
        || head_dim_qk == 0
        || head_dim_vo == 0
    {
        return Err(FlashInferError::invalid_argument(
            "qo_len/kv_len/num_heads/head_dim must be positive",
        ));
    }
    if num_qo_heads % num_kv_heads != 0 {
        return Err(FlashInferError::invalid_argument(
            "num_qo_heads must be divisible by num_kv_heads",
        ));
    }
    if tmp.len() == 0 {
        return Err(FlashInferError::invalid_argument(
            "tmp length must be positive",
        ));
    }

    let expected_q = qo_len
        .checked_mul(num_qo_heads)
        .and_then(|v| v.checked_mul(head_dim_qk))
        .ok_or_else(|| FlashInferError::invalid_argument("q size overflow"))?;
    let expected_k = kv_len
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim_qk))
        .ok_or_else(|| FlashInferError::invalid_argument("k size overflow"))?;
    let expected_v = kv_len
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim_vo))
        .ok_or_else(|| FlashInferError::invalid_argument("v size overflow"))?;
    let expected_out = qo_len
        .checked_mul(num_qo_heads)
        .and_then(|v| v.checked_mul(head_dim_vo))
        .ok_or_else(|| FlashInferError::invalid_argument("out size overflow"))?;

    if q.len() != expected_q {
        return Err(FlashInferError::invalid_argument(format!(
            "q length ({}) must equal qo_len * num_qo_heads * head_dim_qk ({expected_q})",
            q.len()
        )));
    }
    if k.len() != expected_k {
        return Err(FlashInferError::invalid_argument(format!(
            "k length ({}) must equal kv_len * num_kv_heads * head_dim_qk ({expected_k})",
            k.len()
        )));
    }
    if v.len() != expected_v {
        return Err(FlashInferError::invalid_argument(format!(
            "v length ({}) must equal kv_len * num_kv_heads * head_dim_vo ({expected_v})",
            v.len()
        )));
    }
    if out.len() != expected_out {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal qo_len * num_qo_heads * head_dim_vo ({expected_out})",
            out.len()
        )));
    }

    let tmp_len = tmp.len();
    let (q_ptr, _q_sync) = q.device_ptr(stream);
    let (k_ptr, _k_sync) = k.device_ptr(stream);
    let (v_ptr, _v_sync) = v.device_ptr(stream);
    let (tmp_ptr, _tmp_sync) = tmp.device_ptr_mut(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);

    let qo_len_i64 = i64::try_from(qo_len)
        .map_err(|_| FlashInferError::invalid_argument("qo_len does not fit in i64"))?;
    let kv_len_i64 = i64::try_from(kv_len)
        .map_err(|_| FlashInferError::invalid_argument("kv_len does not fit in i64"))?;
    let num_qo_heads_i64 = i64::try_from(num_qo_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_qo_heads does not fit in i64"))?;
    let num_kv_heads_i64 = i64::try_from(num_kv_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_kv_heads does not fit in i64"))?;
    let head_dim_qk_i64 = i64::try_from(head_dim_qk)
        .map_err(|_| FlashInferError::invalid_argument("head_dim_qk does not fit in i64"))?;
    let head_dim_vo_i64 = i64::try_from(head_dim_vo)
        .map_err(|_| FlashInferError::invalid_argument("head_dim_vo does not fit in i64"))?;
    let tmp_len_i64 = i64::try_from(tmp_len)
        .map_err(|_| FlashInferError::invalid_argument("tmp length does not fit in i64"))?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let q_stride1 = head_dim_qk_i64;
    let q_stride0 = num_qo_heads_i64
        .checked_mul(head_dim_qk_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("q stride overflow"))?;
    let out_stride1 = head_dim_vo_i64;
    let out_stride0 = num_qo_heads_i64
        .checked_mul(head_dim_vo_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("out stride overflow"))?;
    let (k_dim0, k_dim1, k_stride0, k_stride1) = match options.kv_layout {
        MhaQkvLayout::Nhd => {
            let s1 = head_dim_qk_i64;
            let s0 = num_kv_heads_i64
                .checked_mul(head_dim_qk_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("k stride overflow"))?;
            (kv_len_i64, num_kv_heads_i64, s0, s1)
        }
        MhaQkvLayout::Hnd => {
            let s1 = head_dim_qk_i64;
            let s0 = kv_len_i64
                .checked_mul(head_dim_qk_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("k stride overflow"))?;
            (num_kv_heads_i64, kv_len_i64, s0, s1)
        }
    };
    let (v_dim0, v_dim1, v_stride0, v_stride1) = match options.kv_layout {
        MhaQkvLayout::Nhd => {
            let s1 = head_dim_vo_i64;
            let s0 = num_kv_heads_i64
                .checked_mul(head_dim_vo_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("v stride overflow"))?;
            (kv_len_i64, num_kv_heads_i64, s0, s1)
        }
        MhaQkvLayout::Hnd => {
            let s1 = head_dim_vo_i64;
            let s0 = kv_len_i64
                .checked_mul(head_dim_vo_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("v stride overflow"))?;
            (num_kv_heads_i64, kv_len_i64, s0, s1)
        }
    };

    let sm_scale = options
        .sm_scale
        .unwrap_or_else(|| 1.0 / (head_dim_qk as f64).sqrt());

    let params = MhaSinglePrefillParams::new(
        MhaTensor3DDesc {
            ptr: q_ptr as usize as *const c_void,
            dim0: qo_len_i64,
            dim1: num_qo_heads_i64,
            dim2: head_dim_qk_i64,
            stride0: q_stride0,
            stride1: q_stride1,
            stride2: 1,
            dtype,
            device_id,
        },
        MhaTensor3DDesc {
            ptr: k_ptr as usize as *const c_void,
            dim0: k_dim0,
            dim1: k_dim1,
            dim2: head_dim_qk_i64,
            stride0: k_stride0,
            stride1: k_stride1,
            stride2: 1,
            dtype,
            device_id,
        },
        MhaTensor3DDesc {
            ptr: v_ptr as usize as *const c_void,
            dim0: v_dim0,
            dim1: v_dim1,
            dim2: head_dim_vo_i64,
            stride0: v_stride0,
            stride1: v_stride1,
            stride2: 1,
            dtype,
            device_id,
        },
        MhaTensor1DU8Desc {
            ptr: tmp_ptr as usize as *const c_void,
            len: tmp_len_i64,
            stride: 1,
            device_id,
        },
        MhaTensor3DDesc {
            ptr: out_ptr as usize as *const c_void,
            dim0: qo_len_i64,
            dim1: num_qo_heads_i64,
            dim2: head_dim_vo_i64,
            stride0: out_stride0,
            stride1: out_stride1,
            stride2: 1,
            dtype,
            device_id,
        },
        stream.cu_stream().cast(),
    )
    .with_mask_mode(options.mask_mode)
    .with_kv_layout(options.kv_layout)
    .with_pos_encoding_mode(options.pos_encoding_mode)
    .with_window_left(options.window_left)
    .with_logits_soft_cap(options.logits_soft_cap)
    .with_sm_scale(sm_scale)
    .with_rope_scale(options.rope_scale)
    .with_rope_theta(options.rope_theta)
    .with_fp16_qk_reduction(options.use_fp16_qk_reduction);

    mha_single_prefill(&params)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn non_null() -> *const c_void {
        std::ptr::NonNull::<u8>::dangling().as_ptr().cast()
    }

    fn valid_params() -> MhaSinglePrefillParams {
        MhaSinglePrefillParams::new(
            MhaTensor3DDesc {
                ptr: non_null(),
                dim0: 4,
                dim1: 8,
                dim2: 128,
                stride0: 1024,
                stride1: 128,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            MhaTensor3DDesc {
                ptr: non_null(),
                dim0: 6,
                dim1: 4,
                dim2: 128,
                stride0: 512,
                stride1: 128,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            MhaTensor3DDesc {
                ptr: non_null(),
                dim0: 6,
                dim1: 4,
                dim2: 128,
                stride0: 512,
                stride1: 128,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            MhaTensor1DU8Desc {
                ptr: non_null(),
                len: 1024,
                stride: 1,
                device_id: 0,
            },
            MhaTensor3DDesc {
                ptr: non_null(),
                dim0: 4,
                dim1: 8,
                dim2: 128,
                stride0: 1024,
                stride1: 128,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            std::ptr::null_mut(),
        )
    }

    #[test]
    fn validate_rejects_dtype_mismatch() {
        let mut params = valid_params();
        params.k.dtype = DType::BF16;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_output_shape_mismatch() {
        let mut params = valid_params();
        params.out.dim2 = 64;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_custom_mask_without_tensor() {
        let params = valid_params().with_mask_mode(MhaMaskMode::Custom);
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_alibi_without_slopes() {
        let params = valid_params().with_pos_encoding_mode(MhaPosEncodingMode::ALiBi);
        assert!(params.validate().is_err());
    }

    #[test]
    fn kernel_uri_matches_expected_pattern() {
        let uri = valid_params().kernel_uri();
        assert!(uri.contains("single_prefill_with_kv_cache_dtype_q_f16"));
        assert!(uri.contains("head_dim_qk_128"));
        assert!(uri.contains("use_swa_False"));
        assert!(uri.contains("use_logits_cap_False"));
    }
}
