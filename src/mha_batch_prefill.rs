use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    any_bool, any_dltensor_ptr, any_f64, any_i64, any_none, any_object_handle, DLDataType,
    DLDevice, DLTensor, TVMFFIAny, KDL_BFLOAT, KDL_CPU, KDL_CUDA, KDL_FLOAT, KDL_INT, KDL_UINT,
};
use crate::mha_prefill::{
    MhaMaskMode, MhaPosEncodingMode, MhaQkvLayout, MhaTensor1DF32Desc, MhaTensor1DU8Desc,
    MhaTensor2DF32Desc, MhaTensor3DDesc,
};
use crate::norm::DType;
use crate::runtime::FlashInferRuntime;

#[derive(Debug, Clone, Copy)]
pub struct MhaTensor1DI32Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct MhaTensor1DU16Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct MhaTensor1DU32Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct MhaHostTensor1DI32Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct MhaHostTensor1DU8Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct MhaBatchPrefillParams {
    /// Query tensor, rank-3: `[qo_indptr_host[-1], num_qo_heads, head_dim_qk]`.
    ///
    /// Cross-reference:
    /// `flashinfer/flashinfer/prefill.py::BatchPrefillWithRaggedKVCacheWrapper.run`.
    pub q: MhaTensor3DDesc,
    /// Key tensor, rank-3:
    /// - `NHD`: `[kv_indptr_host[-1], num_kv_heads, head_dim_qk]`
    /// - `HND`: `[num_kv_heads, kv_indptr_host[-1], head_dim_qk]`.
    pub k: MhaTensor3DDesc,
    /// Value tensor, rank-3:
    /// - `NHD`: `[kv_indptr_host[-1], num_kv_heads, head_dim_vo]`
    /// - `HND`: `[num_kv_heads, kv_indptr_host[-1], head_dim_vo]`.
    pub v: MhaTensor3DDesc,
    /// Query/output ragged offsets on device, rank-1 int32: `[batch_size + 1]`.
    pub qo_indptr: MhaTensor1DI32Desc,
    /// KV ragged offsets on device, rank-1 int32: `[batch_size + 1]`.
    pub kv_indptr: MhaTensor1DI32Desc,
    /// Query/output ragged offsets on host, rank-1 int32: `[batch_size + 1]`.
    ///
    /// This host tensor feeds the planning API (`plan`) in FlashInfer.
    pub qo_indptr_host: MhaHostTensor1DI32Desc,
    /// KV ragged offsets on host, rank-1 int32: `[batch_size + 1]`.
    ///
    /// This host tensor feeds the planning API (`plan`) in FlashInfer.
    pub kv_indptr_host: MhaHostTensor1DI32Desc,
    /// Float workspace on device, rank-1 u8: `[workspace_bytes]`.
    pub float_workspace: MhaTensor1DU8Desc,
    /// Int workspace on device, rank-1 u8: `[workspace_bytes]`.
    pub int_workspace: MhaTensor1DU8Desc,
    /// Page-locked int workspace on host, rank-1 u8: `[workspace_bytes]`.
    ///
    /// Cross-reference:
    /// `flashinfer/flashinfer/prefill.py::BatchPrefillWithRaggedKVCacheWrapper.plan`
    /// (`_pin_memory_int_workspace_buffer`).
    pub page_locked_int_workspace: MhaHostTensor1DU8Desc,
    /// Output tensor, rank-3: `[qo_indptr_host[-1], num_qo_heads, head_dim_vo]`.
    pub out: MhaTensor3DDesc,
    /// Optional log-sum-exp output, rank-2 f32: `[qo_indptr_host[-1], num_qo_heads]`.
    pub lse: Option<MhaTensor2DF32Desc>,
    /// Optional packed custom mask, rank-1 u8.
    pub custom_mask: Option<MhaTensor1DU8Desc>,
    /// Optional mask indptr, rank-1 int32: `[batch_size + 1]`.
    pub mask_indptr: Option<MhaTensor1DI32Desc>,
    /// Optional ALiBi slopes, rank-1 f32: `[num_qo_heads]`.
    pub alibi_slopes: Option<MhaTensor1DF32Desc>,
    /// Optional prefix lengths, rank-1 u32: `[batch_size]`.
    pub prefix_len_ptr: Option<MhaTensor1DU32Desc>,
    /// Optional token positions, rank-1 u16.
    pub token_pos_in_items_ptr: Option<MhaTensor1DU16Desc>,
    /// Optional max item lengths, rank-1 u16: `[batch_size]`.
    pub max_item_len_ptr: Option<MhaTensor1DU16Desc>,
    /// Causal flag used by planning (`plan`) stage.
    pub causal: bool,
    /// Mask mode code used by run stage.
    pub mask_mode: MhaMaskMode,
    /// KV layout enum (`NHD` or `HND`).
    pub kv_layout: MhaQkvLayout,
    /// Positional encoding mode enum.
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
    /// Zero-padded token-position length for multi-item metadata.
    pub token_pos_in_items_len: i64,
    /// Whether to enable fp16 QK reduction in compatible kernels.
    pub use_fp16_qk_reduction: bool,
    /// Whether to enable Programmatic Dependent Launch.
    pub enable_pdl: bool,
    /// FA2 fixed split size; `-1` means auto.
    pub fixed_split_size: i64,
    /// Disable split-kv in FA2 path.
    pub disable_split_kv: bool,
    /// Number of colocated CTAs for planning.
    pub num_colocated_ctas: i64,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl MhaBatchPrefillParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q: MhaTensor3DDesc,
        k: MhaTensor3DDesc,
        v: MhaTensor3DDesc,
        qo_indptr: MhaTensor1DI32Desc,
        kv_indptr: MhaTensor1DI32Desc,
        qo_indptr_host: MhaHostTensor1DI32Desc,
        kv_indptr_host: MhaHostTensor1DI32Desc,
        float_workspace: MhaTensor1DU8Desc,
        int_workspace: MhaTensor1DU8Desc,
        page_locked_int_workspace: MhaHostTensor1DU8Desc,
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
            qo_indptr,
            kv_indptr,
            qo_indptr_host,
            kv_indptr_host,
            float_workspace,
            int_workspace,
            page_locked_int_workspace,
            out,
            lse: None,
            custom_mask: None,
            mask_indptr: None,
            alibi_slopes: None,
            prefix_len_ptr: None,
            token_pos_in_items_ptr: None,
            max_item_len_ptr: None,
            causal: false,
            mask_mode: MhaMaskMode::NonCausal,
            kv_layout: MhaQkvLayout::Nhd,
            pos_encoding_mode: MhaPosEncodingMode::None,
            window_left: -1,
            logits_soft_cap: 0.0,
            sm_scale,
            rope_scale: 1.0,
            rope_theta: 1e4,
            token_pos_in_items_len: 0,
            use_fp16_qk_reduction: false,
            enable_pdl: false,
            fixed_split_size: -1,
            disable_split_kv: false,
            num_colocated_ctas: 0,
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

    pub fn with_mask_indptr(mut self, mask_indptr: MhaTensor1DI32Desc) -> Self {
        self.mask_indptr = Some(mask_indptr);
        self
    }

    pub fn with_alibi_slopes(mut self, alibi_slopes: MhaTensor1DF32Desc) -> Self {
        self.alibi_slopes = Some(alibi_slopes);
        self
    }

    pub fn with_prefix_len_ptr(mut self, prefix_len_ptr: MhaTensor1DU32Desc) -> Self {
        self.prefix_len_ptr = Some(prefix_len_ptr);
        self
    }

    pub fn with_token_pos_in_items_ptr(
        mut self,
        token_pos_in_items_ptr: MhaTensor1DU16Desc,
    ) -> Self {
        self.token_pos_in_items_ptr = Some(token_pos_in_items_ptr);
        self
    }

    pub fn with_max_item_len_ptr(mut self, max_item_len_ptr: MhaTensor1DU16Desc) -> Self {
        self.max_item_len_ptr = Some(max_item_len_ptr);
        self
    }

    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
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

    pub fn with_token_pos_in_items_len(mut self, token_pos_in_items_len: i64) -> Self {
        self.token_pos_in_items_len = token_pos_in_items_len;
        self
    }

    pub fn with_fp16_qk_reduction(mut self, use_fp16_qk_reduction: bool) -> Self {
        self.use_fp16_qk_reduction = use_fp16_qk_reduction;
        self
    }

    pub fn with_enable_pdl(mut self, enable_pdl: bool) -> Self {
        self.enable_pdl = enable_pdl;
        self
    }

    pub fn with_fixed_split_size(mut self, fixed_split_size: i64) -> Self {
        self.fixed_split_size = fixed_split_size;
        self
    }

    pub fn with_disable_split_kv(mut self, disable_split_kv: bool) -> Self {
        self.disable_split_kv = disable_split_kv;
        self
    }

    pub fn with_num_colocated_ctas(mut self, num_colocated_ctas: i64) -> Self {
        self.num_colocated_ctas = num_colocated_ctas;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.q.ptr, "q")?;
        check_non_null(self.k.ptr, "k")?;
        check_non_null(self.v.ptr, "v")?;
        check_non_null(self.qo_indptr.ptr, "qo_indptr")?;
        check_non_null(self.kv_indptr.ptr, "kv_indptr")?;
        check_non_null(self.qo_indptr_host.ptr, "qo_indptr_host")?;
        check_non_null(self.kv_indptr_host.ptr, "kv_indptr_host")?;
        check_non_null(self.float_workspace.ptr, "float_workspace")?;
        check_non_null(self.int_workspace.ptr, "int_workspace")?;
        check_non_null(
            self.page_locked_int_workspace.ptr,
            "page_locked_int_workspace",
        )?;
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

        if self.float_workspace.len <= 0 || self.int_workspace.len <= 0 {
            return Err(FlashInferError::invalid_argument(
                "workspace lengths must be positive",
            ));
        }
        if self.page_locked_int_workspace.len <= 0 {
            return Err(FlashInferError::invalid_argument(
                "page_locked_int_workspace length must be positive",
            ));
        }

        check_contiguous_1d("qo_indptr", self.qo_indptr.stride)?;
        check_contiguous_1d("kv_indptr", self.kv_indptr.stride)?;
        check_contiguous_1d("qo_indptr_host", self.qo_indptr_host.stride)?;
        check_contiguous_1d("kv_indptr_host", self.kv_indptr_host.stride)?;
        check_contiguous_1d("float_workspace", self.float_workspace.stride)?;
        check_contiguous_1d("int_workspace", self.int_workspace.stride)?;
        check_contiguous_1d(
            "page_locked_int_workspace",
            self.page_locked_int_workspace.stride,
        )?;

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
            || self.q.dtype != self.v.dtype
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

        let qo_host = read_host_i32(self.qo_indptr_host)?;
        let kv_host = read_host_i32(self.kv_indptr_host)?;

        let batch_size = validate_indptr("qo_indptr_host", &qo_host)?;
        let kv_batch_size = validate_indptr("kv_indptr_host", &kv_host)?;
        if kv_batch_size != batch_size {
            return Err(FlashInferError::invalid_argument(format!(
                "kv_indptr_host batch size ({kv_batch_size}) must match qo_indptr_host batch size ({batch_size})"
            )));
        }

        if self.qo_indptr.len != self.qo_indptr_host.len {
            return Err(FlashInferError::invalid_argument(
                "qo_indptr (device) length must match qo_indptr_host length",
            ));
        }
        if self.kv_indptr.len != self.kv_indptr_host.len {
            return Err(FlashInferError::invalid_argument(
                "kv_indptr (device) length must match kv_indptr_host length",
            ));
        }

        let qo_total = i64::from(*qo_host.last().unwrap_or(&0));
        let kv_total = i64::from(*kv_host.last().unwrap_or(&0));

        if self.q.dim0 != qo_total {
            return Err(FlashInferError::invalid_argument(format!(
                "q.dim0 ({}) must equal qo_indptr_host[-1] ({qo_total})",
                self.q.dim0
            )));
        }
        if self.out.dim0 != qo_total {
            return Err(FlashInferError::invalid_argument(format!(
                "out.dim0 ({}) must equal qo_indptr_host[-1] ({qo_total})",
                self.out.dim0
            )));
        }

        let num_qo_heads = self.q.dim1;
        let head_dim_qk = self.q.dim2;

        if self.k.dim2 != head_dim_qk {
            return Err(FlashInferError::invalid_argument(
                "k head_dim_qk must match q",
            ));
        }

        let head_dim_vo = self.v.dim2;
        if self.out.dim2 != head_dim_vo {
            return Err(FlashInferError::invalid_argument(
                "out head_dim_vo must match v",
            ));
        }
        if self.out.dim1 != num_qo_heads {
            return Err(FlashInferError::invalid_argument(
                "out num_qo_heads must match q",
            ));
        }

        let num_kv_heads = match self.kv_layout {
            MhaQkvLayout::Nhd => self.k.dim1,
            MhaQkvLayout::Hnd => self.k.dim0,
        };
        let k_kv_len = match self.kv_layout {
            MhaQkvLayout::Nhd => self.k.dim0,
            MhaQkvLayout::Hnd => self.k.dim1,
        };
        let v_kv_len = match self.kv_layout {
            MhaQkvLayout::Nhd => self.v.dim0,
            MhaQkvLayout::Hnd => self.v.dim1,
        };
        let v_num_kv_heads = match self.kv_layout {
            MhaQkvLayout::Nhd => self.v.dim1,
            MhaQkvLayout::Hnd => self.v.dim0,
        };

        if k_kv_len != kv_total || v_kv_len != kv_total {
            return Err(FlashInferError::invalid_argument(
                "k/v kv_len must equal kv_indptr_host[-1]",
            ));
        }
        if v_num_kv_heads != num_kv_heads {
            return Err(FlashInferError::invalid_argument(
                "k/v num_kv_heads must match",
            ));
        }
        if num_qo_heads % num_kv_heads != 0 {
            return Err(FlashInferError::invalid_argument(
                "num_qo_heads must be divisible by num_kv_heads",
            ));
        }

        let device_id = self.q.device_id;
        if self.k.device_id != device_id
            || self.v.device_id != device_id
            || self.qo_indptr.device_id != device_id
            || self.kv_indptr.device_id != device_id
            || self.float_workspace.device_id != device_id
            || self.int_workspace.device_id != device_id
            || self.out.device_id != device_id
        {
            return Err(FlashInferError::invalid_argument(
                "all required device tensors must be on the same CUDA device",
            ));
        }

        if self.window_left < -1 {
            return Err(FlashInferError::invalid_argument(
                "window_left must be -1 or >= 0",
            ));
        }
        if self.fixed_split_size < -1 {
            return Err(FlashInferError::invalid_argument(
                "fixed_split_size must be -1 or >= 0",
            ));
        }
        if self.num_colocated_ctas < 0 {
            return Err(FlashInferError::invalid_argument(
                "num_colocated_ctas must be >= 0",
            ));
        }
        if self.token_pos_in_items_len < 0 {
            return Err(FlashInferError::invalid_argument(
                "token_pos_in_items_len must be >= 0",
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
            if lse.rows != qo_total || lse.cols != num_qo_heads {
                return Err(FlashInferError::invalid_argument(
                    "lse shape must be [qo_indptr_host[-1], num_qo_heads]",
                ));
            }
            if lse.stride_col != 1 || lse.stride_row != lse.cols {
                return Err(FlashInferError::invalid_argument(
                    "lse must be contiguous row-major",
                ));
            }
            if lse.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "lse must be on the same CUDA device",
                ));
            }
        }

        if let Some(custom_mask) = self.custom_mask {
            check_non_null(custom_mask.ptr, "custom_mask")?;
            check_positive("custom_mask.len", custom_mask.len)?;
            check_contiguous_1d("custom_mask", custom_mask.stride)?;
            if custom_mask.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "custom_mask must be on the same CUDA device",
                ));
            }
        }

        if let Some(mask_indptr) = self.mask_indptr {
            check_non_null(mask_indptr.ptr, "mask_indptr")?;
            check_contiguous_1d("mask_indptr", mask_indptr.stride)?;
            if mask_indptr.len != self.qo_indptr_host.len {
                return Err(FlashInferError::invalid_argument(
                    "mask_indptr length must equal batch_size + 1",
                ));
            }
            if mask_indptr.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "mask_indptr must be on the same CUDA device",
                ));
            }
        }

        if self.mask_mode == MhaMaskMode::Custom
            && (self.custom_mask.is_none() || self.mask_indptr.is_none())
        {
            return Err(FlashInferError::invalid_argument(
                "custom_mask and mask_indptr are required when mask_mode is Custom",
            ));
        }

        if let Some(alibi) = self.alibi_slopes {
            check_non_null(alibi.ptr, "alibi_slopes")?;
            check_positive("alibi_slopes.len", alibi.len)?;
            check_contiguous_1d("alibi_slopes", alibi.stride)?;
            if alibi.len != num_qo_heads {
                return Err(FlashInferError::invalid_argument(
                    "alibi_slopes length must equal num_qo_heads",
                ));
            }
            if alibi.device_id != device_id {
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

        if let Some(prefix_len_ptr) = self.prefix_len_ptr {
            check_non_null(prefix_len_ptr.ptr, "prefix_len_ptr")?;
            check_contiguous_1d("prefix_len_ptr", prefix_len_ptr.stride)?;
            if prefix_len_ptr.len != i64::from(batch_size) {
                return Err(FlashInferError::invalid_argument(
                    "prefix_len_ptr length must equal batch_size",
                ));
            }
            if prefix_len_ptr.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "prefix_len_ptr must be on the same CUDA device",
                ));
            }
        }

        if let Some(token_pos_in_items_ptr) = self.token_pos_in_items_ptr {
            check_non_null(token_pos_in_items_ptr.ptr, "token_pos_in_items_ptr")?;
            check_contiguous_1d("token_pos_in_items_ptr", token_pos_in_items_ptr.stride)?;
            check_positive("token_pos_in_items_ptr.len", token_pos_in_items_ptr.len)?;
            if token_pos_in_items_ptr.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "token_pos_in_items_ptr must be on the same CUDA device",
                ));
            }
        }

        if let Some(max_item_len_ptr) = self.max_item_len_ptr {
            check_non_null(max_item_len_ptr.ptr, "max_item_len_ptr")?;
            check_contiguous_1d("max_item_len_ptr", max_item_len_ptr.stride)?;
            if max_item_len_ptr.len != i64::from(batch_size) {
                return Err(FlashInferError::invalid_argument(
                    "max_item_len_ptr length must equal batch_size",
                ));
            }
            if max_item_len_ptr.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "max_item_len_ptr must be on the same CUDA device",
                ));
            }
        }

        Ok(())
    }

    fn kernel_uri(&self) -> String {
        format!(
            "batch_prefill_with_kv_cache_dtype_q_{}_dtype_kv_{}_dtype_o_{}_dtype_idx_i32_head_dim_qk_{}_head_dim_vo_{}_posenc_{}_use_swa_{}_use_logits_cap_{}_f16qk_{}",
            dtype_filename(self.q.dtype),
            dtype_filename(self.k.dtype),
            dtype_filename(self.out.dtype),
            self.q.dim2,
            self.v.dim2,
            pos_encoding_mode_code(self.pos_encoding_mode),
            bool_name(self.window_left >= 0),
            bool_name(self.logits_soft_cap > 0.0),
            bool_name(self.use_fp16_qk_reduction),
        )
    }
}

pub fn mha_batch_prefill(params: &MhaBatchPrefillParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { mha_batch_prefill_with_runtime(runtime, params) }
}

unsafe fn mha_batch_prefill_with_runtime(
    runtime: &FlashInferRuntime,
    params: &MhaBatchPrefillParams,
) -> Result<(), FlashInferError> {
    let kernel_uri = params.kernel_uri();

    let qo_host = read_host_i32(params.qo_indptr_host)?;
    let kv_host = read_host_i32(params.kv_indptr_host)?;
    let kv_len: Vec<i32> = kv_host
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0]))
        .collect();

    let batch_size = i64::try_from(qo_host.len().saturating_sub(1))
        .map_err(|_| FlashInferError::invalid_argument("batch_size does not fit in i64"))?;
    let total_num_rows = params.q.dim0;
    let num_qo_heads = params.q.dim1;
    let num_kv_heads = match params.kv_layout {
        MhaQkvLayout::Nhd => params.k.dim1,
        MhaQkvLayout::Hnd => params.k.dim0,
    };

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

    let mut qo_indptr_shape = [params.qo_indptr.len];
    let mut qo_indptr_strides = [params.qo_indptr.stride];
    let qo_indptr_tensor = DLTensor {
        data: params.qo_indptr.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.qo_indptr.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: qo_indptr_shape.as_mut_ptr(),
        strides: qo_indptr_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut kv_indptr_shape = [params.kv_indptr.len];
    let mut kv_indptr_strides = [params.kv_indptr.stride];
    let kv_indptr_tensor = DLTensor {
        data: params.kv_indptr.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.kv_indptr.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: kv_indptr_shape.as_mut_ptr(),
        strides: kv_indptr_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut qo_host_shape = [params.qo_indptr_host.len];
    let mut qo_host_strides = [params.qo_indptr_host.stride];
    let qo_indptr_host_tensor = DLTensor {
        data: params.qo_indptr_host.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: qo_host_shape.as_mut_ptr(),
        strides: qo_host_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut kv_host_shape = [params.kv_indptr_host.len];
    let mut kv_host_strides = [params.kv_indptr_host.stride];
    let kv_indptr_host_tensor = DLTensor {
        data: params.kv_indptr_host.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: kv_host_shape.as_mut_ptr(),
        strides: kv_host_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut kv_len_shape = [i64::try_from(kv_len.len())
        .map_err(|_| FlashInferError::invalid_argument("kv_len length does not fit in i64"))?];
    let mut kv_len_strides = [1_i64];
    let kv_len_tensor = DLTensor {
        data: kv_len.as_ptr().cast_mut().cast(),
        device: DLDevice {
            device_type: KDL_CPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: kv_len_shape.as_mut_ptr(),
        strides: kv_len_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut float_workspace_shape = [params.float_workspace.len];
    let mut float_workspace_strides = [params.float_workspace.stride];
    let float_workspace_tensor = DLTensor {
        data: params.float_workspace.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.float_workspace.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_u8(),
        shape: float_workspace_shape.as_mut_ptr(),
        strides: float_workspace_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut int_workspace_shape = [params.int_workspace.len];
    let mut int_workspace_strides = [params.int_workspace.stride];
    let int_workspace_tensor = DLTensor {
        data: params.int_workspace.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.int_workspace.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_u8(),
        shape: int_workspace_shape.as_mut_ptr(),
        strides: int_workspace_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut page_locked_workspace_shape = [params.page_locked_int_workspace.len];
    let mut page_locked_workspace_strides = [params.page_locked_int_workspace.stride];
    let page_locked_int_workspace_tensor = DLTensor {
        data: params.page_locked_int_workspace.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: dl_dtype_u8(),
        shape: page_locked_workspace_shape.as_mut_ptr(),
        strides: page_locked_workspace_strides.as_mut_ptr(),
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

    let mut mask_indptr_shape = [0_i64; 1];
    let mut mask_indptr_strides = [0_i64; 1];
    let mask_indptr_tensor = params.mask_indptr.map(|mask_indptr| {
        mask_indptr_shape = [mask_indptr.len];
        mask_indptr_strides = [mask_indptr.stride];
        DLTensor {
            data: mask_indptr.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: mask_indptr.device_id,
            },
            ndim: 1,
            dtype: dl_dtype_i32(),
            shape: mask_indptr_shape.as_mut_ptr(),
            strides: mask_indptr_strides.as_mut_ptr(),
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

    let mut prefix_len_shape = [0_i64; 1];
    let mut prefix_len_strides = [0_i64; 1];
    let prefix_len_tensor = params.prefix_len_ptr.map(|prefix_len| {
        prefix_len_shape = [prefix_len.len];
        prefix_len_strides = [prefix_len.stride];
        DLTensor {
            data: prefix_len.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: prefix_len.device_id,
            },
            ndim: 1,
            dtype: dl_dtype_u32(),
            shape: prefix_len_shape.as_mut_ptr(),
            strides: prefix_len_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    });

    let mut token_pos_shape = [0_i64; 1];
    let mut token_pos_strides = [0_i64; 1];
    let token_pos_tensor = params.token_pos_in_items_ptr.map(|token_pos| {
        token_pos_shape = [token_pos.len];
        token_pos_strides = [token_pos.stride];
        DLTensor {
            data: token_pos.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: token_pos.device_id,
            },
            ndim: 1,
            dtype: dl_dtype_u16(),
            shape: token_pos_shape.as_mut_ptr(),
            strides: token_pos_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    });

    let mut max_item_len_shape = [0_i64; 1];
    let mut max_item_len_strides = [0_i64; 1];
    let max_item_len_tensor = params.max_item_len_ptr.map(|max_item_len| {
        max_item_len_shape = [max_item_len.len];
        max_item_len_strides = [max_item_len.stride];
        DLTensor {
            data: max_item_len.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: max_item_len.device_id,
            },
            ndim: 1,
            dtype: dl_dtype_u16(),
            shape: max_item_len_shape.as_mut_ptr(),
            strides: max_item_len_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    });

    let mut plan_result_view = any_none();
    let plan_args: [TVMFFIAny; 19] = [
        any_dltensor_ptr(&float_workspace_tensor),
        any_dltensor_ptr(&int_workspace_tensor),
        any_dltensor_ptr(&page_locked_int_workspace_tensor),
        any_dltensor_ptr(&qo_indptr_host_tensor),
        any_dltensor_ptr(&kv_indptr_host_tensor),
        any_dltensor_ptr(&kv_len_tensor),
        any_i64(total_num_rows),
        any_i64(batch_size),
        any_i64(num_qo_heads),
        any_i64(num_kv_heads),
        any_i64(1), // page_size for ragged plan path
        any_bool(false),
        any_i64(params.q.dim2),
        any_i64(params.v.dim2),
        any_bool(params.causal),
        any_i64(params.window_left),
        any_i64(params.fixed_split_size),
        any_bool(params.disable_split_kv),
        any_i64(params.num_colocated_ctas),
    ];

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.q.device_id, params.stream)? };
    let mut restore_guard = StreamRestoreGuard::new(runtime, params.q.device_id, previous_stream);

    let call_result = (|| -> Result<(), FlashInferError> {
        // SAFETY: argument packing follows TVMFFIAny ABI.
        unsafe {
            runtime.call_batch_prefill_plan(
                &kernel_uri,
                plan_args.as_ptr(),
                plan_args.len() as i32,
                &mut plan_result_view as *mut _,
            )?;
        }

        // SAFETY: converts returned AnyView into owned Any so lifetime can cross calls safely.
        let plan_result_owned = unsafe { runtime.any_view_to_owned(&plan_result_view)? };
        let mut plan_result_guard = AnyObjectDecRefGuard::new(runtime, &plan_result_owned);

        let lse_any = lse_tensor
            .as_ref()
            .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));
        let custom_mask_any = custom_mask_tensor
            .as_ref()
            .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));
        let mask_indptr_any = mask_indptr_tensor
            .as_ref()
            .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));
        let alibi_any = alibi_tensor
            .as_ref()
            .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));
        let prefix_len_any = prefix_len_tensor
            .as_ref()
            .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));
        let token_pos_any = token_pos_tensor
            .as_ref()
            .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));
        let max_item_len_any = max_item_len_tensor
            .as_ref()
            .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));

        let mut run_result = any_none();
        let run_args: [TVMFFIAny; 25] = [
            any_dltensor_ptr(&float_workspace_tensor),
            any_dltensor_ptr(&int_workspace_tensor),
            plan_result_owned,
            any_dltensor_ptr(&q_tensor),
            any_dltensor_ptr(&k_tensor),
            any_dltensor_ptr(&v_tensor),
            any_dltensor_ptr(&qo_indptr_tensor),
            any_dltensor_ptr(&kv_indptr_tensor),
            any_dltensor_ptr(&out_tensor),
            lse_any,
            any_i64(mask_mode_code(params.mask_mode)),
            any_i64(kv_layout_code(params.kv_layout)),
            any_i64(params.window_left),
            any_bool(params.enable_pdl),
            custom_mask_any,
            mask_indptr_any,
            alibi_any,
            prefix_len_any,
            token_pos_any,
            max_item_len_any,
            any_f64(params.logits_soft_cap),
            any_f64(params.sm_scale),
            any_f64(1.0 / params.rope_scale),
            any_f64(1.0 / params.rope_theta),
            any_i64(params.token_pos_in_items_len),
        ];

        // SAFETY: argument packing follows TVMFFIAny ABI.
        let launch_result = unsafe {
            runtime.call_batch_prefill_ragged(
                &kernel_uri,
                run_args.as_ptr(),
                run_args.len() as i32,
                &mut run_result as *mut _,
            )
        };

        // Ensure plan_info object is released before leaving this scope.
        let _ = plan_result_guard.release_now();

        launch_result
    })();

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

    fn release_now(&mut self) -> Result<(), FlashInferError> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        // SAFETY: object handle was returned by TVM-FFI as part of plan result.
        unsafe {
            self.runtime.object_dec_ref(self.obj);
        }
        Ok(())
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

fn check_contiguous_1d(name: &str, stride: i64) -> Result<(), FlashInferError> {
    if stride != 1 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} stride must be 1"
        )));
    }
    Ok(())
}

fn read_host_i32(desc: MhaHostTensor1DI32Desc) -> Result<Vec<i32>, FlashInferError> {
    if desc.len <= 0 {
        return Err(FlashInferError::invalid_argument(
            "host indptr length must be positive",
        ));
    }
    if desc.stride <= 0 {
        return Err(FlashInferError::invalid_argument(
            "host indptr stride must be positive",
        ));
    }
    let len = usize::try_from(desc.len).map_err(|_| {
        FlashInferError::invalid_argument("host indptr length does not fit in usize")
    })?;
    let stride = usize::try_from(desc.stride).map_err(|_| {
        FlashInferError::invalid_argument("host indptr stride does not fit in usize")
    })?;

    // SAFETY: caller provides a valid host pointer and length/stride metadata.
    let base = desc.ptr.cast::<i32>();
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        // SAFETY: index stays within caller-declared logical extent.
        let value = unsafe { *base.add(i.saturating_mul(stride)) };
        out.push(value);
    }
    Ok(out)
}

fn validate_indptr(name: &str, indptr: &[i32]) -> Result<u32, FlashInferError> {
    if indptr.len() < 2 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} length must be at least 2"
        )));
    }
    if indptr[0] != 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name}[0] must be 0"
        )));
    }
    for i in 1..indptr.len() {
        if indptr[i] < indptr[i - 1] {
            return Err(FlashInferError::invalid_argument(format!(
                "{name} must be non-decreasing"
            )));
        }
    }
    if indptr[indptr.len() - 1] <= 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} last value must be positive"
        )));
    }

    u32::try_from(indptr.len() - 1)
        .map_err(|_| FlashInferError::invalid_argument(format!("{name} batch size overflow")))
}

fn dtype_filename(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
    }
}

fn bool_name(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

fn pos_encoding_mode_code(mode: MhaPosEncodingMode) -> i64 {
    match mode {
        MhaPosEncodingMode::None => 0,
        MhaPosEncodingMode::RoPELlama => 1,
        MhaPosEncodingMode::ALiBi => 2,
    }
}

fn kv_layout_code(layout: MhaQkvLayout) -> i64 {
    match layout {
        MhaQkvLayout::Nhd => 0,
        MhaQkvLayout::Hnd => 1,
    }
}

fn mask_mode_code(mask_mode: MhaMaskMode) -> i64 {
    match mask_mode {
        MhaMaskMode::NonCausal => 0,
        MhaMaskMode::Causal => 1,
        MhaMaskMode::Custom => 2,
        MhaMaskMode::MultiItemScoring => 3,
    }
}

fn dl_dtype_from_norm_dtype(dtype: DType) -> DLDataType {
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

fn dl_dtype_i32() -> DLDataType {
    DLDataType {
        code: KDL_INT,
        bits: 32,
        lanes: 1,
    }
}

fn dl_dtype_u16() -> DLDataType {
    DLDataType {
        code: KDL_UINT,
        bits: 16,
        lanes: 1,
    }
}

fn dl_dtype_u32() -> DLDataType {
    DLDataType {
        code: KDL_UINT,
        bits: 32,
        lanes: 1,
    }
}

#[cfg(feature = "cudarc")]
#[derive(Debug, Clone, Copy)]
pub struct MhaBatchPrefillCudarcOptions {
    pub causal: bool,
    pub pos_encoding_mode: MhaPosEncodingMode,
    pub window_left: i64,
    pub logits_soft_cap: f64,
    pub sm_scale: Option<f64>,
    pub rope_scale: f64,
    pub rope_theta: f64,
    pub use_fp16_qk_reduction: bool,
    pub enable_pdl: bool,
    pub fixed_split_size: i64,
    pub disable_split_kv: bool,
    pub num_colocated_ctas: i64,
}

#[cfg(feature = "cudarc")]
impl Default for MhaBatchPrefillCudarcOptions {
    fn default() -> Self {
        Self {
            causal: false,
            pos_encoding_mode: MhaPosEncodingMode::None,
            window_left: -1,
            logits_soft_cap: 0.0,
            sm_scale: None,
            rope_scale: 1.0,
            rope_theta: 1e4,
            use_fp16_qk_reduction: false,
            enable_pdl: false,
            fixed_split_size: -1,
            disable_split_kv: false,
            num_colocated_ctas: 0,
        }
    }
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn mha_batch_prefill_cudarc<T, Q, K, V, O, QI, KI, FW, IW>(
    stream: &cudarc::driver::CudaStream,
    q: &Q,
    k: &K,
    v: &V,
    qo_indptr: &QI,
    kv_indptr: &KI,
    qo_indptr_host: &[i32],
    kv_indptr_host: &[i32],
    float_workspace: &mut FW,
    int_workspace: &mut IW,
    page_locked_int_workspace: &mut [u8],
    out: &mut O,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim_qk: usize,
    head_dim_vo: usize,
    kv_layout: MhaQkvLayout,
    dtype: DType,
    options: MhaBatchPrefillCudarcOptions,
) -> Result<(), FlashInferError>
where
    Q: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    K: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    V: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    QI: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    KI: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    FW: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
    IW: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    if num_qo_heads == 0 || num_kv_heads == 0 || head_dim_qk == 0 || head_dim_vo == 0 {
        return Err(FlashInferError::invalid_argument(
            "num_heads/head_dim must be positive",
        ));
    }

    if qo_indptr_host.len() < 2 || kv_indptr_host.len() < 2 {
        return Err(FlashInferError::invalid_argument(
            "qo_indptr_host/kv_indptr_host length must be at least 2",
        ));
    }
    if qo_indptr_host.len() != kv_indptr_host.len() {
        return Err(FlashInferError::invalid_argument(
            "qo_indptr_host and kv_indptr_host length mismatch",
        ));
    }

    let qo_total = usize::try_from(*qo_indptr_host.last().unwrap_or(&0))
        .map_err(|_| FlashInferError::invalid_argument("qo total does not fit in usize"))?;
    let kv_total = usize::try_from(*kv_indptr_host.last().unwrap_or(&0))
        .map_err(|_| FlashInferError::invalid_argument("kv total does not fit in usize"))?;

    let expected_q = qo_total
        .checked_mul(num_qo_heads)
        .and_then(|v| v.checked_mul(head_dim_qk))
        .ok_or_else(|| FlashInferError::invalid_argument("q size overflow"))?;
    let expected_k = kv_total
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim_qk))
        .ok_or_else(|| FlashInferError::invalid_argument("k size overflow"))?;
    let expected_v = kv_total
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim_vo))
        .ok_or_else(|| FlashInferError::invalid_argument("v size overflow"))?;
    let expected_out = qo_total
        .checked_mul(num_qo_heads)
        .and_then(|v| v.checked_mul(head_dim_vo))
        .ok_or_else(|| FlashInferError::invalid_argument("out size overflow"))?;

    if q.len() != expected_q {
        return Err(FlashInferError::invalid_argument(format!(
            "q length ({}) must equal qo_total * num_qo_heads * head_dim_qk ({expected_q})",
            q.len()
        )));
    }
    if k.len() != expected_k {
        return Err(FlashInferError::invalid_argument(format!(
            "k length ({}) must equal kv_total * num_kv_heads * head_dim_qk ({expected_k})",
            k.len()
        )));
    }
    if v.len() != expected_v {
        return Err(FlashInferError::invalid_argument(format!(
            "v length ({}) must equal kv_total * num_kv_heads * head_dim_vo ({expected_v})",
            v.len()
        )));
    }
    if out.len() != expected_out {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal qo_total * num_qo_heads * head_dim_vo ({expected_out})",
            out.len()
        )));
    }

    if qo_indptr.len() != qo_indptr_host.len() {
        return Err(FlashInferError::invalid_argument(
            "qo_indptr device length must match qo_indptr_host length",
        ));
    }
    if kv_indptr.len() != kv_indptr_host.len() {
        return Err(FlashInferError::invalid_argument(
            "kv_indptr device length must match kv_indptr_host length",
        ));
    }
    let float_workspace_len = float_workspace.len();
    let int_workspace_len = int_workspace.len();
    if float_workspace_len == 0 || int_workspace_len == 0 {
        return Err(FlashInferError::invalid_argument(
            "float/int workspace lengths must be positive",
        ));
    }
    if page_locked_int_workspace.is_empty() {
        return Err(FlashInferError::invalid_argument(
            "page_locked_int_workspace length must be positive",
        ));
    }

    let (q_ptr, _q_sync) = q.device_ptr(stream);
    let (k_ptr, _k_sync) = k.device_ptr(stream);
    let (v_ptr, _v_sync) = v.device_ptr(stream);
    let (qo_indptr_ptr, _qo_sync) = qo_indptr.device_ptr(stream);
    let (kv_indptr_ptr, _kv_sync) = kv_indptr.device_ptr(stream);
    let (float_workspace_ptr, _float_ws_sync) = float_workspace.device_ptr_mut(stream);
    let (int_workspace_ptr, _int_ws_sync) = int_workspace.device_ptr_mut(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);

    let qo_total_i64 = i64::try_from(qo_total)
        .map_err(|_| FlashInferError::invalid_argument("qo total does not fit in i64"))?;
    let kv_total_i64 = i64::try_from(kv_total)
        .map_err(|_| FlashInferError::invalid_argument("kv total does not fit in i64"))?;
    let qo_indptr_len_i64 = i64::try_from(qo_indptr_host.len()).map_err(|_| {
        FlashInferError::invalid_argument("qo_indptr_host length does not fit in i64")
    })?;
    let kv_indptr_len_i64 = i64::try_from(kv_indptr_host.len()).map_err(|_| {
        FlashInferError::invalid_argument("kv_indptr_host length does not fit in i64")
    })?;
    let num_qo_heads_i64 = i64::try_from(num_qo_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_qo_heads does not fit in i64"))?;
    let num_kv_heads_i64 = i64::try_from(num_kv_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_kv_heads does not fit in i64"))?;
    let head_dim_qk_i64 = i64::try_from(head_dim_qk)
        .map_err(|_| FlashInferError::invalid_argument("head_dim_qk does not fit in i64"))?;
    let head_dim_vo_i64 = i64::try_from(head_dim_vo)
        .map_err(|_| FlashInferError::invalid_argument("head_dim_vo does not fit in i64"))?;
    let float_ws_len_i64 = i64::try_from(float_workspace_len).map_err(|_| {
        FlashInferError::invalid_argument("float_workspace length does not fit in i64")
    })?;
    let int_ws_len_i64 = i64::try_from(int_workspace_len).map_err(|_| {
        FlashInferError::invalid_argument("int_workspace length does not fit in i64")
    })?;
    let page_locked_ws_len_i64 = i64::try_from(page_locked_int_workspace.len()).map_err(|_| {
        FlashInferError::invalid_argument("page_locked_int_workspace length does not fit in i64")
    })?;

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

    let (k_dim0, k_dim1, k_stride0, k_stride1) = match kv_layout {
        MhaQkvLayout::Nhd => {
            let s1 = head_dim_qk_i64;
            let s0 = num_kv_heads_i64
                .checked_mul(head_dim_qk_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("k stride overflow"))?;
            (kv_total_i64, num_kv_heads_i64, s0, s1)
        }
        MhaQkvLayout::Hnd => {
            let s1 = head_dim_qk_i64;
            let s0 = kv_total_i64
                .checked_mul(head_dim_qk_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("k stride overflow"))?;
            (num_kv_heads_i64, kv_total_i64, s0, s1)
        }
    };
    let (v_dim0, v_dim1, v_stride0, v_stride1) = match kv_layout {
        MhaQkvLayout::Nhd => {
            let s1 = head_dim_vo_i64;
            let s0 = num_kv_heads_i64
                .checked_mul(head_dim_vo_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("v stride overflow"))?;
            (kv_total_i64, num_kv_heads_i64, s0, s1)
        }
        MhaQkvLayout::Hnd => {
            let s1 = head_dim_vo_i64;
            let s0 = kv_total_i64
                .checked_mul(head_dim_vo_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("v stride overflow"))?;
            (num_kv_heads_i64, kv_total_i64, s0, s1)
        }
    };

    let sm_scale = options
        .sm_scale
        .unwrap_or_else(|| 1.0 / (head_dim_qk as f64).sqrt());

    let mask_mode = if options.causal {
        MhaMaskMode::Causal
    } else {
        MhaMaskMode::NonCausal
    };

    let params = MhaBatchPrefillParams::new(
        MhaTensor3DDesc {
            ptr: q_ptr as usize as *const c_void,
            dim0: qo_total_i64,
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
        MhaTensor1DI32Desc {
            ptr: qo_indptr_ptr as usize as *const c_void,
            len: qo_indptr_len_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: kv_indptr_ptr as usize as *const c_void,
            len: kv_indptr_len_i64,
            stride: 1,
            device_id,
        },
        MhaHostTensor1DI32Desc {
            ptr: qo_indptr_host.as_ptr().cast(),
            len: qo_indptr_len_i64,
            stride: 1,
        },
        MhaHostTensor1DI32Desc {
            ptr: kv_indptr_host.as_ptr().cast(),
            len: kv_indptr_len_i64,
            stride: 1,
        },
        MhaTensor1DU8Desc {
            ptr: float_workspace_ptr as usize as *const c_void,
            len: float_ws_len_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DU8Desc {
            ptr: int_workspace_ptr as usize as *const c_void,
            len: int_ws_len_i64,
            stride: 1,
            device_id,
        },
        MhaHostTensor1DU8Desc {
            ptr: page_locked_int_workspace.as_ptr().cast(),
            len: page_locked_ws_len_i64,
            stride: 1,
        },
        MhaTensor3DDesc {
            ptr: out_ptr as usize as *const c_void,
            dim0: qo_total_i64,
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
    .with_causal(options.causal)
    .with_mask_mode(mask_mode)
    .with_kv_layout(kv_layout)
    .with_pos_encoding_mode(options.pos_encoding_mode)
    .with_window_left(options.window_left)
    .with_logits_soft_cap(options.logits_soft_cap)
    .with_sm_scale(sm_scale)
    .with_rope_scale(options.rope_scale)
    .with_rope_theta(options.rope_theta)
    .with_fp16_qk_reduction(options.use_fp16_qk_reduction)
    .with_enable_pdl(options.enable_pdl)
    .with_fixed_split_size(options.fixed_split_size)
    .with_disable_split_kv(options.disable_split_kv)
    .with_num_colocated_ctas(options.num_colocated_ctas);

    mha_batch_prefill(&params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_null() -> *const c_void {
        std::ptr::NonNull::<u8>::dangling().as_ptr().cast()
    }

    fn valid_params() -> MhaBatchPrefillParams {
        static QO_HOST: [i32; 3] = [0, 2, 4];
        static KV_HOST: [i32; 3] = [0, 3, 5];

        MhaBatchPrefillParams::new(
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
                dim0: 5,
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
                dim0: 5,
                dim1: 4,
                dim2: 128,
                stride0: 512,
                stride1: 128,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            MhaTensor1DI32Desc {
                ptr: non_null(),
                len: 3,
                stride: 1,
                device_id: 0,
            },
            MhaTensor1DI32Desc {
                ptr: non_null(),
                len: 3,
                stride: 1,
                device_id: 0,
            },
            MhaHostTensor1DI32Desc {
                ptr: QO_HOST.as_ptr().cast(),
                len: 3,
                stride: 1,
            },
            MhaHostTensor1DI32Desc {
                ptr: KV_HOST.as_ptr().cast(),
                len: 3,
                stride: 1,
            },
            MhaTensor1DU8Desc {
                ptr: non_null(),
                len: 4096,
                stride: 1,
                device_id: 0,
            },
            MhaTensor1DU8Desc {
                ptr: non_null(),
                len: 4096,
                stride: 1,
                device_id: 0,
            },
            MhaHostTensor1DU8Desc {
                ptr: non_null(),
                len: 4096,
                stride: 1,
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
    fn validate_rejects_indptr_stride_mismatch() {
        let mut params = valid_params();
        params.qo_indptr.stride = 2;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_shape_mismatch_against_indptr() {
        let mut params = valid_params();
        params.q.dim0 = 3;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_custom_mask_without_mask_indptr() {
        let params = valid_params()
            .with_mask_mode(MhaMaskMode::Custom)
            .with_custom_mask(MhaTensor1DU8Desc {
                ptr: non_null(),
                len: 16,
                stride: 1,
                device_id: 0,
            });
        assert!(params.validate().is_err());
    }

    #[test]
    fn kernel_uri_matches_expected_pattern() {
        let uri = valid_params().kernel_uri();
        assert!(uri.contains("batch_prefill_with_kv_cache_dtype_q_f16"));
        assert!(uri.contains("dtype_idx_i32"));
        assert!(uri.contains("head_dim_qk_128"));
        assert!(uri.contains("use_swa_False"));
        assert!(uri.contains("use_logits_cap_False"));
    }
}
