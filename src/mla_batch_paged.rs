//! Rust FFI for FlashInfer's batched MLA (Multi-head Latent Attention)
//! paged-KV attention kernels (FA2 + Hopper SM90 backends).
//!
//! MLA differs from regular paged attention:
//! - The KV cache is split into two tensors:
//!     * `ckv_cache`: `[num_pages, page_size, head_dim_ckv]` (the latent
//!       KV, e.g. `kv_lora_rank=512` for DeepSeek-V3 / Kimi K2.5)
//!     * `kpe_cache`: `[num_pages, page_size, head_dim_kpe]` (the
//!       decoupled-rope key, e.g. `qk_rope_head_dim=64`)
//!   Both share the same `(num_pages, page_size)` and the same paged
//!   index tables.
//! - The query is also split:
//!     * `q_nope`: `[n, num_heads, head_dim_ckv]`  (matrix-absorbed Q)
//!     * `q_pe`:   `[n, num_heads, head_dim_kpe]`  (decoupled-rope Q)
//! - The output `o` is in the latent (ckv) space:
//!     * `o`: `[n, num_heads, head_dim_ckv]`
//!
//! See `flashinfer/csrc/batch_mla_binding.cu` and
//! `flashinfer/include/flashinfer/attention/mla.cuh` for the C++ side.

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CPU, KDL_CUDA, KDL_FLOAT, KDL_INT, KDL_UINT,
    TVMFFIAny, any_bool, any_dltensor_ptr, any_f64, any_i64, any_none, any_object_handle,
};
use crate::mha_batch_prefill::{MhaHostTensor1DI32Desc, MhaHostTensor1DU8Desc, MhaTensor1DI32Desc};
use crate::mha_prefill::{MhaMaskMode, MhaTensor1DU8Desc, MhaTensor2DF32Desc};
use crate::norm::DType;
use crate::runtime::FlashInferRuntime;

/// FlashInfer MLA kernel backend selector. Maps to the URI suffix:
/// `Fa2` -> default, `Hopper` -> appends `_sm90`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlaBackend {
    /// FlashAttention-2 generic backend (works on Ampere+).
    Fa2,
    /// Hopper SM90 specialized backend (H100).
    Hopper,
}

/// Description of a rank-3 device tensor used by MLA. Layout is dense
/// row-major with the last dimension contiguous (`stride2 == 1`).
#[derive(Debug, Clone, Copy)]
pub struct MlaTensor3DDesc {
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

/// Run-time parameters for a single `BatchMLAPagedAttentionRun` invocation.
#[derive(Debug, Clone, Copy)]
pub struct MlaBatchPagedAttentionParams {
    /// Q-nope (latent component), rank-3: `[n, num_heads, head_dim_ckv]`.
    pub q_nope: MlaTensor3DDesc,
    /// Q-pe (decoupled-rope component), rank-3: `[n, num_heads, head_dim_kpe]`.
    pub q_pe: MlaTensor3DDesc,
    /// CKV (compressed/latent KV) page table, rank-3:
    /// `[num_pages, page_size, head_dim_ckv]`.
    pub ckv_cache: MlaTensor3DDesc,
    /// KPE (decoupled-rope K) page table, rank-3:
    /// `[num_pages, page_size, head_dim_kpe]`.
    pub kpe_cache: MlaTensor3DDesc,
    /// Page indices on device, rank-1 int32.
    pub kv_indices: MhaTensor1DI32Desc,
    /// Output tensor, rank-3: `[n, num_heads, head_dim_ckv]`.
    /// (MLA output lives in the latent space; the model layer is expected
    /// to project it back to per-head v-space via `kv_b_proj` if needed.)
    pub out: MlaTensor3DDesc,
    /// Optional log-sum-exp output, rank-2 f32: `[n, num_heads]`.
    pub lse: Option<MhaTensor2DF32Desc>,
    /// Float workspace on device, rank-1 u8: `[workspace_bytes]`.
    /// Must match the workspace passed to `mla_batch_paged_plan`.
    pub float_workspace: MhaTensor1DU8Desc,
    /// Int workspace on device, rank-1 u8: `[workspace_bytes]`.
    /// Must match the workspace passed to `mla_batch_paged_plan`.
    pub int_workspace: MhaTensor1DU8Desc,
    /// Mask mode: `NonCausal`, `Causal`, or `Custom` (custom requires
    /// extra arguments not yet exposed here).
    pub mask_mode: MhaMaskMode,
    /// Total number of attention heads (rank-local for TP).
    pub num_heads: i64,
    /// Page size (must match plan).
    pub page_size: i64,
    /// Softmax scale; default `1 / sqrt(head_dim_ckv + head_dim_kpe)`.
    pub sm_scale: f64,
    /// If true, returned LSE is in base-`e` (instead of base-2).
    pub return_lse_base_on_e: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl MlaBatchPagedAttentionParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_nope: MlaTensor3DDesc,
        q_pe: MlaTensor3DDesc,
        ckv_cache: MlaTensor3DDesc,
        kpe_cache: MlaTensor3DDesc,
        kv_indices: MhaTensor1DI32Desc,
        out: MlaTensor3DDesc,
        float_workspace: MhaTensor1DU8Desc,
        int_workspace: MhaTensor1DU8Desc,
        num_heads: i64,
        page_size: i64,
        stream: *mut c_void,
    ) -> Self {
        let head_dim_total = q_nope.dim2.saturating_add(q_pe.dim2);
        let sm_scale = if head_dim_total > 0 {
            1.0 / (head_dim_total as f64).sqrt()
        } else {
            1.0
        };
        Self {
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            kv_indices,
            out,
            lse: None,
            float_workspace,
            int_workspace,
            mask_mode: MhaMaskMode::NonCausal,
            num_heads,
            page_size,
            sm_scale,
            return_lse_base_on_e: false,
            stream,
        }
    }

    pub fn with_lse(mut self, lse: MhaTensor2DF32Desc) -> Self {
        self.lse = Some(lse);
        self
    }

    pub fn with_mask_mode(mut self, mask_mode: MhaMaskMode) -> Self {
        self.mask_mode = mask_mode;
        self
    }

    pub fn with_sm_scale(mut self, sm_scale: f64) -> Self {
        self.sm_scale = sm_scale;
        self
    }

    pub fn with_lse_base_on_e(mut self, base_on_e: bool) -> Self {
        self.return_lse_base_on_e = base_on_e;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.q_nope.ptr, "q_nope")?;
        check_non_null(self.q_pe.ptr, "q_pe")?;
        check_non_null(self.ckv_cache.ptr, "ckv_cache")?;
        check_non_null(self.kpe_cache.ptr, "kpe_cache")?;
        check_non_null(self.kv_indices.ptr, "kv_indices")?;
        check_non_null(self.out.ptr, "out")?;
        check_non_null(self.float_workspace.ptr, "float_workspace")?;
        check_non_null(self.int_workspace.ptr, "int_workspace")?;

        check_positive("q_nope.dim0", self.q_nope.dim0)?;
        check_positive("q_nope.dim1", self.q_nope.dim1)?;
        check_positive("q_nope.dim2", self.q_nope.dim2)?;
        check_positive("q_pe.dim0", self.q_pe.dim0)?;
        check_positive("q_pe.dim1", self.q_pe.dim1)?;
        check_positive("q_pe.dim2", self.q_pe.dim2)?;
        check_positive("ckv_cache.dim0", self.ckv_cache.dim0)?;
        check_positive("ckv_cache.dim1", self.ckv_cache.dim1)?;
        check_positive("ckv_cache.dim2", self.ckv_cache.dim2)?;
        check_positive("kpe_cache.dim0", self.kpe_cache.dim0)?;
        check_positive("kpe_cache.dim1", self.kpe_cache.dim1)?;
        check_positive("kpe_cache.dim2", self.kpe_cache.dim2)?;
        check_positive("out.dim0", self.out.dim0)?;
        check_positive("out.dim1", self.out.dim1)?;
        check_positive("out.dim2", self.out.dim2)?;
        check_positive("num_heads", self.num_heads)?;
        check_positive("page_size", self.page_size)?;

        check_contiguous_1d("kv_indices", self.kv_indices.stride)?;
        check_contiguous_1d("float_workspace", self.float_workspace.stride)?;
        check_contiguous_1d("int_workspace", self.int_workspace.stride)?;

        if self.q_nope.stride2 != 1
            || self.q_pe.stride2 != 1
            || self.ckv_cache.stride2 != 1
            || self.kpe_cache.stride2 != 1
            || self.out.stride2 != 1
        {
            return Err(FlashInferError::invalid_argument(
                "MLA tensors require last-dim stride == 1",
            ));
        }

        if self.q_nope.dim0 != self.q_pe.dim0 {
            return Err(FlashInferError::invalid_argument(
                "q_nope.dim0 (n) must match q_pe.dim0",
            ));
        }
        if self.q_nope.dim1 != self.num_heads || self.q_pe.dim1 != self.num_heads {
            return Err(FlashInferError::invalid_argument(
                "q_nope/q_pe dim1 must equal num_heads",
            ));
        }
        if self.out.dim0 != self.q_nope.dim0 {
            return Err(FlashInferError::invalid_argument(
                "out.dim0 must match q_nope.dim0",
            ));
        }
        if self.out.dim1 != self.num_heads {
            return Err(FlashInferError::invalid_argument(
                "out.dim1 must equal num_heads",
            ));
        }
        if self.out.dim2 != self.q_nope.dim2 {
            return Err(FlashInferError::invalid_argument(
                "out.dim2 must equal q_nope.dim2 (head_dim_ckv)",
            ));
        }
        if self.ckv_cache.dim2 != self.q_nope.dim2 {
            return Err(FlashInferError::invalid_argument(
                "ckv_cache.dim2 must equal q_nope.dim2 (head_dim_ckv)",
            ));
        }
        if self.kpe_cache.dim2 != self.q_pe.dim2 {
            return Err(FlashInferError::invalid_argument(
                "kpe_cache.dim2 must equal q_pe.dim2 (head_dim_kpe)",
            ));
        }
        if self.ckv_cache.dim0 != self.kpe_cache.dim0 {
            return Err(FlashInferError::invalid_argument(
                "ckv_cache.dim0 (num_pages) must match kpe_cache.dim0",
            ));
        }
        if self.ckv_cache.dim1 != self.page_size || self.kpe_cache.dim1 != self.page_size {
            return Err(FlashInferError::invalid_argument(
                "ckv_cache.dim1/kpe_cache.dim1 must equal page_size",
            ));
        }

        if self.q_nope.dtype != self.q_pe.dtype || self.q_nope.dtype != self.out.dtype {
            return Err(FlashInferError::invalid_argument(
                "q_nope/q_pe/out dtype mismatch",
            ));
        }
        if self.ckv_cache.dtype != self.kpe_cache.dtype {
            return Err(FlashInferError::invalid_argument(
                "ckv_cache/kpe_cache dtype mismatch",
            ));
        }

        let device_id = self.q_nope.device_id;
        if self.q_pe.device_id != device_id
            || self.ckv_cache.device_id != device_id
            || self.kpe_cache.device_id != device_id
            || self.kv_indices.device_id != device_id
            || self.out.device_id != device_id
            || self.float_workspace.device_id != device_id
            || self.int_workspace.device_id != device_id
        {
            return Err(FlashInferError::invalid_argument(
                "all MLA tensors must be on the same CUDA device",
            ));
        }

        if !self.sm_scale.is_finite() || self.sm_scale <= 0.0 {
            return Err(FlashInferError::invalid_argument(
                "sm_scale must be finite and > 0",
            ));
        }

        if let Some(lse) = self.lse {
            check_non_null(lse.ptr, "lse")?;
            check_positive("lse.rows", lse.rows)?;
            check_positive("lse.cols", lse.cols)?;
            if lse.rows != self.q_nope.dim0 || lse.cols != self.num_heads {
                return Err(FlashInferError::invalid_argument(
                    "lse shape must be [n, num_heads]",
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
        Ok(())
    }
}

/// Plan-time parameters for `BatchMLAPagedAttentionPlan`.
///
/// Plan is a host-side scheduler call that allocates work distribution
/// inside the int workspace; the returned `MlaBatchPagedAttentionPlan`
/// holds the resulting `plan_info` vector for use by `run`.
#[derive(Debug, Clone, Copy)]
pub struct MlaBatchPagedAttentionPlanParams {
    /// Q/output ragged offsets on host, rank-1 int32: `[batch_size + 1]`.
    pub qo_indptr_host: MhaHostTensor1DI32Desc,
    /// KV ragged offsets on host, rank-1 int32: `[batch_size + 1]`.
    pub kv_indptr_host: MhaHostTensor1DI32Desc,
    /// KV sequence lengths on host, rank-1 int32: `[batch_size]`.
    /// (Note: this is the raw KV length, not the `last_page_len` that
    /// regular MHA paged plan uses.)
    pub kv_len_host: MhaHostTensor1DI32Desc,
    /// Float workspace on device, rank-1 u8.
    pub float_workspace: MhaTensor1DU8Desc,
    /// Int workspace on device, rank-1 u8.
    pub int_workspace: MhaTensor1DU8Desc,
    /// Page-locked int workspace on host, rank-1 u8.
    pub page_locked_int_workspace: MhaHostTensor1DU8Desc,
    /// Number of attention heads (rank-local for TP).
    pub num_heads: i64,
    /// Latent KV head dim (e.g. `kv_lora_rank=512` for DeepSeek-V3).
    pub head_dim_ckv: i64,
    /// Decoupled-rope head dim (e.g. `qk_rope_head_dim=64`).
    pub head_dim_kpe: i64,
    /// Output head dim (= `head_dim_ckv` for absorbed MLA).
    pub head_dim_o: i64,
    /// Whether causal masking is applied.
    pub causal: bool,
    /// Q dtype (typically BF16 / F16).
    pub q_dtype: DType,
    /// KV dtype (typically BF16 / F16).
    pub kv_dtype: DType,
    /// Output dtype.
    pub o_dtype: DType,
    /// CUDA device id.
    pub device_id: i32,
    /// Backend selector (FA2 generic vs Hopper SM90).
    pub backend: MlaBackend,
    /// Whether this kernel was JIT-compiled with profiler support.
    /// Affects only kernel URI / cache key.
    pub use_profiler: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl MlaBatchPagedAttentionPlanParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qo_indptr_host: MhaHostTensor1DI32Desc,
        kv_indptr_host: MhaHostTensor1DI32Desc,
        kv_len_host: MhaHostTensor1DI32Desc,
        float_workspace: MhaTensor1DU8Desc,
        int_workspace: MhaTensor1DU8Desc,
        page_locked_int_workspace: MhaHostTensor1DU8Desc,
        num_heads: i64,
        head_dim_ckv: i64,
        head_dim_kpe: i64,
        q_dtype: DType,
        kv_dtype: DType,
        o_dtype: DType,
        device_id: i32,
        stream: *mut c_void,
    ) -> Self {
        Self {
            qo_indptr_host,
            kv_indptr_host,
            kv_len_host,
            float_workspace,
            int_workspace,
            page_locked_int_workspace,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            head_dim_o: head_dim_ckv,
            causal: false,
            q_dtype,
            kv_dtype,
            o_dtype,
            device_id,
            backend: MlaBackend::Fa2,
            use_profiler: false,
            stream,
        }
    }

    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    pub fn with_backend(mut self, backend: MlaBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_use_profiler(mut self, use_profiler: bool) -> Self {
        self.use_profiler = use_profiler;
        self
    }

    pub fn with_head_dim_o(mut self, head_dim_o: i64) -> Self {
        self.head_dim_o = head_dim_o;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.qo_indptr_host.ptr, "qo_indptr_host")?;
        check_non_null(self.kv_indptr_host.ptr, "kv_indptr_host")?;
        check_non_null(self.kv_len_host.ptr, "kv_len_host")?;
        check_non_null(self.float_workspace.ptr, "float_workspace")?;
        check_non_null(self.int_workspace.ptr, "int_workspace")?;
        check_non_null(
            self.page_locked_int_workspace.ptr,
            "page_locked_int_workspace",
        )?;

        check_positive("num_heads", self.num_heads)?;
        check_positive("head_dim_ckv", self.head_dim_ckv)?;
        check_positive("head_dim_kpe", self.head_dim_kpe)?;
        check_positive("head_dim_o", self.head_dim_o)?;

        check_contiguous_1d("qo_indptr_host", self.qo_indptr_host.stride)?;
        check_contiguous_1d("kv_indptr_host", self.kv_indptr_host.stride)?;
        check_contiguous_1d("kv_len_host", self.kv_len_host.stride)?;
        check_contiguous_1d("float_workspace", self.float_workspace.stride)?;
        check_contiguous_1d("int_workspace", self.int_workspace.stride)?;
        check_contiguous_1d(
            "page_locked_int_workspace",
            self.page_locked_int_workspace.stride,
        )?;

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

        let qo_host = read_host_i32(self.qo_indptr_host)?;
        let kv_host = read_host_i32(self.kv_indptr_host)?;
        let kv_len_host = read_host_i32(self.kv_len_host)?;

        let batch_size = validate_indptr("qo_indptr_host", &qo_host)?;
        let kv_batch_size = validate_indptr("kv_indptr_host", &kv_host)?;
        if kv_batch_size != batch_size {
            return Err(FlashInferError::invalid_argument(format!(
                "kv_indptr_host batch size ({kv_batch_size}) must match qo_indptr_host batch size ({batch_size})"
            )));
        }
        let bs_usize = usize::try_from(batch_size)
            .map_err(|_| FlashInferError::invalid_argument("batch_size overflow"))?;
        if kv_len_host.len() != bs_usize {
            return Err(FlashInferError::invalid_argument(
                "kv_len_host length must equal batch_size",
            ));
        }
        if kv_len_host.iter().any(|v| *v < 0) {
            return Err(FlashInferError::invalid_argument(
                "kv_len_host values must be >= 0",
            ));
        }

        if self.float_workspace.device_id != self.device_id
            || self.int_workspace.device_id != self.device_id
        {
            return Err(FlashInferError::invalid_argument(
                "plan workspaces must match plan device_id",
            ));
        }
        Ok(())
    }

    fn kernel_uri(&self) -> String {
        // Mirrors `flashinfer/jit/attention/modules.py::get_batch_mla_uri`:
        //   batch_mla_attention_dtype_q_{q}_dtype_kv_{kv}_dtype_o_{o}
        //   _dtype_idx_{idx}_head_dim_ckv_{ckv}_head_dim_kpe_{kpe}
        //   _profiler_{prof}[ _sm90 ]
        let mut uri = format!(
            "batch_mla_attention_dtype_q_{}_dtype_kv_{}_dtype_o_{}_dtype_idx_i32_head_dim_ckv_{}_head_dim_kpe_{}_profiler_{}",
            dtype_filename(self.q_dtype),
            dtype_filename(self.kv_dtype),
            dtype_filename(self.o_dtype),
            self.head_dim_ckv,
            self.head_dim_kpe,
            bool_python(self.use_profiler),
        );
        if self.backend == MlaBackend::Hopper {
            uri.push_str("_sm90");
        }
        uri
    }
}

/// Owned plan handle returned by `mla_batch_paged_plan`. Holds the
/// underlying TVM-FFI object reference (the `MLAPlanInfo` int64 vector)
/// and decrements it on `Drop`.
pub struct MlaBatchPagedAttentionPlan {
    runtime: &'static FlashInferRuntime,
    plan_result: TVMFFIAny,
    kernel_uri: String,
    device_id: i32,
    batch_size: i64,
    num_heads: i64,
    head_dim_ckv: i64,
    head_dim_kpe: i64,
    head_dim_o: i64,
    q_dtype: DType,
    kv_dtype: DType,
    o_dtype: DType,
    causal: bool,
    float_workspace_len: i64,
    int_workspace_len: i64,
}

impl MlaBatchPagedAttentionPlan {
    pub fn batch_size(&self) -> i64 {
        self.batch_size
    }
    pub fn num_heads(&self) -> i64 {
        self.num_heads
    }
    pub fn head_dim_ckv(&self) -> i64 {
        self.head_dim_ckv
    }
    pub fn head_dim_kpe(&self) -> i64 {
        self.head_dim_kpe
    }

    fn validate_run_compatibility(
        &self,
        params: &MlaBatchPagedAttentionParams,
    ) -> Result<(), FlashInferError> {
        if params.q_nope.dtype != self.q_dtype
            || params.q_pe.dtype != self.q_dtype
            || params.ckv_cache.dtype != self.kv_dtype
            || params.kpe_cache.dtype != self.kv_dtype
            || params.out.dtype != self.o_dtype
        {
            return Err(FlashInferError::invalid_argument(
                "MLA run dtype mismatch with plan",
            ));
        }
        if params.q_nope.device_id != self.device_id {
            return Err(FlashInferError::invalid_argument(
                "MLA run device mismatch with plan",
            ));
        }
        if params.num_heads != self.num_heads {
            return Err(FlashInferError::invalid_argument(
                "MLA run num_heads mismatch with plan",
            ));
        }
        if params.q_nope.dim2 != self.head_dim_ckv {
            return Err(FlashInferError::invalid_argument(
                "MLA run head_dim_ckv mismatch with plan",
            ));
        }
        if params.q_pe.dim2 != self.head_dim_kpe {
            return Err(FlashInferError::invalid_argument(
                "MLA run head_dim_kpe mismatch with plan",
            ));
        }
        if params.out.dim2 != self.head_dim_o {
            return Err(FlashInferError::invalid_argument(
                "MLA run head_dim_o mismatch with plan",
            ));
        }
        if params.float_workspace.len != self.float_workspace_len
            || params.int_workspace.len != self.int_workspace_len
        {
            return Err(FlashInferError::invalid_argument(
                "MLA run workspace size mismatch with plan",
            ));
        }
        if self.causal && params.mask_mode != MhaMaskMode::Causal {
            return Err(FlashInferError::invalid_argument(
                "MLA plan was causal but run mask_mode is not Causal",
            ));
        }
        if !self.causal && params.mask_mode == MhaMaskMode::Causal {
            return Err(FlashInferError::invalid_argument(
                "MLA plan was non-causal but run mask_mode is Causal",
            ));
        }
        Ok(())
    }
}

impl Drop for MlaBatchPagedAttentionPlan {
    fn drop(&mut self) {
        if let Some(obj) = any_object_handle(&self.plan_result) {
            // SAFETY: object handle was created by TVM-FFI plan call.
            unsafe { self.runtime.object_dec_ref(obj) };
        }
    }
}

pub fn mla_batch_paged_plan(
    params: &MlaBatchPagedAttentionPlanParams,
) -> Result<MlaBatchPagedAttentionPlan, FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { mla_batch_paged_plan_with_runtime(runtime, params) }
}

unsafe fn mla_batch_paged_plan_with_runtime(
    runtime: &'static FlashInferRuntime,
    params: &MlaBatchPagedAttentionPlanParams,
) -> Result<MlaBatchPagedAttentionPlan, FlashInferError> {
    let kernel_uri = params.kernel_uri();
    let qo_host = read_host_i32(params.qo_indptr_host)?;
    let batch_size = i64::try_from(qo_host.len().saturating_sub(1))
        .map_err(|_| FlashInferError::invalid_argument("batch_size does not fit in i64"))?;

    // Build DLTensor views for all six args.
    let mut qo_shape = [params.qo_indptr_host.len];
    let mut qo_strides = [params.qo_indptr_host.stride];
    let qo_indptr_t = host_i32_dl_tensor(params.qo_indptr_host.ptr, &mut qo_shape, &mut qo_strides);

    let mut kv_indptr_shape = [params.kv_indptr_host.len];
    let mut kv_indptr_strides = [params.kv_indptr_host.stride];
    let kv_indptr_t = host_i32_dl_tensor(
        params.kv_indptr_host.ptr,
        &mut kv_indptr_shape,
        &mut kv_indptr_strides,
    );

    let mut kv_len_shape = [params.kv_len_host.len];
    let mut kv_len_strides = [params.kv_len_host.stride];
    let kv_len_t = host_i32_dl_tensor(params.kv_len_host.ptr, &mut kv_len_shape, &mut kv_len_strides);

    let mut float_ws_shape = [params.float_workspace.len];
    let mut float_ws_strides = [params.float_workspace.stride];
    let float_ws_t = device_u8_dl_tensor(
        params.float_workspace.ptr,
        params.float_workspace.device_id,
        &mut float_ws_shape,
        &mut float_ws_strides,
    );

    let mut int_ws_shape = [params.int_workspace.len];
    let mut int_ws_strides = [params.int_workspace.stride];
    let int_ws_t = device_u8_dl_tensor(
        params.int_workspace.ptr,
        params.int_workspace.device_id,
        &mut int_ws_shape,
        &mut int_ws_strides,
    );

    let mut pl_ws_shape = [params.page_locked_int_workspace.len];
    let mut pl_ws_strides = [params.page_locked_int_workspace.stride];
    let pl_ws_t = host_u8_dl_tensor(
        params.page_locked_int_workspace.ptr,
        &mut pl_ws_shape,
        &mut pl_ws_strides,
    );

    let mut plan_result_view = any_none();
    // Order matches `BatchMLAPagedAttentionPlan(...)` in batch_mla_binding.cu:
    //   (float_ws, int_ws, page_locked_int_ws, qo_indptr, kv_indptr, kv_len,
    //    num_heads, head_dim_o, causal)
    let plan_args: [TVMFFIAny; 9] = [
        any_dltensor_ptr(&float_ws_t),
        any_dltensor_ptr(&int_ws_t),
        any_dltensor_ptr(&pl_ws_t),
        any_dltensor_ptr(&qo_indptr_t),
        any_dltensor_ptr(&kv_indptr_t),
        any_dltensor_ptr(&kv_len_t),
        any_i64(params.num_heads),
        any_i64(params.head_dim_o),
        any_bool(params.causal),
    ];

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.device_id, params.stream)? };
    let mut restore_guard = StreamRestoreGuard::new(runtime, params.device_id, previous_stream);

    let call_result = (|| -> Result<MlaBatchPagedAttentionPlan, FlashInferError> {
        // SAFETY: argument packing follows TVMFFIAny ABI.
        unsafe {
            runtime.call_batch_mla_plan(
                &kernel_uri,
                plan_args.as_ptr(),
                plan_args.len() as i32,
                &mut plan_result_view as *mut _,
            )?;
        }
        // SAFETY: converts returned AnyView into owned Any so lifetime can cross calls safely.
        let plan_result = unsafe { runtime.any_view_to_owned(&plan_result_view)? };
        Ok(MlaBatchPagedAttentionPlan {
            runtime,
            plan_result,
            kernel_uri,
            device_id: params.device_id,
            batch_size,
            num_heads: params.num_heads,
            head_dim_ckv: params.head_dim_ckv,
            head_dim_kpe: params.head_dim_kpe,
            head_dim_o: params.head_dim_o,
            q_dtype: params.q_dtype,
            kv_dtype: params.kv_dtype,
            o_dtype: params.o_dtype,
            causal: params.causal,
            float_workspace_len: params.float_workspace.len,
            int_workspace_len: params.int_workspace.len,
        })
    })();

    let restore_result = restore_guard.restore_now();
    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(plan), Err(restore_error)) => {
            drop(plan);
            Err(restore_error)
        }
        (Ok(plan), Ok(())) => Ok(plan),
    }
}

pub fn mla_batch_paged_run(
    plan: &MlaBatchPagedAttentionPlan,
    params: &MlaBatchPagedAttentionParams,
) -> Result<(), FlashInferError> {
    params.validate()?;
    plan.validate_run_compatibility(params)?;
    // SAFETY: FFI preconditions validated above.
    unsafe { mla_batch_paged_run_with_runtime(plan.runtime, plan, params) }
}

unsafe fn mla_batch_paged_run_with_runtime(
    runtime: &FlashInferRuntime,
    plan: &MlaBatchPagedAttentionPlan,
    params: &MlaBatchPagedAttentionParams,
) -> Result<(), FlashInferError> {
    // Build DLTensor views for the seven device tensor arguments and the
    // optional LSE.
    let mut float_ws_shape = [params.float_workspace.len];
    let mut float_ws_strides = [params.float_workspace.stride];
    let float_ws_t = device_u8_dl_tensor(
        params.float_workspace.ptr,
        params.float_workspace.device_id,
        &mut float_ws_shape,
        &mut float_ws_strides,
    );

    let mut int_ws_shape = [params.int_workspace.len];
    let mut int_ws_strides = [params.int_workspace.stride];
    let int_ws_t = device_u8_dl_tensor(
        params.int_workspace.ptr,
        params.int_workspace.device_id,
        &mut int_ws_shape,
        &mut int_ws_strides,
    );

    let mut q_nope_shape = [params.q_nope.dim0, params.q_nope.dim1, params.q_nope.dim2];
    let mut q_nope_strides = [
        params.q_nope.stride0,
        params.q_nope.stride1,
        params.q_nope.stride2,
    ];
    let q_nope_t = device_3d_dl_tensor(
        params.q_nope.ptr,
        params.q_nope.device_id,
        params.q_nope.dtype,
        &mut q_nope_shape,
        &mut q_nope_strides,
    );

    let mut q_pe_shape = [params.q_pe.dim0, params.q_pe.dim1, params.q_pe.dim2];
    let mut q_pe_strides = [
        params.q_pe.stride0,
        params.q_pe.stride1,
        params.q_pe.stride2,
    ];
    let q_pe_t = device_3d_dl_tensor(
        params.q_pe.ptr,
        params.q_pe.device_id,
        params.q_pe.dtype,
        &mut q_pe_shape,
        &mut q_pe_strides,
    );

    let mut ckv_shape = [
        params.ckv_cache.dim0,
        params.ckv_cache.dim1,
        params.ckv_cache.dim2,
    ];
    let mut ckv_strides = [
        params.ckv_cache.stride0,
        params.ckv_cache.stride1,
        params.ckv_cache.stride2,
    ];
    let ckv_t = device_3d_dl_tensor(
        params.ckv_cache.ptr,
        params.ckv_cache.device_id,
        params.ckv_cache.dtype,
        &mut ckv_shape,
        &mut ckv_strides,
    );

    let mut kpe_shape = [
        params.kpe_cache.dim0,
        params.kpe_cache.dim1,
        params.kpe_cache.dim2,
    ];
    let mut kpe_strides = [
        params.kpe_cache.stride0,
        params.kpe_cache.stride1,
        params.kpe_cache.stride2,
    ];
    let kpe_t = device_3d_dl_tensor(
        params.kpe_cache.ptr,
        params.kpe_cache.device_id,
        params.kpe_cache.dtype,
        &mut kpe_shape,
        &mut kpe_strides,
    );

    let mut kv_idx_shape = [params.kv_indices.len];
    let mut kv_idx_strides = [params.kv_indices.stride];
    let kv_idx_t = device_i32_dl_tensor(
        params.kv_indices.ptr,
        params.kv_indices.device_id,
        &mut kv_idx_shape,
        &mut kv_idx_strides,
    );

    let mut o_shape = [params.out.dim0, params.out.dim1, params.out.dim2];
    let mut o_strides = [
        params.out.stride0,
        params.out.stride1,
        params.out.stride2,
    ];
    let o_t = device_3d_dl_tensor(
        params.out.ptr,
        params.out.device_id,
        params.out.dtype,
        &mut o_shape,
        &mut o_strides,
    );

    let mut lse_shape = [0_i64; 2];
    let mut lse_strides = [0_i64; 2];
    let lse_t = params.lse.map(|lse| {
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
    let lse_any = lse_t
        .as_ref()
        .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));

    let mut run_result = any_none();
    // Order matches `BatchMLAPagedAttentionRun(...)` in batch_mla_binding.cu:
    //   (float_ws, int_ws, plan_info, q_nope, q_pe, ckv, kpe, kv_indices,
    //    o, maybe_lse, mask_mode_code, num_heads, page_size, sm_scale,
    //    return_lse_base_on_e)
    let run_args: [TVMFFIAny; 15] = [
        any_dltensor_ptr(&float_ws_t),
        any_dltensor_ptr(&int_ws_t),
        plan.plan_result,
        any_dltensor_ptr(&q_nope_t),
        any_dltensor_ptr(&q_pe_t),
        any_dltensor_ptr(&ckv_t),
        any_dltensor_ptr(&kpe_t),
        any_dltensor_ptr(&kv_idx_t),
        any_dltensor_ptr(&o_t),
        lse_any,
        any_i64(mask_mode_code(params.mask_mode)),
        any_i64(params.num_heads),
        any_i64(params.page_size),
        any_f64(params.sm_scale),
        any_bool(params.return_lse_base_on_e),
    ];

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.q_nope.device_id, params.stream)? };
    let mut restore_guard =
        StreamRestoreGuard::new(runtime, params.q_nope.device_id, previous_stream);

    // SAFETY: argument packing follows TVMFFIAny ABI.
    let call_result = unsafe {
        runtime.call_batch_mla_run(
            &plan.kernel_uri,
            run_args.as_ptr(),
            run_args.len() as i32,
            &mut run_result as *mut _,
        )
    };
    let restore_result = restore_guard.restore_now();

    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

// ---- DLTensor builders ----

fn host_i32_dl_tensor(ptr: *const c_void, shape: &mut [i64; 1], strides: &mut [i64; 1]) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn host_u8_dl_tensor(ptr: *const c_void, shape: &mut [i64; 1], strides: &mut [i64; 1]) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: dl_dtype_u8(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn device_u8_dl_tensor(
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
        dtype: dl_dtype_u8(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn device_i32_dl_tensor(
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
        dtype: dl_dtype_i32(),
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    }
}

fn device_3d_dl_tensor(
    ptr: *const c_void,
    device_id: i32,
    dtype: DType,
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

// ---- Stream restore guard (mirrors mha_batch_prefill_paged.rs) ----

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

// ---- Validation helpers ----

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
            "host i32 length must be positive",
        ));
    }
    if desc.stride <= 0 {
        return Err(FlashInferError::invalid_argument(
            "host i32 stride must be positive",
        ));
    }
    let len = usize::try_from(desc.len)
        .map_err(|_| FlashInferError::invalid_argument("host i32 length does not fit in usize"))?;
    let stride = usize::try_from(desc.stride)
        .map_err(|_| FlashInferError::invalid_argument("host i32 stride does not fit in usize"))?;
    let base = desc.ptr.cast::<i32>();
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        // SAFETY: caller provides valid host memory at requested offset.
        let v = unsafe { *base.add(i.saturating_mul(stride)) };
        out.push(v);
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
    if indptr[indptr.len() - 1] < 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} last value must be >= 0"
        )));
    }
    u32::try_from(indptr.len() - 1)
        .map_err(|_| FlashInferError::invalid_argument(format!("{name} batch size overflow")))
}

fn dtype_filename(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F8E4M3FN => "e4m3",
    }
}

fn bool_python(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn mask_mode_code(mask_mode: MhaMaskMode) -> i64 {
    match mask_mode {
        MhaMaskMode::NonCausal => 0,
        MhaMaskMode::Causal => 1,
        MhaMaskMode::Custom => 2,
        MhaMaskMode::MultiItemScoring => 3,
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
            code: 10, // KDL_FLOAT8_E4M3FN
            bits: 8,
            lanes: 1,
        },
    }
}

fn dl_dtype_i32() -> DLDataType {
    DLDataType {
        code: KDL_INT,
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

fn dl_dtype_f32() -> DLDataType {
    DLDataType {
        code: KDL_FLOAT,
        bits: 32,
        lanes: 1,
    }
}
