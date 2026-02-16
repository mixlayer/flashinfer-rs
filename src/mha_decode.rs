use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CPU, KDL_CUDA, KDL_FLOAT, KDL_INT, KDL_UINT,
    TVMFFIAny, any_bool, any_dltensor_ptr, any_f64, any_i64, any_none, any_object_handle,
};
use crate::mha_batch_prefill::{MhaHostTensor1DI32Desc, MhaHostTensor1DU8Desc, MhaTensor1DI32Desc};
use crate::mha_batch_prefill_paged::MhaTensor4DDesc;
use crate::mha_prefill::{
    MhaPosEncodingMode, MhaQkvLayout, MhaTensor1DF32Desc, MhaTensor1DU8Desc, MhaTensor2DF32Desc,
    MhaTensor3DDesc,
};
use crate::norm::DType;
use crate::runtime::FlashInferRuntime;

#[derive(Debug, Clone, Copy)]
pub struct MhaTensor2DDesc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub dtype: DType,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct MhaSingleDecodeParams {
    /// Query tensor, rank-2: `[num_qo_heads, head_dim_qk]`.
    ///
    /// Cross-reference:
    /// `flashinfer/flashinfer/decode.py::single_decode_with_kv_cache`.
    pub q: MhaTensor2DDesc,
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
    /// Output tensor, rank-2: `[num_qo_heads, head_dim_vo]`.
    pub out: MhaTensor2DDesc,
    /// Optional log-sum-exp output, rank-1 f32: `[num_qo_heads]`.
    pub lse: Option<MhaTensor1DF32Desc>,
    /// Optional ALiBi slopes, rank-1 f32: `[num_qo_heads]`.
    pub alibi_slopes: Option<MhaTensor1DF32Desc>,
    /// KV layout enum (`NHD` or `HND`).
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
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl MhaSingleDecodeParams {
    pub fn new(
        q: MhaTensor2DDesc,
        k: MhaTensor3DDesc,
        v: MhaTensor3DDesc,
        tmp: MhaTensor1DU8Desc,
        out: MhaTensor2DDesc,
        stream: *mut c_void,
    ) -> Self {
        let sm_scale = if q.cols > 0 {
            1.0 / (q.cols as f64).sqrt()
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
            alibi_slopes: None,
            kv_layout: MhaQkvLayout::Nhd,
            pos_encoding_mode: MhaPosEncodingMode::None,
            window_left: -1,
            logits_soft_cap: 0.0,
            sm_scale,
            rope_scale: 1.0,
            rope_theta: 1e4,
            stream,
        }
    }

    pub fn with_lse(mut self, lse: MhaTensor1DF32Desc) -> Self {
        self.lse = Some(lse);
        self
    }

    pub fn with_alibi_slopes(mut self, alibi_slopes: MhaTensor1DF32Desc) -> Self {
        self.alibi_slopes = Some(alibi_slopes);
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

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.q.ptr, "q")?;
        check_non_null(self.k.ptr, "k")?;
        check_non_null(self.v.ptr, "v")?;
        check_non_null(self.tmp.ptr, "tmp")?;
        check_non_null(self.out.ptr, "out")?;

        check_positive("q.rows", self.q.rows)?;
        check_positive("q.cols", self.q.cols)?;
        check_positive("k.dim0", self.k.dim0)?;
        check_positive("k.dim1", self.k.dim1)?;
        check_positive("k.dim2", self.k.dim2)?;
        check_positive("v.dim0", self.v.dim0)?;
        check_positive("v.dim1", self.v.dim1)?;
        check_positive("v.dim2", self.v.dim2)?;
        check_positive("out.rows", self.out.rows)?;
        check_positive("out.cols", self.out.cols)?;
        check_positive("tmp.len", self.tmp.len)?;

        if self.tmp.stride != 1 {
            return Err(FlashInferError::invalid_argument("tmp stride must be 1"));
        }

        if self.q.stride_col != 1 || self.q.stride_row != self.q.cols {
            return Err(FlashInferError::invalid_argument(
                "q must be contiguous row-major [num_qo_heads, head_dim_qk]",
            ));
        }
        if self.out.stride_col != 1 || self.out.stride_row != self.out.cols {
            return Err(FlashInferError::invalid_argument(
                "out must be contiguous row-major [num_qo_heads, head_dim_vo]",
            ));
        }
        if self.k.stride2 != 1 || self.v.stride2 != 1 {
            return Err(FlashInferError::invalid_argument(
                "k/v last-dimension stride must be 1",
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

        let num_qo_heads = self.q.rows;
        let head_dim_qk = self.q.cols;

        if self.k.dim2 != head_dim_qk {
            return Err(FlashInferError::invalid_argument(
                "k head_dim_qk must match q",
            ));
        }
        if self.out.rows != num_qo_heads {
            return Err(FlashInferError::invalid_argument(
                "out.rows must equal num_qo_heads",
            ));
        }
        if self.out.cols != self.v.dim2 {
            return Err(FlashInferError::invalid_argument(
                "out head_dim_vo must match v",
            ));
        }
        if self.out.cols != head_dim_qk {
            return Err(FlashInferError::invalid_argument(
                "decode kernels require head_dim_qk == head_dim_vo",
            ));
        }

        let (kv_len_k, num_kv_heads_k, k_expected_stride0, k_expected_stride1) =
            match self.kv_layout {
                MhaQkvLayout::Nhd => (
                    self.k.dim0,
                    self.k.dim1,
                    self.k
                        .dim1
                        .checked_mul(self.k.dim2)
                        .ok_or_else(|| FlashInferError::invalid_argument("k stride overflow"))?,
                    self.k.dim2,
                ),
                MhaQkvLayout::Hnd => (
                    self.k.dim1,
                    self.k.dim0,
                    self.k
                        .dim1
                        .checked_mul(self.k.dim2)
                        .ok_or_else(|| FlashInferError::invalid_argument("k stride overflow"))?,
                    self.k.dim2,
                ),
            };
        let (kv_len_v, num_kv_heads_v, v_expected_stride0, v_expected_stride1) =
            match self.kv_layout {
                MhaQkvLayout::Nhd => (
                    self.v.dim0,
                    self.v.dim1,
                    self.v
                        .dim1
                        .checked_mul(self.v.dim2)
                        .ok_or_else(|| FlashInferError::invalid_argument("v stride overflow"))?,
                    self.v.dim2,
                ),
                MhaQkvLayout::Hnd => (
                    self.v.dim1,
                    self.v.dim0,
                    self.v
                        .dim1
                        .checked_mul(self.v.dim2)
                        .ok_or_else(|| FlashInferError::invalid_argument("v stride overflow"))?,
                    self.v.dim2,
                ),
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

        if self.k.stride1 != k_expected_stride1 || self.k.stride0 != k_expected_stride0 {
            return Err(FlashInferError::invalid_argument(
                "k must be contiguous for the selected kv_layout",
            ));
        }
        if self.v.stride1 != v_expected_stride1 || self.v.stride0 != v_expected_stride0 {
            return Err(FlashInferError::invalid_argument(
                "v must be contiguous for the selected kv_layout",
            ));
        }

        let device_id = self.q.device_id;
        if self.k.device_id != device_id
            || self.v.device_id != device_id
            || self.tmp.device_id != device_id
            || self.out.device_id != device_id
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
            check_positive("lse.len", lse.len)?;
            if lse.len != num_qo_heads {
                return Err(FlashInferError::invalid_argument(
                    "lse length must equal num_qo_heads",
                ));
            }
            if lse.stride != 1 {
                return Err(FlashInferError::invalid_argument("lse stride must be 1"));
            }
            if lse.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "lse must be on the same CUDA device",
                ));
            }
        }

        if let Some(alibi) = self.alibi_slopes {
            check_non_null(alibi.ptr, "alibi_slopes")?;
            check_positive("alibi_slopes.len", alibi.len)?;
            if alibi.len != num_qo_heads {
                return Err(FlashInferError::invalid_argument(
                    "alibi_slopes length must equal num_qo_heads",
                ));
            }
            if alibi.stride != 1 {
                return Err(FlashInferError::invalid_argument(
                    "alibi_slopes stride must be 1",
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

        Ok(())
    }

    fn kernel_uri(&self) -> String {
        format!(
            "single_decode_with_kv_cache_dtype_q_{}_dtype_kv_{}_dtype_o_{}_head_dim_qk_{}_head_dim_vo_{}_posenc_{}_use_swa_{}_use_logits_cap_{}",
            dtype_filename(self.q.dtype),
            dtype_filename(self.k.dtype),
            dtype_filename(self.out.dtype),
            self.q.cols,
            self.out.cols,
            pos_encoding_mode_code(self.pos_encoding_mode),
            bool_name(self.window_left >= 0),
            bool_name(self.logits_soft_cap > 0.0),
        )
    }
}

pub fn mha_single_decode(params: &MhaSingleDecodeParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { mha_single_decode_with_runtime(runtime, params) }
}

unsafe fn mha_single_decode_with_runtime(
    runtime: &FlashInferRuntime,
    params: &MhaSingleDecodeParams,
) -> Result<(), FlashInferError> {
    let kernel_uri = params.kernel_uri();

    let mut q_shape = [params.q.rows, params.q.cols];
    let mut q_strides = [params.q.stride_row, params.q.stride_col];
    let q_tensor = DLTensor {
        data: params.q.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.q.device_id,
        },
        ndim: 2,
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

    let mut out_shape = [params.out.rows, params.out.cols];
    let mut out_strides = [params.out.stride_row, params.out.stride_col];
    let out_tensor = DLTensor {
        data: params.out.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.out.device_id,
        },
        ndim: 2,
        dtype: dl_dtype_from_norm_dtype(params.out.dtype),
        shape: out_shape.as_mut_ptr(),
        strides: out_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut lse_shape = [0_i64; 1];
    let mut lse_strides = [0_i64; 1];
    let lse_tensor = params.lse.map(|lse| {
        lse_shape = [lse.len];
        lse_strides = [lse.stride];
        DLTensor {
            data: lse.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: lse.device_id,
            },
            ndim: 1,
            dtype: dl_dtype_f32(),
            shape: lse_shape.as_mut_ptr(),
            strides: lse_strides.as_mut_ptr(),
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
    let alibi_any = alibi_tensor
        .as_ref()
        .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));

    let args: [TVMFFIAny; 13] = [
        any_dltensor_ptr(&q_tensor),
        any_dltensor_ptr(&k_tensor),
        any_dltensor_ptr(&v_tensor),
        any_dltensor_ptr(&tmp_tensor),
        any_dltensor_ptr(&out_tensor),
        lse_any,
        any_i64(kv_layout_code(params.kv_layout)),
        any_i64(params.window_left),
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
        runtime.call_single_decode(
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

#[derive(Debug, Clone, Copy)]
pub struct MhaBatchPagedDecodePlanParams {
    /// Paged-KV page indptr on host, rank-1 int32: `[batch_size + 1]`.
    pub paged_kv_indptr_host: MhaHostTensor1DI32Desc,
    /// Float workspace on device, rank-1 u8: `[workspace_bytes]`.
    pub float_workspace: MhaTensor1DU8Desc,
    /// Int workspace on device, rank-1 u8: `[workspace_bytes]`.
    pub int_workspace: MhaTensor1DU8Desc,
    /// Page-locked int workspace on host, rank-1 u8: `[workspace_bytes]`.
    pub page_locked_int_workspace: MhaHostTensor1DU8Desc,
    /// Batch size.
    pub batch_size: i64,
    /// Number of query/output heads.
    pub num_qo_heads: i64,
    /// Number of KV heads.
    pub num_kv_heads: i64,
    /// Paged-KV page size.
    pub page_size: i64,
    /// Query/key head dimension.
    pub head_dim_qk: i64,
    /// Value/output head dimension.
    pub head_dim_vo: i64,
    /// Kernel dtype.
    pub dtype: DType,
    /// CUDA device id.
    pub device_id: i32,
    /// KV layout enum (`NHD` or `HND`).
    pub kv_layout: MhaQkvLayout,
    /// Positional encoding mode enum.
    pub pos_encoding_mode: MhaPosEncodingMode,
    /// Left sliding-window size; `-1` disables sliding window.
    pub window_left: i64,
    /// Logits soft cap value; `> 0` enables capping.
    pub logits_soft_cap: f64,
    /// Whether plan generation should support CUDA graph mode.
    pub enable_cuda_graph: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl MhaBatchPagedDecodePlanParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        paged_kv_indptr_host: MhaHostTensor1DI32Desc,
        float_workspace: MhaTensor1DU8Desc,
        int_workspace: MhaTensor1DU8Desc,
        page_locked_int_workspace: MhaHostTensor1DU8Desc,
        batch_size: i64,
        num_qo_heads: i64,
        num_kv_heads: i64,
        page_size: i64,
        head_dim_qk: i64,
        head_dim_vo: i64,
        dtype: DType,
        device_id: i32,
        stream: *mut c_void,
    ) -> Self {
        Self {
            paged_kv_indptr_host,
            float_workspace,
            int_workspace,
            page_locked_int_workspace,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            head_dim_qk,
            head_dim_vo,
            dtype,
            device_id,
            kv_layout: MhaQkvLayout::Nhd,
            pos_encoding_mode: MhaPosEncodingMode::None,
            window_left: -1,
            logits_soft_cap: 0.0,
            enable_cuda_graph: false,
            stream,
        }
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

    pub fn with_enable_cuda_graph(mut self, enable_cuda_graph: bool) -> Self {
        self.enable_cuda_graph = enable_cuda_graph;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.paged_kv_indptr_host.ptr, "paged_kv_indptr_host")?;
        check_non_null(self.float_workspace.ptr, "float_workspace")?;
        check_non_null(self.int_workspace.ptr, "int_workspace")?;
        check_non_null(
            self.page_locked_int_workspace.ptr,
            "page_locked_int_workspace",
        )?;

        check_positive("batch_size", self.batch_size)?;
        check_positive("num_qo_heads", self.num_qo_heads)?;
        check_positive("num_kv_heads", self.num_kv_heads)?;
        check_positive("page_size", self.page_size)?;
        check_positive("head_dim_qk", self.head_dim_qk)?;
        check_positive("head_dim_vo", self.head_dim_vo)?;

        if self.head_dim_qk != self.head_dim_vo {
            return Err(FlashInferError::invalid_argument(
                "decode kernels require head_dim_qk == head_dim_vo",
            ));
        }
        if self.num_qo_heads % self.num_kv_heads != 0 {
            return Err(FlashInferError::invalid_argument(
                "num_qo_heads must be divisible by num_kv_heads",
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

        if self.float_workspace.device_id != self.device_id
            || self.int_workspace.device_id != self.device_id
        {
            return Err(FlashInferError::invalid_argument(
                "plan workspaces must match plan device_id",
            ));
        }
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

        check_contiguous_1d("paged_kv_indptr_host", self.paged_kv_indptr_host.stride)?;
        check_contiguous_1d("float_workspace", self.float_workspace.stride)?;
        check_contiguous_1d("int_workspace", self.int_workspace.stride)?;
        check_contiguous_1d(
            "page_locked_int_workspace",
            self.page_locked_int_workspace.stride,
        )?;

        if self.paged_kv_indptr_host.len != self.batch_size + 1 {
            return Err(FlashInferError::invalid_argument(
                "paged_kv_indptr_host length must equal batch_size + 1",
            ));
        }

        let paged_kv_host = read_host_i32(self.paged_kv_indptr_host)?;
        let host_batch = validate_indptr("paged_kv_indptr_host", &paged_kv_host)?;
        if i64::from(host_batch) != self.batch_size {
            return Err(FlashInferError::invalid_argument(format!(
                "paged_kv_indptr_host batch size ({host_batch}) must equal batch_size ({})",
                self.batch_size
            )));
        }

        Ok(())
    }

    fn kernel_uri(&self) -> String {
        format!(
            "batch_decode_with_kv_cache_dtype_q_{}_dtype_kv_{}_dtype_o_{}_dtype_idx_i32_head_dim_qk_{}_head_dim_vo_{}_posenc_{}_use_swa_{}_use_logits_cap_{}",
            dtype_filename(self.dtype),
            dtype_filename(self.dtype),
            dtype_filename(self.dtype),
            self.head_dim_qk,
            self.head_dim_vo,
            pos_encoding_mode_code(self.pos_encoding_mode),
            bool_name(self.window_left >= 0),
            bool_name(self.logits_soft_cap > 0.0),
        )
    }
}

pub struct MhaBatchPagedDecodePlan {
    runtime: &'static FlashInferRuntime,
    plan_result: TVMFFIAny,
    kernel_uri: String,
    device_id: i32,
    batch_size: i64,
    num_qo_heads: i64,
    num_kv_heads: i64,
    page_size: i64,
    head_dim_qk: i64,
    head_dim_vo: i64,
    dtype: DType,
    kv_layout: MhaQkvLayout,
    pos_encoding_mode: MhaPosEncodingMode,
    window_left: i64,
    logits_soft_cap: f64,
    enable_cuda_graph: bool,
    float_workspace_len: i64,
    int_workspace_len: i64,
}

impl Drop for MhaBatchPagedDecodePlan {
    fn drop(&mut self) {
        if let Some(obj) = any_object_handle(&self.plan_result) {
            // SAFETY: object handle was created by TVM-FFI plan call and owned by this handle.
            unsafe {
                self.runtime.object_dec_ref(obj);
            }
        }
    }
}

impl MhaBatchPagedDecodePlan {
    fn validate_run_compatibility(
        &self,
        params: &MhaBatchPagedDecodeParams,
    ) -> Result<(), FlashInferError> {
        if params.kernel_uri() != self.kernel_uri {
            return Err(FlashInferError::invalid_argument(format!(
                "decode run kernel mismatch: planned `{}` but run requested `{}`",
                self.kernel_uri,
                params.kernel_uri()
            )));
        }
        if params.q.device_id != self.device_id {
            return Err(FlashInferError::invalid_argument(format!(
                "decode run device mismatch: planned device {} but got {}",
                self.device_id, params.q.device_id
            )));
        }
        if params.q.dim0 != self.batch_size {
            return Err(FlashInferError::invalid_argument(format!(
                "decode run batch mismatch: planned {} but got {}",
                self.batch_size, params.q.dim0
            )));
        }
        if params.q.dim1 != self.num_qo_heads {
            return Err(FlashInferError::invalid_argument(format!(
                "decode run num_qo_heads mismatch: planned {} but got {}",
                self.num_qo_heads, params.q.dim1
            )));
        }
        if params.q.dim2 != self.head_dim_qk || params.out.dim2 != self.head_dim_vo {
            return Err(FlashInferError::invalid_argument(format!(
                "decode run head dim mismatch: planned qk/vo=({},{}) but got ({},{})",
                self.head_dim_qk, self.head_dim_vo, params.q.dim2, params.out.dim2
            )));
        }
        if params.q.dtype != self.dtype
            || params.paged_k_cache.dtype != self.dtype
            || params.out.dtype != self.dtype
        {
            return Err(FlashInferError::invalid_argument(
                "decode run dtype mismatch with decode plan",
            ));
        }
        if params.kv_layout != self.kv_layout {
            return Err(FlashInferError::invalid_argument(
                "decode run kv_layout mismatch with decode plan",
            ));
        }
        if params.pos_encoding_mode != self.pos_encoding_mode {
            return Err(FlashInferError::invalid_argument(
                "decode run pos_encoding_mode mismatch with decode plan",
            ));
        }
        if params.window_left != self.window_left {
            return Err(FlashInferError::invalid_argument(
                "decode run window_left mismatch with decode plan",
            ));
        }
        if params.logits_soft_cap != self.logits_soft_cap {
            return Err(FlashInferError::invalid_argument(
                "decode run logits_soft_cap mismatch with decode plan",
            ));
        }
        if params.enable_cuda_graph != self.enable_cuda_graph {
            return Err(FlashInferError::invalid_argument(
                "decode run enable_cuda_graph mismatch with decode plan",
            ));
        }
        if params.float_workspace.len != self.float_workspace_len
            || params.int_workspace.len != self.int_workspace_len
        {
            return Err(FlashInferError::invalid_argument(
                "decode run workspace size mismatch with decode plan",
            ));
        }
        let (_, page_size_k, num_kv_heads_k, _) =
            decode_paged_layout(params.paged_k_cache, params.kv_layout);
        if page_size_k != self.page_size || num_kv_heads_k != self.num_kv_heads {
            return Err(FlashInferError::invalid_argument(format!(
                "decode run page/head mismatch: planned page_size={}, num_kv_heads={} but got page_size={}, num_kv_heads={}",
                self.page_size, self.num_kv_heads, page_size_k, num_kv_heads_k
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MhaBatchPagedDecodeParams {
    /// Query tensor, rank-3: `[batch_size, num_qo_heads, head_dim_qk]`.
    ///
    /// Cross-reference:
    /// `flashinfer/flashinfer/decode.py::BatchDecodeWithPagedKVCacheWrapper.run`.
    pub q: MhaTensor3DDesc,
    /// Paged key cache, rank-4:
    /// - `NHD`: `[num_pages, page_size, num_kv_heads, head_dim_qk]`
    /// - `HND`: `[num_pages, num_kv_heads, page_size, head_dim_qk]`.
    pub paged_k_cache: MhaTensor4DDesc,
    /// Paged value cache, rank-4:
    /// - `NHD`: `[num_pages, page_size, num_kv_heads, head_dim_vo]`
    /// - `HND`: `[num_pages, num_kv_heads, page_size, head_dim_vo]`.
    pub paged_v_cache: MhaTensor4DDesc,
    /// Paged-KV page indptr on device, rank-1 int32: `[batch_size + 1]`.
    pub paged_kv_indptr: MhaTensor1DI32Desc,
    /// Paged-KV page indices on device.
    pub paged_kv_indices: MhaTensor1DI32Desc,
    /// Last-page token counts on device, rank-1 int32: `[batch_size]`.
    pub paged_kv_last_page_len: MhaTensor1DI32Desc,
    /// Float workspace on device, rank-1 u8: `[workspace_bytes]`.
    pub float_workspace: MhaTensor1DU8Desc,
    /// Int workspace on device, rank-1 u8: `[workspace_bytes]`.
    pub int_workspace: MhaTensor1DU8Desc,
    /// Output tensor, rank-3: `[batch_size, num_qo_heads, head_dim_vo]`.
    pub out: MhaTensor3DDesc,
    /// Optional log-sum-exp output, rank-2 f32: `[batch_size, num_qo_heads]`.
    pub lse: Option<MhaTensor2DF32Desc>,
    /// Optional ALiBi slopes, rank-1 f32: `[num_qo_heads]`.
    pub alibi_slopes: Option<MhaTensor1DF32Desc>,
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
    /// Whether to enable Programmatic Dependent Launch.
    pub enable_pdl: bool,
    /// Whether this run expects a CUDA-graph-compatible plan.
    pub enable_cuda_graph: bool,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl MhaBatchPagedDecodeParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q: MhaTensor3DDesc,
        paged_k_cache: MhaTensor4DDesc,
        paged_v_cache: MhaTensor4DDesc,
        paged_kv_indptr: MhaTensor1DI32Desc,
        paged_kv_indices: MhaTensor1DI32Desc,
        paged_kv_last_page_len: MhaTensor1DI32Desc,
        float_workspace: MhaTensor1DU8Desc,
        int_workspace: MhaTensor1DU8Desc,
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
            paged_k_cache,
            paged_v_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            float_workspace,
            int_workspace,
            out,
            lse: None,
            alibi_slopes: None,
            kv_layout: MhaQkvLayout::Nhd,
            pos_encoding_mode: MhaPosEncodingMode::None,
            window_left: -1,
            logits_soft_cap: 0.0,
            sm_scale,
            rope_scale: 1.0,
            rope_theta: 1e4,
            enable_pdl: false,
            enable_cuda_graph: false,
            stream,
        }
    }

    pub fn with_lse(mut self, lse: MhaTensor2DF32Desc) -> Self {
        self.lse = Some(lse);
        self
    }

    pub fn with_alibi_slopes(mut self, alibi_slopes: MhaTensor1DF32Desc) -> Self {
        self.alibi_slopes = Some(alibi_slopes);
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

    pub fn with_enable_pdl(mut self, enable_pdl: bool) -> Self {
        self.enable_pdl = enable_pdl;
        self
    }

    pub fn with_enable_cuda_graph(mut self, enable_cuda_graph: bool) -> Self {
        self.enable_cuda_graph = enable_cuda_graph;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.q.ptr, "q")?;
        check_non_null(self.paged_k_cache.ptr, "paged_k_cache")?;
        check_non_null(self.paged_v_cache.ptr, "paged_v_cache")?;
        check_non_null(self.paged_kv_indptr.ptr, "paged_kv_indptr")?;
        check_non_null(self.paged_kv_indices.ptr, "paged_kv_indices")?;
        check_non_null(self.paged_kv_last_page_len.ptr, "paged_kv_last_page_len")?;
        check_non_null(self.float_workspace.ptr, "float_workspace")?;
        check_non_null(self.int_workspace.ptr, "int_workspace")?;
        check_non_null(self.out.ptr, "out")?;

        check_positive("q.dim0", self.q.dim0)?;
        check_positive("q.dim1", self.q.dim1)?;
        check_positive("q.dim2", self.q.dim2)?;
        check_positive("paged_k_cache.dim0", self.paged_k_cache.dim0)?;
        check_positive("paged_k_cache.dim1", self.paged_k_cache.dim1)?;
        check_positive("paged_k_cache.dim2", self.paged_k_cache.dim2)?;
        check_positive("paged_k_cache.dim3", self.paged_k_cache.dim3)?;
        check_positive("paged_v_cache.dim0", self.paged_v_cache.dim0)?;
        check_positive("paged_v_cache.dim1", self.paged_v_cache.dim1)?;
        check_positive("paged_v_cache.dim2", self.paged_v_cache.dim2)?;
        check_positive("paged_v_cache.dim3", self.paged_v_cache.dim3)?;
        check_positive("paged_kv_indices.len", self.paged_kv_indices.len)?;
        check_positive("out.dim0", self.out.dim0)?;
        check_positive("out.dim1", self.out.dim1)?;
        check_positive("out.dim2", self.out.dim2)?;

        if self.float_workspace.len <= 0 || self.int_workspace.len <= 0 {
            return Err(FlashInferError::invalid_argument(
                "workspace lengths must be positive",
            ));
        }

        check_contiguous_1d("paged_kv_indptr", self.paged_kv_indptr.stride)?;
        check_contiguous_1d("paged_kv_indices", self.paged_kv_indices.stride)?;
        check_contiguous_1d("paged_kv_last_page_len", self.paged_kv_last_page_len.stride)?;
        check_contiguous_1d("float_workspace", self.float_workspace.stride)?;
        check_contiguous_1d("int_workspace", self.int_workspace.stride)?;

        if self.q.stride2 != 1 || self.out.stride2 != 1 {
            return Err(FlashInferError::invalid_argument(
                "q/out last-dimension stride must be 1",
            ));
        }
        if self.paged_k_cache.stride3 != 1 || self.paged_v_cache.stride3 != 1 {
            return Err(FlashInferError::invalid_argument(
                "paged_k_cache/paged_v_cache last-dimension stride must be 1",
            ));
        }

        if self.q.dtype != self.paged_k_cache.dtype
            || self.q.dtype != self.paged_v_cache.dtype
            || self.q.dtype != self.out.dtype
        {
            return Err(FlashInferError::invalid_argument(
                "q/paged_k_cache/paged_v_cache/out dtype mismatch",
            ));
        }

        if self.paged_k_cache.stride0 != self.paged_v_cache.stride0
            || self.paged_k_cache.stride1 != self.paged_v_cache.stride1
            || self.paged_k_cache.stride2 != self.paged_v_cache.stride2
            || self.paged_k_cache.stride3 != self.paged_v_cache.stride3
        {
            return Err(FlashInferError::invalid_argument(
                "paged_k_cache and paged_v_cache must have identical strides",
            ));
        }

        let batch_size = self.q.dim0;
        if self.out.dim0 != batch_size || self.out.dim1 != self.q.dim1 {
            return Err(FlashInferError::invalid_argument(
                "out shape must be [batch_size, num_qo_heads, head_dim_vo]",
            ));
        }
        if self.q.dim2 != self.out.dim2 {
            return Err(FlashInferError::invalid_argument(
                "decode kernels require head_dim_qk == head_dim_vo",
            ));
        }
        if self.paged_kv_indptr.len != batch_size + 1 {
            return Err(FlashInferError::invalid_argument(
                "paged_kv_indptr length must equal batch_size + 1",
            ));
        }
        if self.paged_kv_last_page_len.len != batch_size {
            return Err(FlashInferError::invalid_argument(
                "paged_kv_last_page_len length must equal batch_size",
            ));
        }

        let (num_pages_k, page_size_k, num_kv_heads_k, head_dim_qk) =
            decode_paged_layout(self.paged_k_cache, self.kv_layout);
        let (num_pages_v, page_size_v, num_kv_heads_v, head_dim_vo) =
            decode_paged_layout(self.paged_v_cache, self.kv_layout);

        if num_pages_k != num_pages_v || page_size_k != page_size_v {
            return Err(FlashInferError::invalid_argument(
                "paged_k_cache and paged_v_cache must have matching num_pages and page_size",
            ));
        }
        if num_kv_heads_k != num_kv_heads_v {
            return Err(FlashInferError::invalid_argument(
                "paged_k_cache and paged_v_cache num_kv_heads mismatch",
            ));
        }
        if self.q.dim2 != head_dim_qk {
            return Err(FlashInferError::invalid_argument(
                "q head_dim_qk must match paged_k_cache",
            ));
        }
        if self.out.dim2 != head_dim_vo {
            return Err(FlashInferError::invalid_argument(
                "out head_dim_vo must match paged_v_cache",
            ));
        }
        if self.q.dim1 % num_kv_heads_k != 0 {
            return Err(FlashInferError::invalid_argument(
                "num_qo_heads must be divisible by num_kv_heads",
            ));
        }

        let device_id = self.q.device_id;
        if self.paged_k_cache.device_id != device_id
            || self.paged_v_cache.device_id != device_id
            || self.paged_kv_indptr.device_id != device_id
            || self.paged_kv_indices.device_id != device_id
            || self.paged_kv_last_page_len.device_id != device_id
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
            if lse.rows != batch_size || lse.cols != self.q.dim1 {
                return Err(FlashInferError::invalid_argument(
                    "lse shape must be [batch_size, num_qo_heads]",
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

        if let Some(alibi) = self.alibi_slopes {
            check_non_null(alibi.ptr, "alibi_slopes")?;
            check_positive("alibi_slopes.len", alibi.len)?;
            if alibi.len != self.q.dim1 {
                return Err(FlashInferError::invalid_argument(
                    "alibi_slopes length must equal num_qo_heads",
                ));
            }
            if alibi.stride != 1 {
                return Err(FlashInferError::invalid_argument(
                    "alibi_slopes stride must be 1",
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

        Ok(())
    }

    fn kernel_uri(&self) -> String {
        format!(
            "batch_decode_with_kv_cache_dtype_q_{}_dtype_kv_{}_dtype_o_{}_dtype_idx_i32_head_dim_qk_{}_head_dim_vo_{}_posenc_{}_use_swa_{}_use_logits_cap_{}",
            dtype_filename(self.q.dtype),
            dtype_filename(self.paged_k_cache.dtype),
            dtype_filename(self.out.dtype),
            self.q.dim2,
            self.out.dim2,
            pos_encoding_mode_code(self.pos_encoding_mode),
            bool_name(self.window_left >= 0),
            bool_name(self.logits_soft_cap > 0.0),
        )
    }
}

pub fn mha_batch_decode_paged_plan(
    params: &MhaBatchPagedDecodePlanParams,
) -> Result<MhaBatchPagedDecodePlan, FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { mha_batch_decode_paged_plan_with_runtime(runtime, params) }
}

unsafe fn mha_batch_decode_paged_plan_with_runtime(
    runtime: &'static FlashInferRuntime,
    params: &MhaBatchPagedDecodePlanParams,
) -> Result<MhaBatchPagedDecodePlan, FlashInferError> {
    let kernel_uri = params.kernel_uri();

    let mut paged_kv_host_shape = [params.paged_kv_indptr_host.len];
    let mut paged_kv_host_strides = [params.paged_kv_indptr_host.stride];
    let paged_kv_indptr_host_tensor = DLTensor {
        data: params.paged_kv_indptr_host.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: paged_kv_host_shape.as_mut_ptr(),
        strides: paged_kv_host_strides.as_mut_ptr(),
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

    let empty_q_data: [u8; 0] = [];
    let empty_kv_data: [u8; 0] = [];
    let mut empty_q_shape = [0_i64];
    let mut empty_q_strides = [1_i64];
    let empty_q_tensor = DLTensor {
        data: empty_q_data.as_ptr().cast_mut().cast(),
        device: DLDevice {
            device_type: KDL_CPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: dl_dtype_from_norm_dtype(params.dtype),
        shape: empty_q_shape.as_mut_ptr(),
        strides: empty_q_strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let mut empty_kv_shape = [0_i64];
    let mut empty_kv_strides = [1_i64];
    let empty_kv_tensor = DLTensor {
        data: empty_kv_data.as_ptr().cast_mut().cast(),
        device: DLDevice {
            device_type: KDL_CPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: dl_dtype_from_norm_dtype(params.dtype),
        shape: empty_kv_shape.as_mut_ptr(),
        strides: empty_kv_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut plan_result_view = any_none();
    let plan_args: [TVMFFIAny; 15] = [
        any_dltensor_ptr(&float_workspace_tensor),
        any_dltensor_ptr(&int_workspace_tensor),
        any_dltensor_ptr(&page_locked_int_workspace_tensor),
        any_dltensor_ptr(&paged_kv_indptr_host_tensor),
        any_i64(params.batch_size),
        any_i64(params.num_qo_heads),
        any_i64(params.num_kv_heads),
        any_i64(params.page_size),
        any_bool(params.enable_cuda_graph),
        any_i64(params.window_left),
        any_f64(params.logits_soft_cap),
        any_i64(params.head_dim_qk),
        any_i64(params.head_dim_vo),
        any_dltensor_ptr(&empty_q_tensor),
        any_dltensor_ptr(&empty_kv_tensor),
    ];

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.device_id, params.stream)? };
    let mut restore_guard = StreamRestoreGuard::new(runtime, params.device_id, previous_stream);

    let call_result = (|| -> Result<MhaBatchPagedDecodePlan, FlashInferError> {
        // SAFETY: argument packing follows TVMFFIAny ABI.
        unsafe {
            runtime.call_batch_decode_plan(
                &kernel_uri,
                plan_args.as_ptr(),
                plan_args.len() as i32,
                &mut plan_result_view as *mut _,
            )?;
        }

        // SAFETY: converts returned AnyView into owned Any so lifetime can cross calls safely.
        let plan_result = unsafe { runtime.any_view_to_owned(&plan_result_view)? };
        Ok(MhaBatchPagedDecodePlan {
            runtime,
            plan_result,
            kernel_uri,
            device_id: params.device_id,
            batch_size: params.batch_size,
            num_qo_heads: params.num_qo_heads,
            num_kv_heads: params.num_kv_heads,
            page_size: params.page_size,
            head_dim_qk: params.head_dim_qk,
            head_dim_vo: params.head_dim_vo,
            dtype: params.dtype,
            kv_layout: params.kv_layout,
            pos_encoding_mode: params.pos_encoding_mode,
            window_left: params.window_left,
            logits_soft_cap: params.logits_soft_cap,
            enable_cuda_graph: params.enable_cuda_graph,
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

pub fn mha_batch_decode_paged_run(
    plan: &MhaBatchPagedDecodePlan,
    params: &MhaBatchPagedDecodeParams,
) -> Result<(), FlashInferError> {
    params.validate()?;
    plan.validate_run_compatibility(params)?;
    // SAFETY: FFI preconditions are validated by `params.validate` and `plan` compatibility checks.
    unsafe { mha_batch_decode_paged_run_with_runtime(plan.runtime, plan, params) }
}

unsafe fn mha_batch_decode_paged_run_with_runtime(
    runtime: &FlashInferRuntime,
    plan: &MhaBatchPagedDecodePlan,
    params: &MhaBatchPagedDecodeParams,
) -> Result<(), FlashInferError> {
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

    let mut paged_k_shape = [
        params.paged_k_cache.dim0,
        params.paged_k_cache.dim1,
        params.paged_k_cache.dim2,
        params.paged_k_cache.dim3,
    ];
    let mut paged_k_strides = [
        params.paged_k_cache.stride0,
        params.paged_k_cache.stride1,
        params.paged_k_cache.stride2,
        params.paged_k_cache.stride3,
    ];
    let paged_k_tensor = DLTensor {
        data: params.paged_k_cache.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.paged_k_cache.device_id,
        },
        ndim: 4,
        dtype: dl_dtype_from_norm_dtype(params.paged_k_cache.dtype),
        shape: paged_k_shape.as_mut_ptr(),
        strides: paged_k_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut paged_v_shape = [
        params.paged_v_cache.dim0,
        params.paged_v_cache.dim1,
        params.paged_v_cache.dim2,
        params.paged_v_cache.dim3,
    ];
    let mut paged_v_strides = [
        params.paged_v_cache.stride0,
        params.paged_v_cache.stride1,
        params.paged_v_cache.stride2,
        params.paged_v_cache.stride3,
    ];
    let paged_v_tensor = DLTensor {
        data: params.paged_v_cache.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.paged_v_cache.device_id,
        },
        ndim: 4,
        dtype: dl_dtype_from_norm_dtype(params.paged_v_cache.dtype),
        shape: paged_v_shape.as_mut_ptr(),
        strides: paged_v_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut paged_kv_indptr_shape = [params.paged_kv_indptr.len];
    let mut paged_kv_indptr_strides = [params.paged_kv_indptr.stride];
    let paged_kv_indptr_tensor = DLTensor {
        data: params.paged_kv_indptr.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.paged_kv_indptr.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: paged_kv_indptr_shape.as_mut_ptr(),
        strides: paged_kv_indptr_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut paged_kv_indices_shape = [params.paged_kv_indices.len];
    let mut paged_kv_indices_strides = [params.paged_kv_indices.stride];
    let paged_kv_indices_tensor = DLTensor {
        data: params.paged_kv_indices.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.paged_kv_indices.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: paged_kv_indices_shape.as_mut_ptr(),
        strides: paged_kv_indices_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut paged_kv_last_page_len_shape = [params.paged_kv_last_page_len.len];
    let mut paged_kv_last_page_len_strides = [params.paged_kv_last_page_len.stride];
    let paged_kv_last_page_len_tensor = DLTensor {
        data: params.paged_kv_last_page_len.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.paged_kv_last_page_len.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: paged_kv_last_page_len_shape.as_mut_ptr(),
        strides: paged_kv_last_page_len_strides.as_mut_ptr(),
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
    let alibi_any = alibi_tensor
        .as_ref()
        .map_or_else(any_none, |t| any_dltensor_ptr(t as *const DLTensor));

    let mut run_result = any_none();
    let run_args: [TVMFFIAny; 19] = [
        any_dltensor_ptr(&float_workspace_tensor),
        any_dltensor_ptr(&int_workspace_tensor),
        plan.plan_result,
        any_dltensor_ptr(&q_tensor),
        any_dltensor_ptr(&paged_k_tensor),
        any_dltensor_ptr(&paged_v_tensor),
        any_dltensor_ptr(&paged_kv_indptr_tensor),
        any_dltensor_ptr(&paged_kv_indices_tensor),
        any_dltensor_ptr(&paged_kv_last_page_len_tensor),
        any_dltensor_ptr(&out_tensor),
        lse_any,
        any_i64(kv_layout_code(params.kv_layout)),
        any_i64(params.window_left),
        any_bool(params.enable_pdl),
        alibi_any,
        any_f64(params.logits_soft_cap),
        any_f64(params.sm_scale),
        any_f64(1.0 / params.rope_scale),
        any_f64(1.0 / params.rope_theta),
    ];

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.q.device_id, params.stream)? };
    let mut restore_guard = StreamRestoreGuard::new(runtime, params.q.device_id, previous_stream);

    // SAFETY: argument packing follows TVMFFIAny ABI.
    let call_result = unsafe {
        runtime.call_batch_decode_run(
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

#[cfg(feature = "cudarc")]
#[derive(Debug, Clone, Copy)]
pub struct MhaSingleDecodeCudarcOptions {
    pub kv_layout: MhaQkvLayout,
    pub pos_encoding_mode: MhaPosEncodingMode,
    pub window_left: i64,
    pub logits_soft_cap: f64,
    pub sm_scale: Option<f64>,
    pub rope_scale: f64,
    pub rope_theta: f64,
}

#[cfg(feature = "cudarc")]
impl Default for MhaSingleDecodeCudarcOptions {
    fn default() -> Self {
        Self {
            kv_layout: MhaQkvLayout::Nhd,
            pos_encoding_mode: MhaPosEncodingMode::None,
            window_left: -1,
            logits_soft_cap: 0.0,
            sm_scale: None,
            rope_scale: 1.0,
            rope_theta: 1e4,
        }
    }
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn mha_single_decode_cudarc<T, Q, K, V, W, O>(
    stream: &cudarc::driver::CudaStream,
    q: &Q,
    k: &K,
    v: &V,
    tmp: &mut W,
    out: &mut O,
    kv_len: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim_qk: usize,
    head_dim_vo: usize,
    dtype: DType,
    options: MhaSingleDecodeCudarcOptions,
) -> Result<(), FlashInferError>
where
    Q: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    K: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    V: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    if kv_len == 0 || num_qo_heads == 0 || num_kv_heads == 0 || head_dim_qk == 0 || head_dim_vo == 0
    {
        return Err(FlashInferError::invalid_argument(
            "kv_len/num_heads/head_dim must be positive",
        ));
    }
    if num_qo_heads % num_kv_heads != 0 {
        return Err(FlashInferError::invalid_argument(
            "num_qo_heads must be divisible by num_kv_heads",
        ));
    }
    if head_dim_qk != head_dim_vo {
        return Err(FlashInferError::invalid_argument(
            "decode kernels require head_dim_qk == head_dim_vo",
        ));
    }
    if tmp.len() == 0 {
        return Err(FlashInferError::invalid_argument(
            "tmp length must be positive",
        ));
    }

    let q_len_expected = num_qo_heads
        .checked_mul(head_dim_qk)
        .ok_or_else(|| FlashInferError::invalid_argument("q size overflow"))?;
    let k_len_expected = kv_len
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim_qk))
        .ok_or_else(|| FlashInferError::invalid_argument("k size overflow"))?;
    let v_len_expected = kv_len
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim_vo))
        .ok_or_else(|| FlashInferError::invalid_argument("v size overflow"))?;
    let out_len_expected = num_qo_heads
        .checked_mul(head_dim_vo)
        .ok_or_else(|| FlashInferError::invalid_argument("out size overflow"))?;

    if q.len() != q_len_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "q length ({}) must equal num_qo_heads * head_dim_qk ({q_len_expected})",
            q.len()
        )));
    }
    if k.len() != k_len_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "k length ({}) must equal kv_len * num_kv_heads * head_dim_qk ({k_len_expected})",
            k.len()
        )));
    }
    if v.len() != v_len_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "v length ({}) must equal kv_len * num_kv_heads * head_dim_vo ({v_len_expected})",
            v.len()
        )));
    }
    if out.len() != out_len_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal num_qo_heads * head_dim_vo ({out_len_expected})",
            out.len()
        )));
    }

    let tmp_len = tmp.len();
    let (q_ptr, _q_sync) = q.device_ptr(stream);
    let (k_ptr, _k_sync) = k.device_ptr(stream);
    let (v_ptr, _v_sync) = v.device_ptr(stream);
    let (tmp_ptr, _tmp_sync) = tmp.device_ptr_mut(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);

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

    let params = MhaSingleDecodeParams::new(
        MhaTensor2DDesc {
            ptr: q_ptr as usize as *const c_void,
            rows: num_qo_heads_i64,
            cols: head_dim_qk_i64,
            stride_row: head_dim_qk_i64,
            stride_col: 1,
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
        MhaTensor2DDesc {
            ptr: out_ptr as usize as *const c_void,
            rows: num_qo_heads_i64,
            cols: head_dim_vo_i64,
            stride_row: head_dim_vo_i64,
            stride_col: 1,
            dtype,
            device_id,
        },
        stream.cu_stream().cast(),
    )
    .with_kv_layout(options.kv_layout)
    .with_pos_encoding_mode(options.pos_encoding_mode)
    .with_window_left(options.window_left)
    .with_logits_soft_cap(options.logits_soft_cap)
    .with_sm_scale(sm_scale)
    .with_rope_scale(options.rope_scale)
    .with_rope_theta(options.rope_theta);

    mha_single_decode(&params)
}

#[cfg(feature = "cudarc")]
#[derive(Debug, Clone, Copy)]
pub struct MhaBatchDecodeCudarcOptions {
    pub kv_layout: MhaQkvLayout,
    pub pos_encoding_mode: MhaPosEncodingMode,
    pub window_left: i64,
    pub logits_soft_cap: f64,
    pub sm_scale: Option<f64>,
    pub rope_scale: f64,
    pub rope_theta: f64,
    pub enable_pdl: bool,
    pub enable_cuda_graph: bool,
}

#[cfg(feature = "cudarc")]
impl Default for MhaBatchDecodeCudarcOptions {
    fn default() -> Self {
        Self {
            kv_layout: MhaQkvLayout::Nhd,
            pos_encoding_mode: MhaPosEncodingMode::None,
            window_left: -1,
            logits_soft_cap: 0.0,
            sm_scale: None,
            rope_scale: 1.0,
            rope_theta: 1e4,
            enable_pdl: false,
            enable_cuda_graph: false,
        }
    }
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn mha_batch_decode_paged_cudarc_plan<FW, IW>(
    stream: &cudarc::driver::CudaStream,
    paged_kv_indptr_host: &[i32],
    float_workspace: &mut FW,
    int_workspace: &mut IW,
    page_locked_int_workspace: &mut [u8],
    batch_size: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim_qk: usize,
    head_dim_vo: usize,
    page_size: usize,
    dtype: DType,
    options: MhaBatchDecodeCudarcOptions,
) -> Result<MhaBatchPagedDecodePlan, FlashInferError>
where
    FW: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
    IW: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    if batch_size == 0 {
        return Err(FlashInferError::invalid_argument(
            "batch_size must be positive",
        ));
    }
    if num_qo_heads == 0
        || num_kv_heads == 0
        || head_dim_qk == 0
        || head_dim_vo == 0
        || page_size == 0
    {
        return Err(FlashInferError::invalid_argument(
            "num_heads/head_dim/page_size must be positive",
        ));
    }
    if head_dim_qk != head_dim_vo {
        return Err(FlashInferError::invalid_argument(
            "decode kernels require head_dim_qk == head_dim_vo",
        ));
    }
    if num_qo_heads % num_kv_heads != 0 {
        return Err(FlashInferError::invalid_argument(
            "num_qo_heads must be divisible by num_kv_heads",
        ));
    }
    if paged_kv_indptr_host.len() != batch_size + 1 {
        return Err(FlashInferError::invalid_argument(
            "paged_kv_indptr_host length must equal batch_size + 1",
        ));
    }
    let host_batch = validate_indptr("paged_kv_indptr_host", paged_kv_indptr_host)?;
    if usize::try_from(host_batch)
        .map_err(|_| FlashInferError::invalid_argument("paged_kv_indptr_host batch overflow"))?
        != batch_size
    {
        return Err(FlashInferError::invalid_argument(
            "paged_kv_indptr_host batch must equal batch_size",
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

    let (float_workspace_ptr, _float_ws_sync) = float_workspace.device_ptr_mut(stream);
    let (int_workspace_ptr, _int_ws_sync) = int_workspace.device_ptr_mut(stream);

    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let batch_size_i64 = i64::try_from(batch_size)
        .map_err(|_| FlashInferError::invalid_argument("batch_size does not fit in i64"))?;
    let page_size_i64 = i64::try_from(page_size)
        .map_err(|_| FlashInferError::invalid_argument("page_size does not fit in i64"))?;
    let num_qo_heads_i64 = i64::try_from(num_qo_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_qo_heads does not fit in i64"))?;
    let num_kv_heads_i64 = i64::try_from(num_kv_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_kv_heads does not fit in i64"))?;
    let head_dim_qk_i64 = i64::try_from(head_dim_qk)
        .map_err(|_| FlashInferError::invalid_argument("head_dim_qk does not fit in i64"))?;
    let head_dim_vo_i64 = i64::try_from(head_dim_vo)
        .map_err(|_| FlashInferError::invalid_argument("head_dim_vo does not fit in i64"))?;
    let paged_kv_indptr_host_len_i64 = i64::try_from(paged_kv_indptr_host.len()).map_err(|_| {
        FlashInferError::invalid_argument("paged_kv_indptr_host length does not fit in i64")
    })?;
    let float_ws_len_i64 = i64::try_from(float_workspace_len).map_err(|_| {
        FlashInferError::invalid_argument("float_workspace length does not fit in i64")
    })?;
    let int_ws_len_i64 = i64::try_from(int_workspace_len).map_err(|_| {
        FlashInferError::invalid_argument("int_workspace length does not fit in i64")
    })?;
    let page_locked_ws_len_i64 = i64::try_from(page_locked_int_workspace.len()).map_err(|_| {
        FlashInferError::invalid_argument("page_locked_int_workspace length does not fit in i64")
    })?;

    let params = MhaBatchPagedDecodePlanParams::new(
        MhaHostTensor1DI32Desc {
            ptr: paged_kv_indptr_host.as_ptr().cast(),
            len: paged_kv_indptr_host_len_i64,
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
        batch_size_i64,
        num_qo_heads_i64,
        num_kv_heads_i64,
        page_size_i64,
        head_dim_qk_i64,
        head_dim_vo_i64,
        dtype,
        device_id,
        stream.cu_stream().cast(),
    )
    .with_kv_layout(options.kv_layout)
    .with_pos_encoding_mode(options.pos_encoding_mode)
    .with_window_left(options.window_left)
    .with_logits_soft_cap(options.logits_soft_cap)
    .with_enable_cuda_graph(options.enable_cuda_graph);

    mha_batch_decode_paged_plan(&params)
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn mha_batch_decode_paged_cudarc_run<T, Q, K, V, O, PI, I, L, FW, IW>(
    stream: &cudarc::driver::CudaStream,
    plan: &MhaBatchPagedDecodePlan,
    q: &Q,
    paged_k_cache: &K,
    paged_v_cache: &V,
    paged_kv_indptr: &PI,
    paged_kv_indices: &I,
    paged_kv_last_page_len: &L,
    float_workspace: &mut FW,
    int_workspace: &mut IW,
    out: &mut O,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim_qk: usize,
    head_dim_vo: usize,
    page_size: usize,
    dtype: DType,
    options: MhaBatchDecodeCudarcOptions,
) -> Result<(), FlashInferError>
where
    Q: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    K: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    V: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    PI: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    I: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    L: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    FW: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
    IW: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    if num_qo_heads == 0
        || num_kv_heads == 0
        || head_dim_qk == 0
        || head_dim_vo == 0
        || page_size == 0
    {
        return Err(FlashInferError::invalid_argument(
            "num_heads/head_dim/page_size must be positive",
        ));
    }
    if head_dim_qk != head_dim_vo {
        return Err(FlashInferError::invalid_argument(
            "decode kernels require head_dim_qk == head_dim_vo",
        ));
    }
    if num_qo_heads % num_kv_heads != 0 {
        return Err(FlashInferError::invalid_argument(
            "num_qo_heads must be divisible by num_kv_heads",
        ));
    }

    let q_row_size = num_qo_heads
        .checked_mul(head_dim_qk)
        .ok_or_else(|| FlashInferError::invalid_argument("q row size overflow"))?;
    if q_row_size == 0 || q.len() % q_row_size != 0 {
        return Err(FlashInferError::invalid_argument(
            "q length must be divisible by num_qo_heads * head_dim_qk",
        ));
    }
    let batch_size = q.len() / q_row_size;
    if batch_size == 0 {
        return Err(FlashInferError::invalid_argument(
            "batch_size inferred from q must be positive",
        ));
    }
    if paged_kv_last_page_len.len() != batch_size {
        return Err(FlashInferError::invalid_argument(
            "paged_kv_last_page_len length must equal batch_size",
        ));
    }
    if paged_kv_indptr.len() != batch_size + 1 {
        return Err(FlashInferError::invalid_argument(
            "paged_kv_indptr length must equal batch_size + 1",
        ));
    }

    let out_len_expected = batch_size
        .checked_mul(num_qo_heads)
        .and_then(|v| v.checked_mul(head_dim_vo))
        .ok_or_else(|| FlashInferError::invalid_argument("out size overflow"))?;
    if out.len() != out_len_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "out length ({}) must equal batch_size * num_qo_heads * head_dim_vo ({out_len_expected})",
            out.len()
        )));
    }

    let k_page_unit = page_size
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim_qk))
        .ok_or_else(|| FlashInferError::invalid_argument("paged_k_cache size overflow"))?;
    let v_page_unit = page_size
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim_vo))
        .ok_or_else(|| FlashInferError::invalid_argument("paged_v_cache size overflow"))?;
    if paged_k_cache.len() % k_page_unit != 0 {
        return Err(FlashInferError::invalid_argument(
            "paged_k_cache length must be divisible by page_size * num_kv_heads * head_dim_qk",
        ));
    }
    if paged_v_cache.len() % v_page_unit != 0 {
        return Err(FlashInferError::invalid_argument(
            "paged_v_cache length must be divisible by page_size * num_kv_heads * head_dim_vo",
        ));
    }
    let num_pages_k = paged_k_cache.len() / k_page_unit;
    let num_pages_v = paged_v_cache.len() / v_page_unit;
    if num_pages_k != num_pages_v {
        return Err(FlashInferError::invalid_argument(
            "paged_k_cache and paged_v_cache must have same number of pages",
        ));
    }

    let float_workspace_len = float_workspace.len();
    let int_workspace_len = int_workspace.len();
    if float_workspace_len == 0 || int_workspace_len == 0 {
        return Err(FlashInferError::invalid_argument(
            "float/int workspace lengths must be positive",
        ));
    }

    let (q_ptr, _q_sync) = q.device_ptr(stream);
    let (paged_k_ptr, _k_sync) = paged_k_cache.device_ptr(stream);
    let (paged_v_ptr, _v_sync) = paged_v_cache.device_ptr(stream);
    let (paged_kv_indptr_ptr, _pindptr_sync) = paged_kv_indptr.device_ptr(stream);
    let (paged_kv_indices_ptr, _pidx_sync) = paged_kv_indices.device_ptr(stream);
    let (paged_kv_last_page_len_ptr, _plast_sync) = paged_kv_last_page_len.device_ptr(stream);
    let (float_workspace_ptr, _float_ws_sync) = float_workspace.device_ptr_mut(stream);
    let (int_workspace_ptr, _int_ws_sync) = int_workspace.device_ptr_mut(stream);
    let (out_ptr, _out_sync) = out.device_ptr_mut(stream);

    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let batch_size_i64 = i64::try_from(batch_size)
        .map_err(|_| FlashInferError::invalid_argument("batch_size does not fit in i64"))?;
    let num_pages_i64 = i64::try_from(num_pages_k)
        .map_err(|_| FlashInferError::invalid_argument("num_pages does not fit in i64"))?;
    let page_size_i64 = i64::try_from(page_size)
        .map_err(|_| FlashInferError::invalid_argument("page_size does not fit in i64"))?;
    let num_qo_heads_i64 = i64::try_from(num_qo_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_qo_heads does not fit in i64"))?;
    let num_kv_heads_i64 = i64::try_from(num_kv_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_kv_heads does not fit in i64"))?;
    let head_dim_qk_i64 = i64::try_from(head_dim_qk)
        .map_err(|_| FlashInferError::invalid_argument("head_dim_qk does not fit in i64"))?;
    let head_dim_vo_i64 = i64::try_from(head_dim_vo)
        .map_err(|_| FlashInferError::invalid_argument("head_dim_vo does not fit in i64"))?;
    let paged_kv_indptr_len_i64 = i64::try_from(paged_kv_indptr.len())
        .map_err(|_| FlashInferError::invalid_argument("paged_kv_indptr length overflow"))?;
    let page_entries_i64 = i64::try_from(paged_kv_indices.len())
        .map_err(|_| FlashInferError::invalid_argument("paged_kv_indices length overflow"))?;
    let float_ws_len_i64 = i64::try_from(float_workspace_len).map_err(|_| {
        FlashInferError::invalid_argument("float_workspace length does not fit in i64")
    })?;
    let int_ws_len_i64 = i64::try_from(int_workspace_len).map_err(|_| {
        FlashInferError::invalid_argument("int_workspace length does not fit in i64")
    })?;

    let q_stride1 = head_dim_qk_i64;
    let q_stride0 = num_qo_heads_i64
        .checked_mul(head_dim_qk_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("q stride overflow"))?;
    let out_stride1 = head_dim_vo_i64;
    let out_stride0 = num_qo_heads_i64
        .checked_mul(head_dim_vo_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("out stride overflow"))?;

    let (k_dim1, k_dim2, k_stride1, k_stride2) = match options.kv_layout {
        MhaQkvLayout::Nhd => {
            let s2 = head_dim_qk_i64;
            let s1 = num_kv_heads_i64
                .checked_mul(head_dim_qk_i64)
                .ok_or_else(|| {
                    FlashInferError::invalid_argument("paged_k_cache stride overflow")
                })?;
            (page_size_i64, num_kv_heads_i64, s1, s2)
        }
        MhaQkvLayout::Hnd => {
            let s2 = head_dim_qk_i64;
            let s1 = page_size_i64.checked_mul(head_dim_qk_i64).ok_or_else(|| {
                FlashInferError::invalid_argument("paged_k_cache stride overflow")
            })?;
            (num_kv_heads_i64, page_size_i64, s1, s2)
        }
    };
    let (v_dim1, v_dim2, v_stride1, v_stride2) = match options.kv_layout {
        MhaQkvLayout::Nhd => {
            let s2 = head_dim_vo_i64;
            let s1 = num_kv_heads_i64
                .checked_mul(head_dim_vo_i64)
                .ok_or_else(|| {
                    FlashInferError::invalid_argument("paged_v_cache stride overflow")
                })?;
            (page_size_i64, num_kv_heads_i64, s1, s2)
        }
        MhaQkvLayout::Hnd => {
            let s2 = head_dim_vo_i64;
            let s1 = page_size_i64.checked_mul(head_dim_vo_i64).ok_or_else(|| {
                FlashInferError::invalid_argument("paged_v_cache stride overflow")
            })?;
            (num_kv_heads_i64, page_size_i64, s1, s2)
        }
    };

    let k_stride0 = match options.kv_layout {
        MhaQkvLayout::Nhd => page_size_i64
            .checked_mul(num_kv_heads_i64)
            .and_then(|v| v.checked_mul(head_dim_qk_i64))
            .ok_or_else(|| FlashInferError::invalid_argument("paged_k_cache stride overflow"))?,
        MhaQkvLayout::Hnd => num_kv_heads_i64
            .checked_mul(page_size_i64)
            .and_then(|v| v.checked_mul(head_dim_qk_i64))
            .ok_or_else(|| FlashInferError::invalid_argument("paged_k_cache stride overflow"))?,
    };
    let v_stride0 = match options.kv_layout {
        MhaQkvLayout::Nhd => page_size_i64
            .checked_mul(num_kv_heads_i64)
            .and_then(|v| v.checked_mul(head_dim_vo_i64))
            .ok_or_else(|| FlashInferError::invalid_argument("paged_v_cache stride overflow"))?,
        MhaQkvLayout::Hnd => num_kv_heads_i64
            .checked_mul(page_size_i64)
            .and_then(|v| v.checked_mul(head_dim_vo_i64))
            .ok_or_else(|| FlashInferError::invalid_argument("paged_v_cache stride overflow"))?,
    };
    if k_stride0 != v_stride0 {
        return Err(FlashInferError::invalid_argument(
            "contiguous paged cache requires head_dim_qk == head_dim_vo so K/V strides match",
        ));
    }

    let sm_scale = options
        .sm_scale
        .unwrap_or_else(|| 1.0 / (head_dim_qk as f64).sqrt());

    let params = MhaBatchPagedDecodeParams::new(
        MhaTensor3DDesc {
            ptr: q_ptr as usize as *const c_void,
            dim0: batch_size_i64,
            dim1: num_qo_heads_i64,
            dim2: head_dim_qk_i64,
            stride0: q_stride0,
            stride1: q_stride1,
            stride2: 1,
            dtype,
            device_id,
        },
        MhaTensor4DDesc {
            ptr: paged_k_ptr as usize as *const c_void,
            dim0: num_pages_i64,
            dim1: k_dim1,
            dim2: k_dim2,
            dim3: head_dim_qk_i64,
            stride0: k_stride0,
            stride1: k_stride1,
            stride2: k_stride2,
            stride3: 1,
            dtype,
            device_id,
        },
        MhaTensor4DDesc {
            ptr: paged_v_ptr as usize as *const c_void,
            dim0: num_pages_i64,
            dim1: v_dim1,
            dim2: v_dim2,
            dim3: head_dim_vo_i64,
            stride0: v_stride0,
            stride1: v_stride1,
            stride2: v_stride2,
            stride3: 1,
            dtype,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: paged_kv_indptr_ptr as usize as *const c_void,
            len: paged_kv_indptr_len_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: paged_kv_indices_ptr as usize as *const c_void,
            len: page_entries_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: paged_kv_last_page_len_ptr as usize as *const c_void,
            len: batch_size_i64,
            stride: 1,
            device_id,
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
        MhaTensor3DDesc {
            ptr: out_ptr as usize as *const c_void,
            dim0: batch_size_i64,
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
    .with_kv_layout(options.kv_layout)
    .with_pos_encoding_mode(options.pos_encoding_mode)
    .with_window_left(options.window_left)
    .with_logits_soft_cap(options.logits_soft_cap)
    .with_sm_scale(sm_scale)
    .with_rope_scale(options.rope_scale)
    .with_rope_theta(options.rope_theta)
    .with_enable_pdl(options.enable_pdl)
    .with_enable_cuda_graph(options.enable_cuda_graph);

    mha_batch_decode_paged_run(plan, &params)
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

fn pos_encoding_mode_code(value: MhaPosEncodingMode) -> i64 {
    match value {
        MhaPosEncodingMode::None => 0,
        MhaPosEncodingMode::RoPELlama => 1,
        MhaPosEncodingMode::ALiBi => 2,
    }
}

fn kv_layout_code(value: MhaQkvLayout) -> i64 {
    match value {
        MhaQkvLayout::Nhd => 0,
        MhaQkvLayout::Hnd => 1,
    }
}

fn decode_paged_layout(desc: MhaTensor4DDesc, layout: MhaQkvLayout) -> (i64, i64, i64, i64) {
    match layout {
        MhaQkvLayout::Nhd => (desc.dim0, desc.dim1, desc.dim2, desc.dim3),
        MhaQkvLayout::Hnd => (desc.dim0, desc.dim2, desc.dim1, desc.dim3),
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

    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        // SAFETY: pointer validity is guaranteed by the caller contract.
        let value = unsafe { *desc.ptr.cast::<i32>().add(i * stride) };
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
    for window in indptr.windows(2) {
        if window[1] < window[0] {
            return Err(FlashInferError::invalid_argument(format!(
                "{name} must be non-decreasing",
            )));
        }
    }
    let batch = indptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| FlashInferError::invalid_argument(format!("{name} is empty")))?;
    u32::try_from(batch)
        .map_err(|_| FlashInferError::invalid_argument(format!("{name} batch size overflow")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn non_null() -> *const c_void {
        std::ptr::NonNull::<u8>::dangling().as_ptr().cast()
    }

    fn valid_single_params() -> MhaSingleDecodeParams {
        MhaSingleDecodeParams::new(
            MhaTensor2DDesc {
                ptr: non_null(),
                rows: 8,
                cols: 128,
                stride_row: 128,
                stride_col: 1,
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
                len: 4096,
                stride: 1,
                device_id: 0,
            },
            MhaTensor2DDesc {
                ptr: non_null(),
                rows: 8,
                cols: 128,
                stride_row: 128,
                stride_col: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            std::ptr::null_mut(),
        )
    }

    fn valid_batch_params() -> MhaBatchPagedDecodeParams {
        MhaBatchPagedDecodeParams::new(
            MhaTensor3DDesc {
                ptr: non_null(),
                dim0: 2,
                dim1: 8,
                dim2: 128,
                stride0: 1024,
                stride1: 128,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            MhaTensor4DDesc {
                ptr: non_null(),
                dim0: 3,
                dim1: 2,
                dim2: 4,
                dim3: 128,
                stride0: 1024,
                stride1: 512,
                stride2: 128,
                stride3: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            MhaTensor4DDesc {
                ptr: non_null(),
                dim0: 3,
                dim1: 2,
                dim2: 4,
                dim3: 128,
                stride0: 1024,
                stride1: 512,
                stride2: 128,
                stride3: 1,
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
            MhaTensor1DI32Desc {
                ptr: non_null(),
                len: 2,
                stride: 1,
                device_id: 0,
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
            MhaTensor3DDesc {
                ptr: non_null(),
                dim0: 2,
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

    fn valid_plan_params() -> MhaBatchPagedDecodePlanParams {
        static PAGED_KV_HOST: [i32; 3] = [0, 2, 3];

        MhaBatchPagedDecodePlanParams::new(
            MhaHostTensor1DI32Desc {
                ptr: PAGED_KV_HOST.as_ptr().cast(),
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
            2,
            8,
            4,
            2,
            128,
            128,
            DType::F16,
            0,
            std::ptr::null_mut(),
        )
    }

    #[test]
    fn single_validate_rejects_output_head_dim_mismatch() {
        let mut params = valid_single_params();
        params.out.cols = 64;
        assert!(params.validate().is_err());
    }

    #[test]
    fn single_validate_rejects_alibi_mode_without_slopes() {
        let params = valid_single_params().with_pos_encoding_mode(MhaPosEncodingMode::ALiBi);
        assert!(params.validate().is_err());
    }

    #[test]
    fn single_kernel_uri_matches_expected_pattern() {
        let uri = valid_single_params().kernel_uri();
        assert!(uri.contains("single_decode_with_kv_cache_dtype_q_f16"));
        assert!(uri.contains("head_dim_qk_128"));
        assert!(uri.contains("use_swa_False"));
        assert!(uri.contains("use_logits_cap_False"));
    }

    #[test]
    fn batch_validate_rejects_paged_indptr_length_mismatch() {
        let mut params = valid_batch_params();
        params.paged_kv_indptr.len = 4;
        assert!(params.validate().is_err());
    }

    #[test]
    fn batch_validate_rejects_out_shape_mismatch() {
        let mut params = valid_batch_params();
        params.out.dim0 = 3;
        assert!(params.validate().is_err());
    }

    #[test]
    fn batch_kernel_uri_matches_expected_pattern() {
        let uri = valid_batch_params().kernel_uri();
        assert!(uri.contains("batch_decode_with_kv_cache_dtype_q_f16"));
        assert!(uri.contains("dtype_idx_i32"));
        assert!(uri.contains("head_dim_qk_128"));
        assert!(uri.contains("use_swa_False"));
        assert!(uri.contains("use_logits_cap_False"));
    }

    #[test]
    fn plan_validate_rejects_host_indptr_length_mismatch() {
        let mut params = valid_plan_params();
        params.paged_kv_indptr_host.len = 4;
        assert!(params.validate().is_err());
    }
}
