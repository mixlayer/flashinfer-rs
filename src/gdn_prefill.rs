use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    any_dltensor_ptr, any_f64, any_none, DLDataType, DLDevice, DLTensor, TVMFFIAny, KDL_BFLOAT,
    KDL_CUDA, KDL_FLOAT, KDL_INT, KDL_UINT,
};
use crate::norm::DType;
use crate::runtime::FlashInferRuntime;

#[derive(Debug, Clone, Copy)]
pub struct Tensor1DI64Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Tensor1DU8Desc {
    pub ptr: *const c_void,
    pub len: i64,
    pub stride: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Tensor2DF32Desc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Tensor3DDesc {
    pub ptr: *const c_void,
    pub seq_len: i64,
    pub num_heads: i64,
    pub head_size: i64,
    pub stride_seq: i64,
    pub stride_head: i64,
    pub stride_dim: i64,
    pub dtype: DType,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Tensor4DF32Desc {
    pub ptr: *const c_void,
    pub dim0: i64,
    pub dim1: i64,
    pub dim2: i64,
    pub dim3: i64,
    pub stride0: i64,
    pub stride1: i64,
    pub stride2: i64,
    pub stride3: i64,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct GdnPrefillSm90Params {
    /// Output tensor, rank-3: `[packed_seq, num_sab_heads, head_size]`.
    ///
    /// `num_sab_heads = max(num_q_heads, num_v_heads)`.
    pub output: Tensor3DDesc,
    /// Output state tensor, rank-4:
    /// `[num_seqs, num_sab_heads, head_size, head_size]`.
    ///
    /// `num_seqs = cu_seqlens.len - 1`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/gdn_prefill_launcher.cu::gdn_prefill`.
    pub output_state: Tensor4DF32Desc,
    /// Query tensor, rank-3: `[packed_seq, num_q_heads, head_size]`.
    pub q: Tensor3DDesc,
    /// Key tensor, rank-3: `[packed_seq, num_k_heads, head_size]`.
    pub k: Tensor3DDesc,
    /// Value tensor, rank-3: `[packed_seq, num_v_heads, head_size]`.
    pub v: Tensor3DDesc,
    /// Prefix sums over sequence lengths, rank-1 int64: `[num_seqs + 1]`.
    pub cu_seqlens: Tensor1DI64Desc,
    /// Optional prior state, rank-4:
    /// `[num_seqs, num_sab_heads, head_size, head_size]`.
    ///
    /// Must match `output_state` shape when provided.
    pub input_state: Option<Tensor4DF32Desc>,
    /// Optional alpha factors, rank-2 f32: `[packed_seq, num_sab_heads]`.
    pub alpha: Option<Tensor2DF32Desc>,
    /// Optional beta factors, rank-2 f32: `[packed_seq, num_sab_heads]`.
    pub beta: Option<Tensor2DF32Desc>,
    /// Scale applied in kernel; `0.0` means use kernel default (`1/sqrt(head_size)`).
    pub scale: f64,
    /// Scratch/workspace buffer, rank-1 u8: `[workspace_bytes]`.
    pub workspace_buffer: Tensor1DU8Desc,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl GdnPrefillSm90Params {
    pub fn new(
        output: Tensor3DDesc,
        output_state: Tensor4DF32Desc,
        q: Tensor3DDesc,
        k: Tensor3DDesc,
        v: Tensor3DDesc,
        cu_seqlens: Tensor1DI64Desc,
        workspace_buffer: Tensor1DU8Desc,
        stream: *mut c_void,
    ) -> Self {
        Self {
            output,
            output_state,
            q,
            k,
            v,
            cu_seqlens,
            input_state: None,
            alpha: None,
            beta: None,
            scale: 0.0,
            workspace_buffer,
            stream,
        }
    }

    pub fn with_input_state(mut self, input_state: Tensor4DF32Desc) -> Self {
        self.input_state = Some(input_state);
        self
    }

    pub fn with_alpha(mut self, alpha: Tensor2DF32Desc) -> Self {
        self.alpha = Some(alpha);
        self
    }

    pub fn with_beta(mut self, beta: Tensor2DF32Desc) -> Self {
        self.beta = Some(beta);
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.output.ptr, "output")?;
        check_non_null(self.output_state.ptr, "output_state")?;
        check_non_null(self.q.ptr, "q")?;
        check_non_null(self.k.ptr, "k")?;
        check_non_null(self.v.ptr, "v")?;
        check_non_null(self.cu_seqlens.ptr, "cu_seqlens")?;
        check_non_null(self.workspace_buffer.ptr, "workspace_buffer")?;

        check_positive("q.seq_len", self.q.seq_len)?;
        check_positive("q.num_heads", self.q.num_heads)?;
        check_positive("q.head_size", self.q.head_size)?;
        check_positive("k.seq_len", self.k.seq_len)?;
        check_positive("k.num_heads", self.k.num_heads)?;
        check_positive("k.head_size", self.k.head_size)?;
        check_positive("v.seq_len", self.v.seq_len)?;
        check_positive("v.num_heads", self.v.num_heads)?;
        check_positive("v.head_size", self.v.head_size)?;
        check_positive("output.seq_len", self.output.seq_len)?;
        check_positive("output.num_heads", self.output.num_heads)?;
        check_positive("output.head_size", self.output.head_size)?;
        check_positive("output_state.dim0", self.output_state.dim0)?;
        check_positive("output_state.dim1", self.output_state.dim1)?;
        check_positive("output_state.dim2", self.output_state.dim2)?;
        check_positive("output_state.dim3", self.output_state.dim3)?;
        if self.cu_seqlens.len < 2 {
            return Err(FlashInferError::invalid_argument(
                "cu_seqlens length must be at least 2",
            ));
        }
        check_positive("workspace_buffer.len", self.workspace_buffer.len)?;

        check_contiguous_3d(
            "q",
            self.q.seq_len,
            self.q.num_heads,
            self.q.head_size,
            self.q.stride_seq,
            self.q.stride_head,
            self.q.stride_dim,
        )?;
        check_contiguous_3d(
            "k",
            self.k.seq_len,
            self.k.num_heads,
            self.k.head_size,
            self.k.stride_seq,
            self.k.stride_head,
            self.k.stride_dim,
        )?;
        check_contiguous_3d(
            "v",
            self.v.seq_len,
            self.v.num_heads,
            self.v.head_size,
            self.v.stride_seq,
            self.v.stride_head,
            self.v.stride_dim,
        )?;
        check_contiguous_3d(
            "output",
            self.output.seq_len,
            self.output.num_heads,
            self.output.head_size,
            self.output.stride_seq,
            self.output.stride_head,
            self.output.stride_dim,
        )?;
        check_contiguous_4d(
            "output_state",
            self.output_state.dim0,
            self.output_state.dim1,
            self.output_state.dim2,
            self.output_state.dim3,
            self.output_state.stride0,
            self.output_state.stride1,
            self.output_state.stride2,
            self.output_state.stride3,
        )?;
        check_contiguous_1d("cu_seqlens", self.cu_seqlens.stride)?;
        check_contiguous_1d("workspace_buffer", self.workspace_buffer.stride)?;

        if self.output.dtype != self.q.dtype
            || self.output.dtype != self.k.dtype
            || self.output.dtype != self.v.dtype
        {
            return Err(FlashInferError::invalid_argument(
                "output/q/k/v dtype mismatch",
            ));
        }

        let packed_seq = self.q.seq_len;
        let head_size = self.q.head_size;

        if self.k.seq_len != packed_seq
            || self.v.seq_len != packed_seq
            || self.output.seq_len != packed_seq
        {
            return Err(FlashInferError::invalid_argument(
                "q/k/v/output must share packed sequence length",
            ));
        }

        if self.k.head_size != head_size
            || self.v.head_size != head_size
            || self.output.head_size != head_size
        {
            return Err(FlashInferError::invalid_argument(
                "q/k/v/output head_size mismatch",
            ));
        }

        let num_q_heads = self.q.num_heads;
        let num_k_heads = self.k.num_heads;
        let num_v_heads = self.v.num_heads;
        let num_sab_heads = num_q_heads.max(num_v_heads);

        if num_q_heads >= num_v_heads {
            if num_q_heads % num_v_heads != 0 {
                return Err(FlashInferError::invalid_argument(
                    "GQA head ratio must divide exactly",
                ));
            }
            if num_k_heads != num_v_heads {
                return Err(FlashInferError::invalid_argument(
                    "for GQA, num_k_heads must equal num_v_heads",
                ));
            }
        } else {
            if num_v_heads % num_q_heads != 0 {
                return Err(FlashInferError::invalid_argument(
                    "GVA head ratio must divide exactly",
                ));
            }
            if num_q_heads != num_k_heads {
                return Err(FlashInferError::invalid_argument(
                    "for GVA, num_q_heads must equal num_k_heads",
                ));
            }
        }

        if self.output.num_heads != num_sab_heads {
            return Err(FlashInferError::invalid_argument(format!(
                "output.num_heads ({}) must equal max(num_q_heads, num_v_heads) ({num_sab_heads})",
                self.output.num_heads
            )));
        }

        let num_seqs = self.cu_seqlens.len - 1;
        if self.output_state.dim0 != num_seqs
            || self.output_state.dim1 != num_sab_heads
            || self.output_state.dim2 != head_size
            || self.output_state.dim3 != head_size
        {
            return Err(FlashInferError::invalid_argument(
                "output_state shape must be [num_seqs, num_sab_heads, head_size, head_size]",
            ));
        }

        if let Some(input_state) = self.input_state {
            check_non_null(input_state.ptr, "input_state")?;
            check_contiguous_4d(
                "input_state",
                input_state.dim0,
                input_state.dim1,
                input_state.dim2,
                input_state.dim3,
                input_state.stride0,
                input_state.stride1,
                input_state.stride2,
                input_state.stride3,
            )?;
            if input_state.dim0 != self.output_state.dim0
                || input_state.dim1 != self.output_state.dim1
                || input_state.dim2 != self.output_state.dim2
                || input_state.dim3 != self.output_state.dim3
            {
                return Err(FlashInferError::invalid_argument(
                    "input_state shape must match output_state",
                ));
            }
        }

        if let Some(alpha) = self.alpha {
            check_non_null(alpha.ptr, "alpha")?;
            check_contiguous_2d(
                "alpha",
                alpha.rows,
                alpha.cols,
                alpha.stride_row,
                alpha.stride_col,
            )?;
            if alpha.rows != packed_seq || alpha.cols != num_sab_heads {
                return Err(FlashInferError::invalid_argument(
                    "alpha shape must be [packed_seq, num_sab_heads]",
                ));
            }
        }

        if let Some(beta) = self.beta {
            check_non_null(beta.ptr, "beta")?;
            check_contiguous_2d(
                "beta",
                beta.rows,
                beta.cols,
                beta.stride_row,
                beta.stride_col,
            )?;
            if beta.rows != packed_seq || beta.cols != num_sab_heads {
                return Err(FlashInferError::invalid_argument(
                    "beta shape must be [packed_seq, num_sab_heads]",
                ));
            }
        }

        let device_id = self.q.device_id;
        if self.output.device_id != device_id
            || self.output_state.device_id != device_id
            || self.k.device_id != device_id
            || self.v.device_id != device_id
            || self.cu_seqlens.device_id != device_id
            || self.workspace_buffer.device_id != device_id
        {
            return Err(FlashInferError::invalid_argument(
                "all required tensors must be on the same CUDA device",
            ));
        }

        if let Some(input_state) = self.input_state {
            if input_state.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "input_state must be on the same CUDA device",
                ));
            }
        }

        if let Some(alpha) = self.alpha {
            if alpha.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "alpha must be on the same CUDA device",
                ));
            }
        }

        if let Some(beta) = self.beta {
            if beta.device_id != device_id {
                return Err(FlashInferError::invalid_argument(
                    "beta must be on the same CUDA device",
                ));
            }
        }

        if !self.scale.is_finite() {
            return Err(FlashInferError::invalid_argument("scale must be finite"));
        }

        Ok(())
    }
}

pub fn gdn_prefill_sm90(params: &GdnPrefillSm90Params) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { gdn_prefill_sm90_with_runtime(runtime, params) }
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_sm90_cudarc<T, O, OS, Q, K, V, C, W>(
    stream: &cudarc::driver::CudaStream,
    output: &mut O,
    output_state: &mut OS,
    q: &Q,
    k: &K,
    v: &V,
    cu_seqlens: &C,
    workspace_buffer: &mut W,
    packed_seq: usize,
    num_q_heads: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_size: usize,
    dtype: DType,
) -> Result<(), FlashInferError>
where
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    OS: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtrMut<f32>,
    Q: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    K: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    V: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    C: cudarc::driver::DeviceSlice<i64> + cudarc::driver::DevicePtr<i64>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    gdn_prefill_sm90_cudarc_with_scale(
        stream,
        output,
        output_state,
        q,
        k,
        v,
        cu_seqlens,
        workspace_buffer,
        packed_seq,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        dtype,
        0.0,
    )
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_sm90_cudarc_with_scale<T, O, OS, Q, K, V, C, W>(
    stream: &cudarc::driver::CudaStream,
    output: &mut O,
    output_state: &mut OS,
    q: &Q,
    k: &K,
    v: &V,
    cu_seqlens: &C,
    workspace_buffer: &mut W,
    packed_seq: usize,
    num_q_heads: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_size: usize,
    dtype: DType,
    scale: f64,
) -> Result<(), FlashInferError>
where
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    OS: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtrMut<f32>,
    Q: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    K: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    V: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    C: cudarc::driver::DeviceSlice<i64> + cudarc::driver::DevicePtr<i64>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    if packed_seq == 0 || num_q_heads == 0 || num_k_heads == 0 || num_v_heads == 0 || head_size == 0
    {
        return Err(FlashInferError::invalid_argument(
            "packed_seq/num_*_heads/head_size must be positive",
        ));
    }
    if cu_seqlens.len() < 2 {
        return Err(FlashInferError::invalid_argument(
            "cu_seqlens length must be at least 2",
        ));
    }
    let workspace_len = workspace_buffer.len();
    if workspace_len == 0 {
        return Err(FlashInferError::invalid_argument(
            "workspace_buffer length must be positive",
        ));
    }

    let num_sab_heads = num_q_heads.max(num_v_heads);
    let num_seqs = cu_seqlens.len() - 1;

    let expected_q = packed_seq
        .checked_mul(num_q_heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("q size overflow"))?;
    let expected_k = packed_seq
        .checked_mul(num_k_heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("k size overflow"))?;
    let expected_v = packed_seq
        .checked_mul(num_v_heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("v size overflow"))?;
    let expected_output = packed_seq
        .checked_mul(num_sab_heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("output size overflow"))?;
    let expected_output_state = num_seqs
        .checked_mul(num_sab_heads)
        .and_then(|v| v.checked_mul(head_size))
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("output_state size overflow"))?;

    if q.len() != expected_q {
        return Err(FlashInferError::invalid_argument(format!(
            "q length ({}) must equal packed_seq * num_q_heads * head_size ({expected_q})",
            q.len()
        )));
    }
    if k.len() != expected_k {
        return Err(FlashInferError::invalid_argument(format!(
            "k length ({}) must equal packed_seq * num_k_heads * head_size ({expected_k})",
            k.len()
        )));
    }
    if v.len() != expected_v {
        return Err(FlashInferError::invalid_argument(format!(
            "v length ({}) must equal packed_seq * num_v_heads * head_size ({expected_v})",
            v.len()
        )));
    }
    if output.len() != expected_output {
        return Err(FlashInferError::invalid_argument(format!(
            "output length ({}) must equal packed_seq * max(num_q_heads, num_v_heads) * head_size ({expected_output})",
            output.len()
        )));
    }
    if output_state.len() != expected_output_state {
        return Err(FlashInferError::invalid_argument(format!(
            "output_state length ({}) must equal (cu_seqlens.len() - 1) * max(num_q_heads, num_v_heads) * head_size * head_size ({expected_output_state})",
            output_state.len()
        )));
    }

    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let (output_state_ptr, _output_state_sync) = output_state.device_ptr_mut(stream);
    let (q_ptr, _q_sync) = q.device_ptr(stream);
    let (k_ptr, _k_sync) = k.device_ptr(stream);
    let (v_ptr, _v_sync) = v.device_ptr(stream);
    let (cu_seqlens_ptr, _cu_seqlens_sync) = cu_seqlens.device_ptr(stream);
    let (workspace_ptr, _workspace_sync) = workspace_buffer.device_ptr_mut(stream);

    let packed_seq_i64 = i64::try_from(packed_seq)
        .map_err(|_| FlashInferError::invalid_argument("packed_seq does not fit in i64"))?;
    let num_q_heads_i64 = i64::try_from(num_q_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_q_heads does not fit in i64"))?;
    let num_k_heads_i64 = i64::try_from(num_k_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_k_heads does not fit in i64"))?;
    let num_v_heads_i64 = i64::try_from(num_v_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_v_heads does not fit in i64"))?;
    let num_sab_heads_i64 = i64::try_from(num_sab_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_sab_heads does not fit in i64"))?;
    let head_size_i64 = i64::try_from(head_size)
        .map_err(|_| FlashInferError::invalid_argument("head_size does not fit in i64"))?;
    let num_seqs_i64 = i64::try_from(num_seqs)
        .map_err(|_| FlashInferError::invalid_argument("num_seqs does not fit in i64"))?;
    let cu_seqlens_len_i64 = i64::try_from(cu_seqlens.len())
        .map_err(|_| FlashInferError::invalid_argument("cu_seqlens length does not fit in i64"))?;
    let workspace_len_i64 = i64::try_from(workspace_len).map_err(|_| {
        FlashInferError::invalid_argument("workspace_buffer length does not fit in i64")
    })?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let q_stride_seq = num_q_heads_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("q stride overflow"))?;
    let k_stride_seq = num_k_heads_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("k stride overflow"))?;
    let v_stride_seq = num_v_heads_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("v stride overflow"))?;
    let output_stride_seq = num_sab_heads_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("output stride overflow"))?;
    let output_state_stride1 = head_size_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("output_state stride overflow"))?;
    let output_state_stride0 = num_sab_heads_i64
        .checked_mul(output_state_stride1)
        .ok_or_else(|| FlashInferError::invalid_argument("output_state stride overflow"))?;

    let params = GdnPrefillSm90Params::new(
        Tensor3DDesc {
            ptr: output_ptr as usize as *const c_void,
            seq_len: packed_seq_i64,
            num_heads: num_sab_heads_i64,
            head_size: head_size_i64,
            stride_seq: output_stride_seq,
            stride_head: head_size_i64,
            stride_dim: 1,
            dtype,
            device_id,
        },
        Tensor4DF32Desc {
            ptr: output_state_ptr as usize as *const c_void,
            dim0: num_seqs_i64,
            dim1: num_sab_heads_i64,
            dim2: head_size_i64,
            dim3: head_size_i64,
            stride0: output_state_stride0,
            stride1: output_state_stride1,
            stride2: head_size_i64,
            stride3: 1,
            device_id,
        },
        Tensor3DDesc {
            ptr: q_ptr as usize as *const c_void,
            seq_len: packed_seq_i64,
            num_heads: num_q_heads_i64,
            head_size: head_size_i64,
            stride_seq: q_stride_seq,
            stride_head: head_size_i64,
            stride_dim: 1,
            dtype,
            device_id,
        },
        Tensor3DDesc {
            ptr: k_ptr as usize as *const c_void,
            seq_len: packed_seq_i64,
            num_heads: num_k_heads_i64,
            head_size: head_size_i64,
            stride_seq: k_stride_seq,
            stride_head: head_size_i64,
            stride_dim: 1,
            dtype,
            device_id,
        },
        Tensor3DDesc {
            ptr: v_ptr as usize as *const c_void,
            seq_len: packed_seq_i64,
            num_heads: num_v_heads_i64,
            head_size: head_size_i64,
            stride_seq: v_stride_seq,
            stride_head: head_size_i64,
            stride_dim: 1,
            dtype,
            device_id,
        },
        Tensor1DI64Desc {
            ptr: cu_seqlens_ptr as usize as *const c_void,
            len: cu_seqlens_len_i64,
            stride: 1,
            device_id,
        },
        Tensor1DU8Desc {
            ptr: workspace_ptr as usize as *const c_void,
            len: workspace_len_i64,
            stride: 1,
            device_id,
        },
        stream.cu_stream().cast(),
    )
    .with_scale(scale);

    gdn_prefill_sm90(&params)
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_sm90_cudarc_with_options<T, O, OS, Q, K, V, C, W, IS, A, B>(
    stream: &cudarc::driver::CudaStream,
    output: &mut O,
    output_state: &mut OS,
    q: &Q,
    k: &K,
    v: &V,
    cu_seqlens: &C,
    workspace_buffer: &mut W,
    packed_seq: usize,
    num_q_heads: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_size: usize,
    dtype: DType,
    input_state: Option<&IS>,
    alpha: Option<&A>,
    beta: Option<&B>,
    scale: f64,
) -> Result<(), FlashInferError>
where
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    OS: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtrMut<f32>,
    Q: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    K: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    V: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    C: cudarc::driver::DeviceSlice<i64> + cudarc::driver::DevicePtr<i64>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
    IS: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    A: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    B: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
{
    if packed_seq == 0 || num_q_heads == 0 || num_k_heads == 0 || num_v_heads == 0 || head_size == 0
    {
        return Err(FlashInferError::invalid_argument(
            "packed_seq/num_*_heads/head_size must be positive",
        ));
    }
    if cu_seqlens.len() < 2 {
        return Err(FlashInferError::invalid_argument(
            "cu_seqlens length must be at least 2",
        ));
    }
    let workspace_len = workspace_buffer.len();
    if workspace_len == 0 {
        return Err(FlashInferError::invalid_argument(
            "workspace_buffer length must be positive",
        ));
    }

    let num_sab_heads = num_q_heads.max(num_v_heads);
    let num_seqs = cu_seqlens.len() - 1;

    let expected_q = packed_seq
        .checked_mul(num_q_heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("q size overflow"))?;
    let expected_k = packed_seq
        .checked_mul(num_k_heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("k size overflow"))?;
    let expected_v = packed_seq
        .checked_mul(num_v_heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("v size overflow"))?;
    let expected_output = packed_seq
        .checked_mul(num_sab_heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("output size overflow"))?;
    let expected_output_state = num_seqs
        .checked_mul(num_sab_heads)
        .and_then(|v| v.checked_mul(head_size))
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| FlashInferError::invalid_argument("output_state size overflow"))?;
    let expected_alpha_beta = packed_seq
        .checked_mul(num_sab_heads)
        .ok_or_else(|| FlashInferError::invalid_argument("alpha/beta size overflow"))?;

    if q.len() != expected_q {
        return Err(FlashInferError::invalid_argument(format!(
            "q length ({}) must equal packed_seq * num_q_heads * head_size ({expected_q})",
            q.len()
        )));
    }
    if k.len() != expected_k {
        return Err(FlashInferError::invalid_argument(format!(
            "k length ({}) must equal packed_seq * num_k_heads * head_size ({expected_k})",
            k.len()
        )));
    }
    if v.len() != expected_v {
        return Err(FlashInferError::invalid_argument(format!(
            "v length ({}) must equal packed_seq * num_v_heads * head_size ({expected_v})",
            v.len()
        )));
    }
    if output.len() != expected_output {
        return Err(FlashInferError::invalid_argument(format!(
            "output length ({}) must equal packed_seq * max(num_q_heads, num_v_heads) * head_size ({expected_output})",
            output.len()
        )));
    }
    if output_state.len() != expected_output_state {
        return Err(FlashInferError::invalid_argument(format!(
            "output_state length ({}) must equal (cu_seqlens.len() - 1) * max(num_q_heads, num_v_heads) * head_size * head_size ({expected_output_state})",
            output_state.len()
        )));
    }

    if let Some(input_state_tensor) = input_state {
        if input_state_tensor.len() != expected_output_state {
            return Err(FlashInferError::invalid_argument(format!(
                "input_state length ({}) must equal output_state length ({expected_output_state})",
                input_state_tensor.len()
            )));
        }
    }
    if let Some(alpha_tensor) = alpha {
        if alpha_tensor.len() != expected_alpha_beta {
            return Err(FlashInferError::invalid_argument(format!(
                "alpha length ({}) must equal packed_seq * max(num_q_heads, num_v_heads) ({expected_alpha_beta})",
                alpha_tensor.len()
            )));
        }
    }
    if let Some(beta_tensor) = beta {
        if beta_tensor.len() != expected_alpha_beta {
            return Err(FlashInferError::invalid_argument(format!(
                "beta length ({}) must equal packed_seq * max(num_q_heads, num_v_heads) ({expected_alpha_beta})",
                beta_tensor.len()
            )));
        }
    }

    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let (output_state_ptr, _output_state_sync) = output_state.device_ptr_mut(stream);
    let (q_ptr, _q_sync) = q.device_ptr(stream);
    let (k_ptr, _k_sync) = k.device_ptr(stream);
    let (v_ptr, _v_sync) = v.device_ptr(stream);
    let (cu_seqlens_ptr, _cu_seqlens_sync) = cu_seqlens.device_ptr(stream);
    let (workspace_ptr, _workspace_sync) = workspace_buffer.device_ptr_mut(stream);

    let input_state_device =
        input_state.map(|input_state_tensor| input_state_tensor.device_ptr(stream));
    let alpha_device = alpha.map(|alpha_tensor| alpha_tensor.device_ptr(stream));
    let beta_device = beta.map(|beta_tensor| beta_tensor.device_ptr(stream));

    let packed_seq_i64 = i64::try_from(packed_seq)
        .map_err(|_| FlashInferError::invalid_argument("packed_seq does not fit in i64"))?;
    let num_q_heads_i64 = i64::try_from(num_q_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_q_heads does not fit in i64"))?;
    let num_k_heads_i64 = i64::try_from(num_k_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_k_heads does not fit in i64"))?;
    let num_v_heads_i64 = i64::try_from(num_v_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_v_heads does not fit in i64"))?;
    let num_sab_heads_i64 = i64::try_from(num_sab_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_sab_heads does not fit in i64"))?;
    let head_size_i64 = i64::try_from(head_size)
        .map_err(|_| FlashInferError::invalid_argument("head_size does not fit in i64"))?;
    let num_seqs_i64 = i64::try_from(num_seqs)
        .map_err(|_| FlashInferError::invalid_argument("num_seqs does not fit in i64"))?;
    let cu_seqlens_len_i64 = i64::try_from(cu_seqlens.len())
        .map_err(|_| FlashInferError::invalid_argument("cu_seqlens length does not fit in i64"))?;
    let workspace_len_i64 = i64::try_from(workspace_len).map_err(|_| {
        FlashInferError::invalid_argument("workspace_buffer length does not fit in i64")
    })?;
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let q_stride_seq = num_q_heads_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("q stride overflow"))?;
    let k_stride_seq = num_k_heads_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("k stride overflow"))?;
    let v_stride_seq = num_v_heads_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("v stride overflow"))?;
    let output_stride_seq = num_sab_heads_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("output stride overflow"))?;
    let output_state_stride1 = head_size_i64
        .checked_mul(head_size_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("output_state stride overflow"))?;
    let output_state_stride0 = num_sab_heads_i64
        .checked_mul(output_state_stride1)
        .ok_or_else(|| FlashInferError::invalid_argument("output_state stride overflow"))?;

    let mut params = GdnPrefillSm90Params::new(
        Tensor3DDesc {
            ptr: output_ptr as usize as *const c_void,
            seq_len: packed_seq_i64,
            num_heads: num_sab_heads_i64,
            head_size: head_size_i64,
            stride_seq: output_stride_seq,
            stride_head: head_size_i64,
            stride_dim: 1,
            dtype,
            device_id,
        },
        Tensor4DF32Desc {
            ptr: output_state_ptr as usize as *const c_void,
            dim0: num_seqs_i64,
            dim1: num_sab_heads_i64,
            dim2: head_size_i64,
            dim3: head_size_i64,
            stride0: output_state_stride0,
            stride1: output_state_stride1,
            stride2: head_size_i64,
            stride3: 1,
            device_id,
        },
        Tensor3DDesc {
            ptr: q_ptr as usize as *const c_void,
            seq_len: packed_seq_i64,
            num_heads: num_q_heads_i64,
            head_size: head_size_i64,
            stride_seq: q_stride_seq,
            stride_head: head_size_i64,
            stride_dim: 1,
            dtype,
            device_id,
        },
        Tensor3DDesc {
            ptr: k_ptr as usize as *const c_void,
            seq_len: packed_seq_i64,
            num_heads: num_k_heads_i64,
            head_size: head_size_i64,
            stride_seq: k_stride_seq,
            stride_head: head_size_i64,
            stride_dim: 1,
            dtype,
            device_id,
        },
        Tensor3DDesc {
            ptr: v_ptr as usize as *const c_void,
            seq_len: packed_seq_i64,
            num_heads: num_v_heads_i64,
            head_size: head_size_i64,
            stride_seq: v_stride_seq,
            stride_head: head_size_i64,
            stride_dim: 1,
            dtype,
            device_id,
        },
        Tensor1DI64Desc {
            ptr: cu_seqlens_ptr as usize as *const c_void,
            len: cu_seqlens_len_i64,
            stride: 1,
            device_id,
        },
        Tensor1DU8Desc {
            ptr: workspace_ptr as usize as *const c_void,
            len: workspace_len_i64,
            stride: 1,
            device_id,
        },
        stream.cu_stream().cast(),
    )
    .with_scale(scale);

    if let Some((input_state_ptr, _input_state_sync)) = input_state_device {
        params = params.with_input_state(Tensor4DF32Desc {
            ptr: input_state_ptr as usize as *const c_void,
            dim0: num_seqs_i64,
            dim1: num_sab_heads_i64,
            dim2: head_size_i64,
            dim3: head_size_i64,
            stride0: output_state_stride0,
            stride1: output_state_stride1,
            stride2: head_size_i64,
            stride3: 1,
            device_id,
        });
    }
    if let Some((alpha_ptr, _alpha_sync)) = alpha_device {
        params = params.with_alpha(Tensor2DF32Desc {
            ptr: alpha_ptr as usize as *const c_void,
            rows: packed_seq_i64,
            cols: num_sab_heads_i64,
            stride_row: num_sab_heads_i64,
            stride_col: 1,
            device_id,
        });
    }
    if let Some((beta_ptr, _beta_sync)) = beta_device {
        params = params.with_beta(Tensor2DF32Desc {
            ptr: beta_ptr as usize as *const c_void,
            rows: packed_seq_i64,
            cols: num_sab_heads_i64,
            stride_row: num_sab_heads_i64,
            stride_col: 1,
            device_id,
        });
    }

    gdn_prefill_sm90(&params)
}

unsafe fn gdn_prefill_sm90_with_runtime(
    runtime: &FlashInferRuntime,
    params: &GdnPrefillSm90Params,
) -> Result<(), FlashInferError> {
    let mut output_shape = [
        params.output.seq_len,
        params.output.num_heads,
        params.output.head_size,
    ];
    let mut output_strides = [
        params.output.stride_seq,
        params.output.stride_head,
        params.output.stride_dim,
    ];
    let output_tensor = DLTensor {
        data: params.output.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.output.device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_norm_dtype(params.output.dtype),
        shape: output_shape.as_mut_ptr(),
        strides: output_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut output_state_shape = [
        params.output_state.dim0,
        params.output_state.dim1,
        params.output_state.dim2,
        params.output_state.dim3,
    ];
    let mut output_state_strides = [
        params.output_state.stride0,
        params.output_state.stride1,
        params.output_state.stride2,
        params.output_state.stride3,
    ];
    let output_state_tensor = DLTensor {
        data: params.output_state.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.output_state.device_id,
        },
        ndim: 4,
        dtype: dl_dtype_f32(),
        shape: output_state_shape.as_mut_ptr(),
        strides: output_state_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut q_shape = [params.q.seq_len, params.q.num_heads, params.q.head_size];
    let mut q_strides = [
        params.q.stride_seq,
        params.q.stride_head,
        params.q.stride_dim,
    ];
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

    let mut k_shape = [params.k.seq_len, params.k.num_heads, params.k.head_size];
    let mut k_strides = [
        params.k.stride_seq,
        params.k.stride_head,
        params.k.stride_dim,
    ];
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

    let mut v_shape = [params.v.seq_len, params.v.num_heads, params.v.head_size];
    let mut v_strides = [
        params.v.stride_seq,
        params.v.stride_head,
        params.v.stride_dim,
    ];
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

    let mut cu_seqlens_shape = [params.cu_seqlens.len];
    let mut cu_seqlens_strides = [params.cu_seqlens.stride];
    let cu_seqlens_tensor = DLTensor {
        data: params.cu_seqlens.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.cu_seqlens.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i64(),
        shape: cu_seqlens_shape.as_mut_ptr(),
        strides: cu_seqlens_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut workspace_shape = [params.workspace_buffer.len];
    let mut workspace_strides = [params.workspace_buffer.stride];
    let workspace_tensor = DLTensor {
        data: params.workspace_buffer.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.workspace_buffer.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_u8(),
        shape: workspace_shape.as_mut_ptr(),
        strides: workspace_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut input_state_shape = [0_i64; 4];
    let mut input_state_strides = [0_i64; 4];
    let input_state_tensor = params.input_state.map(|input_state| {
        input_state_shape = [
            input_state.dim0,
            input_state.dim1,
            input_state.dim2,
            input_state.dim3,
        ];
        input_state_strides = [
            input_state.stride0,
            input_state.stride1,
            input_state.stride2,
            input_state.stride3,
        ];
        DLTensor {
            data: input_state.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: input_state.device_id,
            },
            ndim: 4,
            dtype: dl_dtype_f32(),
            shape: input_state_shape.as_mut_ptr(),
            strides: input_state_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    });

    let mut alpha_shape = [0_i64; 2];
    let mut alpha_strides = [0_i64; 2];
    let alpha_tensor = params.alpha.map(|alpha| {
        alpha_shape = [alpha.rows, alpha.cols];
        alpha_strides = [alpha.stride_row, alpha.stride_col];
        DLTensor {
            data: alpha.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: alpha.device_id,
            },
            ndim: 2,
            dtype: dl_dtype_f32(),
            shape: alpha_shape.as_mut_ptr(),
            strides: alpha_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    });

    let mut beta_shape = [0_i64; 2];
    let mut beta_strides = [0_i64; 2];
    let beta_tensor = params.beta.map(|beta| {
        beta_shape = [beta.rows, beta.cols];
        beta_strides = [beta.stride_row, beta.stride_col];
        DLTensor {
            data: beta.ptr.cast_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: beta.device_id,
            },
            ndim: 2,
            dtype: dl_dtype_f32(),
            shape: beta_shape.as_mut_ptr(),
            strides: beta_strides.as_mut_ptr(),
            byte_offset: 0,
        }
    });

    let input_state_any = input_state_tensor.as_ref().map_or_else(any_none, |tensor| {
        any_dltensor_ptr(tensor as *const DLTensor)
    });
    let alpha_any = alpha_tensor.as_ref().map_or_else(any_none, |tensor| {
        any_dltensor_ptr(tensor as *const DLTensor)
    });
    let beta_any = beta_tensor.as_ref().map_or_else(any_none, |tensor| {
        any_dltensor_ptr(tensor as *const DLTensor)
    });

    let args: [TVMFFIAny; 11] = [
        any_dltensor_ptr(&output_tensor),
        any_dltensor_ptr(&output_state_tensor),
        any_dltensor_ptr(&q_tensor),
        any_dltensor_ptr(&k_tensor),
        any_dltensor_ptr(&v_tensor),
        any_dltensor_ptr(&cu_seqlens_tensor),
        input_state_any,
        alpha_any,
        beta_any,
        any_f64(params.scale),
        any_dltensor_ptr(&workspace_tensor),
    ];
    let mut result = any_none();

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream = unsafe { runtime.set_stream(params.q.device_id, params.stream)? };
    let mut restore_guard = StreamRestoreGuard::new(runtime, params.q.device_id, previous_stream);
    // SAFETY: argument packing follows TVMFFIAny ABI.
    let call_result = unsafe {
        runtime.call_gdn_prefill(args.as_ptr(), args.len() as i32, &mut result as *mut _)
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

fn dl_dtype_i64() -> DLDataType {
    DLDataType {
        code: KDL_INT,
        bits: 64,
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
            "{name} must be contiguous (stride == 1)"
        )));
    }
    Ok(())
}

fn check_contiguous_2d(
    name: &str,
    rows: i64,
    cols: i64,
    stride_row: i64,
    stride_col: i64,
) -> Result<(), FlashInferError> {
    let expected_stride_row = cols;
    if stride_col != 1 || stride_row != expected_stride_row {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be contiguous; expected strides [{expected_stride_row}, 1], got [{stride_row}, {stride_col}]"
        )));
    }
    if rows <= 0 || cols <= 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} dimensions must be positive"
        )));
    }
    Ok(())
}

fn check_contiguous_3d(
    name: &str,
    dim0: i64,
    dim1: i64,
    dim2: i64,
    stride0: i64,
    stride1: i64,
    stride2: i64,
) -> Result<(), FlashInferError> {
    if dim0 <= 0 || dim1 <= 0 || dim2 <= 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} dimensions must be positive"
        )));
    }

    let expected_stride1 = dim2;
    let expected_stride0 = dim1.checked_mul(dim2).ok_or_else(|| {
        FlashInferError::invalid_argument(format!("{name} stride computation overflow"))
    })?;

    if stride2 != 1 || stride1 != expected_stride1 || stride0 != expected_stride0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be contiguous; expected strides [{expected_stride0}, {expected_stride1}, 1], got [{stride0}, {stride1}, {stride2}]"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_contiguous_4d(
    name: &str,
    dim0: i64,
    dim1: i64,
    dim2: i64,
    dim3: i64,
    stride0: i64,
    stride1: i64,
    stride2: i64,
    stride3: i64,
) -> Result<(), FlashInferError> {
    let expected_stride2 = dim3;
    let expected_stride1 = dim2.checked_mul(dim3).ok_or_else(|| {
        FlashInferError::invalid_argument(format!("{name} stride computation overflow"))
    })?;
    let expected_stride0 = dim1
        .checked_mul(dim2)
        .and_then(|v| v.checked_mul(dim3))
        .ok_or_else(|| {
            FlashInferError::invalid_argument(format!("{name} stride computation overflow"))
        })?;

    if stride3 != 1
        || stride2 != expected_stride2
        || stride1 != expected_stride1
        || stride0 != expected_stride0
    {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} must be contiguous; expected strides [{expected_stride0}, {expected_stride1}, {expected_stride2}, 1], got [{stride0}, {stride1}, {stride2}, {stride3}]"
        )));
    }

    if dim0 <= 0 || dim1 <= 0 || dim2 <= 0 || dim3 <= 0 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} dimensions must be positive"
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

    fn valid_params() -> GdnPrefillSm90Params {
        GdnPrefillSm90Params::new(
            Tensor3DDesc {
                ptr: non_null(),
                seq_len: 8,
                num_heads: 4,
                head_size: 64,
                stride_seq: 256,
                stride_head: 64,
                stride_dim: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor4DF32Desc {
                ptr: non_null(),
                dim0: 2,
                dim1: 4,
                dim2: 64,
                dim3: 64,
                stride0: 16384,
                stride1: 4096,
                stride2: 64,
                stride3: 1,
                device_id: 0,
            },
            Tensor3DDesc {
                ptr: non_null(),
                seq_len: 8,
                num_heads: 4,
                head_size: 64,
                stride_seq: 256,
                stride_head: 64,
                stride_dim: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor3DDesc {
                ptr: non_null(),
                seq_len: 8,
                num_heads: 2,
                head_size: 64,
                stride_seq: 128,
                stride_head: 64,
                stride_dim: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor3DDesc {
                ptr: non_null(),
                seq_len: 8,
                num_heads: 2,
                head_size: 64,
                stride_seq: 128,
                stride_head: 64,
                stride_dim: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            Tensor1DI64Desc {
                ptr: non_null(),
                len: 3,
                stride: 1,
                device_id: 0,
            },
            Tensor1DU8Desc {
                ptr: non_null(),
                len: 1024,
                stride: 1,
                device_id: 0,
            },
            std::ptr::null_mut(),
        )
    }

    #[test]
    fn validate_rejects_bad_head_ratio() {
        let mut params = valid_params();
        params.v.num_heads = 3;
        params.k.num_heads = 3;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_output_head_mismatch() {
        let mut params = valid_params();
        params.output.num_heads = 2;
        params.output.stride_seq = 128;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_state_shape_mismatch() {
        let mut params = valid_params();
        params.output_state.dim1 = 2;
        params.output_state.stride0 = 8192;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_optional_shape_mismatch() {
        let params = valid_params().with_alpha(Tensor2DF32Desc {
            ptr: non_null(),
            rows: 8,
            cols: 3,
            stride_row: 3,
            stride_col: 1,
            device_id: 0,
        });
        assert!(params.validate().is_err());
    }
}
