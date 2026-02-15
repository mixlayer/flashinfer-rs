use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_FLOAT, KDL_INT, TVMFFIAny,
    any_dltensor_ptr, any_i64, any_none,
};
use crate::mha_batch_prefill::MhaTensor1DI32Desc;
use crate::mha_batch_prefill_paged::MhaTensor4DDesc;
use crate::mha_prefill::{MhaQkvLayout, MhaTensor3DDesc};
use crate::norm::DType;
use crate::runtime::FlashInferRuntime;

#[derive(Debug, Clone, Copy)]
pub struct PagedMlaTensor2DDesc {
    pub ptr: *const c_void,
    pub rows: i64,
    pub cols: i64,
    pub stride_row: i64,
    pub stride_col: i64,
    pub dtype: DType,
    pub device_id: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct PagedKvAppendParams {
    /// Key entries to append, rank-3: `[nnz, num_kv_heads, head_dim]`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/page.cu::append_paged_kv_cache` (`CHECK_DIM(3, append_key)`).
    pub append_key: MhaTensor3DDesc,
    /// Value entries to append, rank-3: `[nnz, num_kv_heads, head_dim]`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/page.cu::append_paged_kv_cache` (`CHECK_DIM(3, append_value)`).
    pub append_value: MhaTensor3DDesc,
    /// Batch index per appended token, rank-1 int32: `[nnz]`.
    pub batch_indices: MhaTensor1DI32Desc,
    /// Position per appended token within request sequence, rank-1 int32: `[nnz]`.
    pub positions: MhaTensor1DI32Desc,
    /// Paged key cache, rank-4:
    /// - `NHD`: `[num_pages, page_size, num_kv_heads, head_dim]`
    /// - `HND`: `[num_pages, num_kv_heads, page_size, head_dim]`.
    pub paged_k_cache: MhaTensor4DDesc,
    /// Paged value cache, rank-4:
    /// - `NHD`: `[num_pages, page_size, num_kv_heads, head_dim]`
    /// - `HND`: `[num_pages, num_kv_heads, page_size, head_dim]`.
    pub paged_v_cache: MhaTensor4DDesc,
    /// Page-table indices, rank-1 int32: `[kv_indptr[-1]]`.
    pub kv_indices: MhaTensor1DI32Desc,
    /// Page-table indptr, rank-1 int32: `[batch_size + 1]`.
    pub kv_indptr: MhaTensor1DI32Desc,
    /// Valid length in last page for each request, rank-1 int32: `[batch_size]`.
    pub kv_last_page_len: MhaTensor1DI32Desc,
    /// KV page layout enum (`NHD` or `HND`).
    pub kv_layout: MhaQkvLayout,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl PagedKvAppendParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        append_key: MhaTensor3DDesc,
        append_value: MhaTensor3DDesc,
        batch_indices: MhaTensor1DI32Desc,
        positions: MhaTensor1DI32Desc,
        paged_k_cache: MhaTensor4DDesc,
        paged_v_cache: MhaTensor4DDesc,
        kv_indices: MhaTensor1DI32Desc,
        kv_indptr: MhaTensor1DI32Desc,
        kv_last_page_len: MhaTensor1DI32Desc,
        stream: *mut c_void,
    ) -> Self {
        Self {
            append_key,
            append_value,
            batch_indices,
            positions,
            paged_k_cache,
            paged_v_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            kv_layout: MhaQkvLayout::Nhd,
            stream,
        }
    }

    pub fn with_kv_layout(mut self, kv_layout: MhaQkvLayout) -> Self {
        self.kv_layout = kv_layout;
        self
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.append_key.ptr, "append_key")?;
        check_non_null(self.append_value.ptr, "append_value")?;
        check_non_null(self.batch_indices.ptr, "batch_indices")?;
        check_non_null(self.positions.ptr, "positions")?;
        check_non_null(self.paged_k_cache.ptr, "paged_k_cache")?;
        check_non_null(self.paged_v_cache.ptr, "paged_v_cache")?;
        check_non_null(self.kv_indices.ptr, "kv_indices")?;
        check_non_null(self.kv_indptr.ptr, "kv_indptr")?;
        check_non_null(self.kv_last_page_len.ptr, "kv_last_page_len")?;

        check_positive("append_key.dim0", self.append_key.dim0)?;
        check_positive("append_key.dim1", self.append_key.dim1)?;
        check_positive("append_key.dim2", self.append_key.dim2)?;
        check_positive("append_value.dim0", self.append_value.dim0)?;
        check_positive("append_value.dim1", self.append_value.dim1)?;
        check_positive("append_value.dim2", self.append_value.dim2)?;
        check_positive("batch_indices.len", self.batch_indices.len)?;
        check_positive("positions.len", self.positions.len)?;
        check_positive("paged_k_cache.dim0", self.paged_k_cache.dim0)?;
        check_positive("paged_k_cache.dim1", self.paged_k_cache.dim1)?;
        check_positive("paged_k_cache.dim2", self.paged_k_cache.dim2)?;
        check_positive("paged_k_cache.dim3", self.paged_k_cache.dim3)?;
        check_positive("paged_v_cache.dim0", self.paged_v_cache.dim0)?;
        check_positive("paged_v_cache.dim1", self.paged_v_cache.dim1)?;
        check_positive("paged_v_cache.dim2", self.paged_v_cache.dim2)?;
        check_positive("paged_v_cache.dim3", self.paged_v_cache.dim3)?;
        check_positive("kv_indices.len", self.kv_indices.len)?;
        check_positive("kv_indptr.len", self.kv_indptr.len)?;
        check_positive("kv_last_page_len.len", self.kv_last_page_len.len)?;

        check_contiguous_1d("batch_indices", self.batch_indices.stride)?;
        check_contiguous_1d("positions", self.positions.stride)?;
        check_contiguous_1d("kv_indices", self.kv_indices.stride)?;
        check_contiguous_1d("kv_indptr", self.kv_indptr.stride)?;
        check_contiguous_1d("kv_last_page_len", self.kv_last_page_len.stride)?;

        check_last_dim_contiguous("append_key", self.append_key.stride2)?;
        check_last_dim_contiguous("append_value", self.append_value.stride2)?;
        check_last_dim_contiguous("paged_k_cache", self.paged_k_cache.stride3)?;
        check_last_dim_contiguous("paged_v_cache", self.paged_v_cache.stride3)?;

        if self.append_key.dtype != self.append_value.dtype
            || self.append_key.dtype != self.paged_k_cache.dtype
            || self.append_key.dtype != self.paged_v_cache.dtype
        {
            return Err(FlashInferError::invalid_argument(
                "append_key/append_value/paged_k_cache/paged_v_cache dtype mismatch",
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

        if self.append_key.dim0 != self.append_value.dim0 {
            return Err(FlashInferError::invalid_argument(
                "append_key and append_value nnz mismatch",
            ));
        }
        if self.append_key.dim1 != self.append_value.dim1
            || self.append_key.dim2 != self.append_value.dim2
        {
            return Err(FlashInferError::invalid_argument(
                "append_key and append_value shape mismatch",
            ));
        }

        let append_nnz = self.append_key.dim0;
        if self.batch_indices.len != append_nnz || self.positions.len != append_nnz {
            return Err(FlashInferError::invalid_argument(
                "batch_indices and positions length must equal append nnz",
            ));
        }

        let batch_size = self.kv_last_page_len.len;
        if self.kv_indptr.len != batch_size + 1 {
            return Err(FlashInferError::invalid_argument(
                "kv_indptr length must equal kv_last_page_len length + 1",
            ));
        }

        if self.kv_indptr.len < 2 {
            return Err(FlashInferError::invalid_argument(
                "kv_indptr length must be at least 2",
            ));
        }

        let (_num_pages_k, page_size_k, num_kv_heads_k, head_dim_k) =
            decode_paged_layout(self.paged_k_cache, self.kv_layout);
        let (_num_pages_v, page_size_v, num_kv_heads_v, head_dim_v) =
            decode_paged_layout(self.paged_v_cache, self.kv_layout);

        if page_size_k != page_size_v
            || num_kv_heads_k != num_kv_heads_v
            || head_dim_k != head_dim_v
        {
            return Err(FlashInferError::invalid_argument(
                "paged_k_cache and paged_v_cache layout dimensions must match",
            ));
        }

        if self.append_key.dim1 != num_kv_heads_k || self.append_key.dim2 != head_dim_k {
            return Err(FlashInferError::invalid_argument(
                "append_key shape must match paged cache [num_kv_heads, head_dim]",
            ));
        }

        if self.append_value.dim1 != num_kv_heads_v || self.append_value.dim2 != head_dim_v {
            return Err(FlashInferError::invalid_argument(
                "append_value shape must match paged cache [num_kv_heads, head_dim]",
            ));
        }

        let device_id = self.append_key.device_id;
        if self.append_value.device_id != device_id
            || self.batch_indices.device_id != device_id
            || self.positions.device_id != device_id
            || self.paged_k_cache.device_id != device_id
            || self.paged_v_cache.device_id != device_id
            || self.kv_indices.device_id != device_id
            || self.kv_indptr.device_id != device_id
            || self.kv_last_page_len.device_id != device_id
        {
            return Err(FlashInferError::invalid_argument(
                "all tensors must be on the same CUDA device",
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PagedMlaKvAppendParams {
    /// Compressed-KV entries to append, rank-2: `[nnz, 512]`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/page.cu::append_paged_mla_kv_cache` (`CHECK_DIM(2, append_ckv)`).
    pub append_ckv: PagedMlaTensor2DDesc,
    /// KPE entries to append, rank-2: `[nnz, 64]`.
    ///
    /// Cross-reference:
    /// `flashinfer/csrc/page.cu::append_paged_mla_kv_cache` (`CHECK_DIM(2, append_kpe)`).
    pub append_kpe: PagedMlaTensor2DDesc,
    /// Batch index per appended token, rank-1 int32: `[nnz]`.
    pub batch_indices: MhaTensor1DI32Desc,
    /// Position per appended token within request sequence, rank-1 int32: `[nnz]`.
    pub positions: MhaTensor1DI32Desc,
    /// MLA compressed KV cache, rank-3: `[num_pages, page_size, 512]`.
    pub ckv_cache: MhaTensor3DDesc,
    /// MLA KPE cache, rank-3: `[num_pages, page_size, 64]`.
    pub kpe_cache: MhaTensor3DDesc,
    /// Page-table indices, rank-1 int32: `[kv_indptr[-1]]`.
    pub kv_indices: MhaTensor1DI32Desc,
    /// Page-table indptr, rank-1 int32: `[batch_size + 1]`.
    pub kv_indptr: MhaTensor1DI32Desc,
    /// Valid length in last page for each request, rank-1 int32: `[batch_size]`.
    pub kv_last_page_len: MhaTensor1DI32Desc,
    /// CUDA stream (`cudaStream_t`) used for async launch.
    pub stream: *mut c_void,
}

impl PagedMlaKvAppendParams {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        append_ckv: PagedMlaTensor2DDesc,
        append_kpe: PagedMlaTensor2DDesc,
        batch_indices: MhaTensor1DI32Desc,
        positions: MhaTensor1DI32Desc,
        ckv_cache: MhaTensor3DDesc,
        kpe_cache: MhaTensor3DDesc,
        kv_indices: MhaTensor1DI32Desc,
        kv_indptr: MhaTensor1DI32Desc,
        kv_last_page_len: MhaTensor1DI32Desc,
        stream: *mut c_void,
    ) -> Self {
        Self {
            append_ckv,
            append_kpe,
            batch_indices,
            positions,
            ckv_cache,
            kpe_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            stream,
        }
    }

    pub fn validate(&self) -> Result<(), FlashInferError> {
        check_non_null(self.append_ckv.ptr, "append_ckv")?;
        check_non_null(self.append_kpe.ptr, "append_kpe")?;
        check_non_null(self.batch_indices.ptr, "batch_indices")?;
        check_non_null(self.positions.ptr, "positions")?;
        check_non_null(self.ckv_cache.ptr, "ckv_cache")?;
        check_non_null(self.kpe_cache.ptr, "kpe_cache")?;
        check_non_null(self.kv_indices.ptr, "kv_indices")?;
        check_non_null(self.kv_indptr.ptr, "kv_indptr")?;
        check_non_null(self.kv_last_page_len.ptr, "kv_last_page_len")?;

        check_positive("append_ckv.rows", self.append_ckv.rows)?;
        check_positive("append_ckv.cols", self.append_ckv.cols)?;
        check_positive("append_kpe.rows", self.append_kpe.rows)?;
        check_positive("append_kpe.cols", self.append_kpe.cols)?;
        check_positive("batch_indices.len", self.batch_indices.len)?;
        check_positive("positions.len", self.positions.len)?;
        check_positive("ckv_cache.dim0", self.ckv_cache.dim0)?;
        check_positive("ckv_cache.dim1", self.ckv_cache.dim1)?;
        check_positive("ckv_cache.dim2", self.ckv_cache.dim2)?;
        check_positive("kpe_cache.dim0", self.kpe_cache.dim0)?;
        check_positive("kpe_cache.dim1", self.kpe_cache.dim1)?;
        check_positive("kpe_cache.dim2", self.kpe_cache.dim2)?;
        check_positive("kv_indices.len", self.kv_indices.len)?;
        check_positive("kv_indptr.len", self.kv_indptr.len)?;
        check_positive("kv_last_page_len.len", self.kv_last_page_len.len)?;

        check_contiguous_1d("batch_indices", self.batch_indices.stride)?;
        check_contiguous_1d("positions", self.positions.stride)?;
        check_contiguous_1d("kv_indices", self.kv_indices.stride)?;
        check_contiguous_1d("kv_indptr", self.kv_indptr.stride)?;
        check_contiguous_1d("kv_last_page_len", self.kv_last_page_len.stride)?;

        check_last_dim_contiguous("append_ckv", self.append_ckv.stride_col)?;
        check_last_dim_contiguous("append_kpe", self.append_kpe.stride_col)?;
        check_last_dim_contiguous("ckv_cache", self.ckv_cache.stride2)?;
        check_last_dim_contiguous("kpe_cache", self.kpe_cache.stride2)?;

        if self.append_ckv.dtype != self.append_kpe.dtype
            || self.append_ckv.dtype != self.ckv_cache.dtype
            || self.append_ckv.dtype != self.kpe_cache.dtype
        {
            return Err(FlashInferError::invalid_argument(
                "append_ckv/append_kpe/ckv_cache/kpe_cache dtype mismatch",
            ));
        }

        if self.append_ckv.rows != self.append_kpe.rows {
            return Err(FlashInferError::invalid_argument(
                "append_ckv and append_kpe nnz mismatch",
            ));
        }

        let nnz = self.append_ckv.rows;
        if self.batch_indices.len != nnz || self.positions.len != nnz {
            return Err(FlashInferError::invalid_argument(
                "batch_indices and positions length must equal append nnz",
            ));
        }

        if self.kv_indptr.len != self.kv_last_page_len.len + 1 {
            return Err(FlashInferError::invalid_argument(
                "kv_indptr length must equal kv_last_page_len length + 1",
            ));
        }

        if self.kv_indptr.len < 2 {
            return Err(FlashInferError::invalid_argument(
                "kv_indptr length must be at least 2",
            ));
        }

        if self.ckv_cache.dim0 != self.kpe_cache.dim0 || self.ckv_cache.dim1 != self.kpe_cache.dim1
        {
            return Err(FlashInferError::invalid_argument(
                "ckv_cache and kpe_cache must share [num_pages, page_size]",
            ));
        }

        if self.ckv_cache.dim2 != self.append_ckv.cols
            || self.kpe_cache.dim2 != self.append_kpe.cols
        {
            return Err(FlashInferError::invalid_argument(
                "append feature dimensions must match cache feature dimensions",
            ));
        }

        if self.append_ckv.cols != 512 || self.append_kpe.cols != 64 {
            return Err(FlashInferError::invalid_argument(
                "MLA append requires ckv_dim=512 and kpe_dim=64",
            ));
        }

        let device_id = self.append_ckv.device_id;
        if self.append_kpe.device_id != device_id
            || self.batch_indices.device_id != device_id
            || self.positions.device_id != device_id
            || self.ckv_cache.device_id != device_id
            || self.kpe_cache.device_id != device_id
            || self.kv_indices.device_id != device_id
            || self.kv_indptr.device_id != device_id
            || self.kv_last_page_len.device_id != device_id
        {
            return Err(FlashInferError::invalid_argument(
                "all tensors must be on the same CUDA device",
            ));
        }

        Ok(())
    }
}

pub fn append_paged_kv_cache(params: &PagedKvAppendParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: all FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { append_paged_kv_cache_with_runtime(runtime, params) }
}

pub fn append_paged_mla_kv_cache(params: &PagedMlaKvAppendParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: all FFI preconditions are validated by `params.validate` and runtime initialization.
    unsafe { append_paged_mla_kv_cache_with_runtime(runtime, params) }
}

unsafe fn append_paged_kv_cache_with_runtime(
    runtime: &FlashInferRuntime,
    params: &PagedKvAppendParams,
) -> Result<(), FlashInferError> {
    let mut append_key_shape = [
        params.append_key.dim0,
        params.append_key.dim1,
        params.append_key.dim2,
    ];
    let mut append_key_strides = [
        params.append_key.stride0,
        params.append_key.stride1,
        params.append_key.stride2,
    ];
    let append_key_tensor = DLTensor {
        data: params.append_key.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.append_key.device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_norm_dtype(params.append_key.dtype),
        shape: append_key_shape.as_mut_ptr(),
        strides: append_key_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut append_value_shape = [
        params.append_value.dim0,
        params.append_value.dim1,
        params.append_value.dim2,
    ];
    let mut append_value_strides = [
        params.append_value.stride0,
        params.append_value.stride1,
        params.append_value.stride2,
    ];
    let append_value_tensor = DLTensor {
        data: params.append_value.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.append_value.device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_norm_dtype(params.append_value.dtype),
        shape: append_value_shape.as_mut_ptr(),
        strides: append_value_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut batch_indices_shape = [params.batch_indices.len];
    let mut batch_indices_strides = [params.batch_indices.stride];
    let batch_indices_tensor = DLTensor {
        data: params.batch_indices.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.batch_indices.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: batch_indices_shape.as_mut_ptr(),
        strides: batch_indices_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut positions_shape = [params.positions.len];
    let mut positions_strides = [params.positions.stride];
    let positions_tensor = DLTensor {
        data: params.positions.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.positions.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: positions_shape.as_mut_ptr(),
        strides: positions_strides.as_mut_ptr(),
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

    let mut kv_indices_shape = [params.kv_indices.len];
    let mut kv_indices_strides = [params.kv_indices.stride];
    let kv_indices_tensor = DLTensor {
        data: params.kv_indices.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.kv_indices.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: kv_indices_shape.as_mut_ptr(),
        strides: kv_indices_strides.as_mut_ptr(),
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

    let mut kv_last_page_len_shape = [params.kv_last_page_len.len];
    let mut kv_last_page_len_strides = [params.kv_last_page_len.stride];
    let kv_last_page_len_tensor = DLTensor {
        data: params.kv_last_page_len.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.kv_last_page_len.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: kv_last_page_len_shape.as_mut_ptr(),
        strides: kv_last_page_len_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut args = [any_none(); 10];
    pack_paged_kv_append_args(
        &mut args,
        &append_key_tensor,
        &append_value_tensor,
        &batch_indices_tensor,
        &positions_tensor,
        &paged_k_tensor,
        &paged_v_tensor,
        &kv_indices_tensor,
        &kv_indptr_tensor,
        &kv_last_page_len_tensor,
        params.kv_layout,
    );

    let mut result = any_none();

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream =
        unsafe { runtime.set_stream(params.append_key.device_id, params.stream)? };
    let mut restore_guard =
        StreamRestoreGuard::new(runtime, params.append_key.device_id, previous_stream);

    // SAFETY: argument packing follows TVMFFIAny ABI.
    let call_result = unsafe {
        runtime.call_append_paged_kv_cache(args.as_ptr(), args.len() as i32, &mut result as *mut _)
    };
    let restore_result = restore_guard.restore_now();

    match (call_result, restore_result) {
        (Err(call_error), _) => Err(call_error),
        (Ok(()), Err(restore_error)) => Err(restore_error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

unsafe fn append_paged_mla_kv_cache_with_runtime(
    runtime: &FlashInferRuntime,
    params: &PagedMlaKvAppendParams,
) -> Result<(), FlashInferError> {
    let mut append_ckv_shape = [params.append_ckv.rows, params.append_ckv.cols];
    let mut append_ckv_strides = [params.append_ckv.stride_row, params.append_ckv.stride_col];
    let append_ckv_tensor = DLTensor {
        data: params.append_ckv.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.append_ckv.device_id,
        },
        ndim: 2,
        dtype: dl_dtype_from_norm_dtype(params.append_ckv.dtype),
        shape: append_ckv_shape.as_mut_ptr(),
        strides: append_ckv_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut append_kpe_shape = [params.append_kpe.rows, params.append_kpe.cols];
    let mut append_kpe_strides = [params.append_kpe.stride_row, params.append_kpe.stride_col];
    let append_kpe_tensor = DLTensor {
        data: params.append_kpe.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.append_kpe.device_id,
        },
        ndim: 2,
        dtype: dl_dtype_from_norm_dtype(params.append_kpe.dtype),
        shape: append_kpe_shape.as_mut_ptr(),
        strides: append_kpe_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut batch_indices_shape = [params.batch_indices.len];
    let mut batch_indices_strides = [params.batch_indices.stride];
    let batch_indices_tensor = DLTensor {
        data: params.batch_indices.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.batch_indices.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: batch_indices_shape.as_mut_ptr(),
        strides: batch_indices_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut positions_shape = [params.positions.len];
    let mut positions_strides = [params.positions.stride];
    let positions_tensor = DLTensor {
        data: params.positions.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.positions.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: positions_shape.as_mut_ptr(),
        strides: positions_strides.as_mut_ptr(),
        byte_offset: 0,
    };

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
    let ckv_tensor = DLTensor {
        data: params.ckv_cache.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.ckv_cache.device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_norm_dtype(params.ckv_cache.dtype),
        shape: ckv_shape.as_mut_ptr(),
        strides: ckv_strides.as_mut_ptr(),
        byte_offset: 0,
    };

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
    let kpe_tensor = DLTensor {
        data: params.kpe_cache.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.kpe_cache.device_id,
        },
        ndim: 3,
        dtype: dl_dtype_from_norm_dtype(params.kpe_cache.dtype),
        shape: kpe_shape.as_mut_ptr(),
        strides: kpe_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut kv_indices_shape = [params.kv_indices.len];
    let mut kv_indices_strides = [params.kv_indices.stride];
    let kv_indices_tensor = DLTensor {
        data: params.kv_indices.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.kv_indices.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: kv_indices_shape.as_mut_ptr(),
        strides: kv_indices_strides.as_mut_ptr(),
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

    let mut kv_last_page_len_shape = [params.kv_last_page_len.len];
    let mut kv_last_page_len_strides = [params.kv_last_page_len.stride];
    let kv_last_page_len_tensor = DLTensor {
        data: params.kv_last_page_len.ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id: params.kv_last_page_len.device_id,
        },
        ndim: 1,
        dtype: dl_dtype_i32(),
        shape: kv_last_page_len_shape.as_mut_ptr(),
        strides: kv_last_page_len_strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let mut args = [any_none(); 9];
    pack_paged_mla_kv_append_args(
        &mut args,
        &append_ckv_tensor,
        &append_kpe_tensor,
        &batch_indices_tensor,
        &positions_tensor,
        &ckv_tensor,
        &kpe_tensor,
        &kv_indices_tensor,
        &kv_indptr_tensor,
        &kv_last_page_len_tensor,
    );

    let mut result = any_none();

    // SAFETY: stream context API contract comes from TVM-FFI and is validated on load.
    let previous_stream =
        unsafe { runtime.set_stream(params.append_ckv.device_id, params.stream)? };
    let mut restore_guard =
        StreamRestoreGuard::new(runtime, params.append_ckv.device_id, previous_stream);

    // SAFETY: argument packing follows TVMFFIAny ABI.
    let call_result = unsafe {
        runtime.call_append_paged_mla_kv_cache(
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

fn pack_paged_kv_append_args(
    args: &mut [TVMFFIAny; 10],
    append_key: &DLTensor,
    append_value: &DLTensor,
    batch_indices: &DLTensor,
    positions: &DLTensor,
    paged_k_cache: &DLTensor,
    paged_v_cache: &DLTensor,
    kv_indices: &DLTensor,
    kv_indptr: &DLTensor,
    kv_last_page_len: &DLTensor,
    kv_layout: MhaQkvLayout,
) {
    *args = [
        any_dltensor_ptr(append_key as *const DLTensor),
        any_dltensor_ptr(append_value as *const DLTensor),
        any_dltensor_ptr(batch_indices as *const DLTensor),
        any_dltensor_ptr(positions as *const DLTensor),
        any_dltensor_ptr(paged_k_cache as *const DLTensor),
        any_dltensor_ptr(paged_v_cache as *const DLTensor),
        any_dltensor_ptr(kv_indices as *const DLTensor),
        any_dltensor_ptr(kv_indptr as *const DLTensor),
        any_dltensor_ptr(kv_last_page_len as *const DLTensor),
        any_i64(kv_layout_code(kv_layout)),
    ];
}

fn pack_paged_mla_kv_append_args(
    args: &mut [TVMFFIAny; 9],
    append_ckv: &DLTensor,
    append_kpe: &DLTensor,
    batch_indices: &DLTensor,
    positions: &DLTensor,
    ckv_cache: &DLTensor,
    kpe_cache: &DLTensor,
    kv_indices: &DLTensor,
    kv_indptr: &DLTensor,
    kv_last_page_len: &DLTensor,
) {
    *args = [
        any_dltensor_ptr(append_ckv as *const DLTensor),
        any_dltensor_ptr(append_kpe as *const DLTensor),
        any_dltensor_ptr(batch_indices as *const DLTensor),
        any_dltensor_ptr(positions as *const DLTensor),
        any_dltensor_ptr(ckv_cache as *const DLTensor),
        any_dltensor_ptr(kpe_cache as *const DLTensor),
        any_dltensor_ptr(kv_indices as *const DLTensor),
        any_dltensor_ptr(kv_indptr as *const DLTensor),
        any_dltensor_ptr(kv_last_page_len as *const DLTensor),
    ];
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn append_paged_kv_cache_cudarc<T, AK, AV, BI, POS, PK, PV, KVI, KVP, KVL>(
    stream: &cudarc::driver::CudaStream,
    append_key: &AK,
    append_value: &AV,
    batch_indices: &BI,
    positions: &POS,
    paged_k_cache: &mut PK,
    paged_v_cache: &mut PV,
    kv_indices: &KVI,
    kv_indptr: &KVP,
    kv_last_page_len: &KVL,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    kv_layout: MhaQkvLayout,
    dtype: DType,
) -> Result<(), FlashInferError>
where
    AK: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    AV: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    BI: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    POS: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    PK: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    PV: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    KVI: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    KVP: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    KVL: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
{
    if num_kv_heads == 0 || head_dim == 0 || page_size == 0 {
        return Err(FlashInferError::invalid_argument(
            "num_kv_heads/head_dim/page_size must be positive",
        ));
    }

    let nnz = batch_indices.len();
    if nnz == 0 {
        return Err(FlashInferError::invalid_argument(
            "batch_indices length must be positive",
        ));
    }
    if positions.len() != nnz {
        return Err(FlashInferError::invalid_argument(
            "positions length must equal batch_indices length",
        ));
    }

    if kv_indptr.len() < 2 {
        return Err(FlashInferError::invalid_argument(
            "kv_indptr length must be at least 2",
        ));
    }

    let batch_size = kv_indptr.len() - 1;
    if kv_last_page_len.len() != batch_size {
        return Err(FlashInferError::invalid_argument(
            "kv_last_page_len length must equal kv_indptr length - 1",
        ));
    }
    if kv_indices.len() == 0 {
        return Err(FlashInferError::invalid_argument(
            "kv_indices length must be positive",
        ));
    }

    let append_expected = nnz
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| FlashInferError::invalid_argument("append tensor size overflow"))?;

    if append_key.len() != append_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "append_key length ({}) must equal nnz * num_kv_heads * head_dim ({append_expected})",
            append_key.len()
        )));
    }
    if append_value.len() != append_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "append_value length ({}) must equal nnz * num_kv_heads * head_dim ({append_expected})",
            append_value.len()
        )));
    }

    let page_unit = page_size
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| FlashInferError::invalid_argument("paged cache unit size overflow"))?;

    if paged_k_cache.len() % page_unit != 0 {
        return Err(FlashInferError::invalid_argument(
            "paged_k_cache length must be divisible by page_size * num_kv_heads * head_dim",
        ));
    }
    if paged_v_cache.len() % page_unit != 0 {
        return Err(FlashInferError::invalid_argument(
            "paged_v_cache length must be divisible by page_size * num_kv_heads * head_dim",
        ));
    }

    let num_pages_k = paged_k_cache.len() / page_unit;
    let num_pages_v = paged_v_cache.len() / page_unit;
    if num_pages_k != num_pages_v {
        return Err(FlashInferError::invalid_argument(
            "paged_k_cache and paged_v_cache must have the same number of pages",
        ));
    }

    let (append_key_ptr, _append_key_sync) = append_key.device_ptr(stream);
    let (append_value_ptr, _append_value_sync) = append_value.device_ptr(stream);
    let (batch_indices_ptr, _batch_indices_sync) = batch_indices.device_ptr(stream);
    let (positions_ptr, _positions_sync) = positions.device_ptr(stream);
    let (paged_k_ptr, _paged_k_sync) = paged_k_cache.device_ptr_mut(stream);
    let (paged_v_ptr, _paged_v_sync) = paged_v_cache.device_ptr_mut(stream);
    let (kv_indices_ptr, _kv_indices_sync) = kv_indices.device_ptr(stream);
    let (kv_indptr_ptr, _kv_indptr_sync) = kv_indptr.device_ptr(stream);
    let (kv_last_page_len_ptr, _kv_last_page_len_sync) = kv_last_page_len.device_ptr(stream);

    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let nnz_i64 = i64::try_from(nnz)
        .map_err(|_| FlashInferError::invalid_argument("nnz does not fit in i64"))?;
    let num_kv_heads_i64 = i64::try_from(num_kv_heads)
        .map_err(|_| FlashInferError::invalid_argument("num_kv_heads does not fit in i64"))?;
    let head_dim_i64 = i64::try_from(head_dim)
        .map_err(|_| FlashInferError::invalid_argument("head_dim does not fit in i64"))?;
    let page_size_i64 = i64::try_from(page_size)
        .map_err(|_| FlashInferError::invalid_argument("page_size does not fit in i64"))?;
    let num_pages_i64 = i64::try_from(num_pages_k)
        .map_err(|_| FlashInferError::invalid_argument("num_pages does not fit in i64"))?;
    let kv_indices_len_i64 = i64::try_from(kv_indices.len())
        .map_err(|_| FlashInferError::invalid_argument("kv_indices length does not fit in i64"))?;
    let kv_indptr_len_i64 = i64::try_from(kv_indptr.len())
        .map_err(|_| FlashInferError::invalid_argument("kv_indptr length does not fit in i64"))?;
    let kv_last_page_len_len_i64 = i64::try_from(kv_last_page_len.len()).map_err(|_| {
        FlashInferError::invalid_argument("kv_last_page_len length does not fit in i64")
    })?;

    let append_stride1 = head_dim_i64;
    let append_stride0 = num_kv_heads_i64
        .checked_mul(head_dim_i64)
        .ok_or_else(|| FlashInferError::invalid_argument("append stride overflow"))?;

    let cache_stride0 = page_size_i64
        .checked_mul(num_kv_heads_i64)
        .and_then(|v| v.checked_mul(head_dim_i64))
        .ok_or_else(|| FlashInferError::invalid_argument("paged cache stride overflow"))?;

    let (cache_dim1, cache_dim2, cache_stride1, cache_stride2) = match kv_layout {
        MhaQkvLayout::Nhd => {
            let s2 = head_dim_i64;
            let s1 = num_kv_heads_i64
                .checked_mul(head_dim_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("paged cache stride overflow"))?;
            (page_size_i64, num_kv_heads_i64, s1, s2)
        }
        MhaQkvLayout::Hnd => {
            let s2 = head_dim_i64;
            let s1 = page_size_i64
                .checked_mul(head_dim_i64)
                .ok_or_else(|| FlashInferError::invalid_argument("paged cache stride overflow"))?;
            (num_kv_heads_i64, page_size_i64, s1, s2)
        }
    };

    let params = PagedKvAppendParams::new(
        MhaTensor3DDesc {
            ptr: append_key_ptr as usize as *const c_void,
            dim0: nnz_i64,
            dim1: num_kv_heads_i64,
            dim2: head_dim_i64,
            stride0: append_stride0,
            stride1: append_stride1,
            stride2: 1,
            dtype,
            device_id,
        },
        MhaTensor3DDesc {
            ptr: append_value_ptr as usize as *const c_void,
            dim0: nnz_i64,
            dim1: num_kv_heads_i64,
            dim2: head_dim_i64,
            stride0: append_stride0,
            stride1: append_stride1,
            stride2: 1,
            dtype,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: batch_indices_ptr as usize as *const c_void,
            len: nnz_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: positions_ptr as usize as *const c_void,
            len: nnz_i64,
            stride: 1,
            device_id,
        },
        MhaTensor4DDesc {
            ptr: paged_k_ptr as usize as *const c_void,
            dim0: num_pages_i64,
            dim1: cache_dim1,
            dim2: cache_dim2,
            dim3: head_dim_i64,
            stride0: cache_stride0,
            stride1: cache_stride1,
            stride2: cache_stride2,
            stride3: 1,
            dtype,
            device_id,
        },
        MhaTensor4DDesc {
            ptr: paged_v_ptr as usize as *const c_void,
            dim0: num_pages_i64,
            dim1: cache_dim1,
            dim2: cache_dim2,
            dim3: head_dim_i64,
            stride0: cache_stride0,
            stride1: cache_stride1,
            stride2: cache_stride2,
            stride3: 1,
            dtype,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: kv_indices_ptr as usize as *const c_void,
            len: kv_indices_len_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: kv_indptr_ptr as usize as *const c_void,
            len: kv_indptr_len_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: kv_last_page_len_ptr as usize as *const c_void,
            len: kv_last_page_len_len_i64,
            stride: 1,
            device_id,
        },
        stream.cu_stream().cast(),
    )
    .with_kv_layout(kv_layout);

    append_paged_kv_cache(&params)
}

#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn append_paged_mla_kv_cache_cudarc<T, AC, AP, BI, POS, CK, KP, KVI, KVP, KVL>(
    stream: &cudarc::driver::CudaStream,
    append_ckv: &AC,
    append_kpe: &AP,
    batch_indices: &BI,
    positions: &POS,
    ckv_cache: &mut CK,
    kpe_cache: &mut KP,
    kv_indices: &KVI,
    kv_indptr: &KVP,
    kv_last_page_len: &KVL,
    page_size: usize,
    dtype: DType,
) -> Result<(), FlashInferError>
where
    AC: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    AP: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtr<T>,
    BI: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    POS: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    CK: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    KP: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    KVI: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    KVP: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
    KVL: cudarc::driver::DeviceSlice<i32> + cudarc::driver::DevicePtr<i32>,
{
    if page_size == 0 {
        return Err(FlashInferError::invalid_argument(
            "page_size must be positive",
        ));
    }

    let nnz = batch_indices.len();
    if nnz == 0 {
        return Err(FlashInferError::invalid_argument(
            "batch_indices length must be positive",
        ));
    }
    if positions.len() != nnz {
        return Err(FlashInferError::invalid_argument(
            "positions length must equal batch_indices length",
        ));
    }

    if kv_indptr.len() < 2 {
        return Err(FlashInferError::invalid_argument(
            "kv_indptr length must be at least 2",
        ));
    }
    if kv_last_page_len.len() != kv_indptr.len() - 1 {
        return Err(FlashInferError::invalid_argument(
            "kv_last_page_len length must equal kv_indptr length - 1",
        ));
    }
    if kv_indices.len() == 0 {
        return Err(FlashInferError::invalid_argument(
            "kv_indices length must be positive",
        ));
    }

    let append_ckv_expected = nnz
        .checked_mul(512)
        .ok_or_else(|| FlashInferError::invalid_argument("append_ckv size overflow"))?;
    let append_kpe_expected = nnz
        .checked_mul(64)
        .ok_or_else(|| FlashInferError::invalid_argument("append_kpe size overflow"))?;

    if append_ckv.len() != append_ckv_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "append_ckv length ({}) must equal nnz * 512 ({append_ckv_expected})",
            append_ckv.len()
        )));
    }
    if append_kpe.len() != append_kpe_expected {
        return Err(FlashInferError::invalid_argument(format!(
            "append_kpe length ({}) must equal nnz * 64 ({append_kpe_expected})",
            append_kpe.len()
        )));
    }

    let ckv_page_unit = page_size
        .checked_mul(512)
        .ok_or_else(|| FlashInferError::invalid_argument("ckv page unit overflow"))?;
    let kpe_page_unit = page_size
        .checked_mul(64)
        .ok_or_else(|| FlashInferError::invalid_argument("kpe page unit overflow"))?;

    if ckv_cache.len() % ckv_page_unit != 0 {
        return Err(FlashInferError::invalid_argument(
            "ckv_cache length must be divisible by page_size * 512",
        ));
    }
    if kpe_cache.len() % kpe_page_unit != 0 {
        return Err(FlashInferError::invalid_argument(
            "kpe_cache length must be divisible by page_size * 64",
        ));
    }

    let num_pages_ckv = ckv_cache.len() / ckv_page_unit;
    let num_pages_kpe = kpe_cache.len() / kpe_page_unit;
    if num_pages_ckv != num_pages_kpe {
        return Err(FlashInferError::invalid_argument(
            "ckv_cache and kpe_cache must have the same number of pages",
        ));
    }

    let (append_ckv_ptr, _append_ckv_sync) = append_ckv.device_ptr(stream);
    let (append_kpe_ptr, _append_kpe_sync) = append_kpe.device_ptr(stream);
    let (batch_indices_ptr, _batch_indices_sync) = batch_indices.device_ptr(stream);
    let (positions_ptr, _positions_sync) = positions.device_ptr(stream);
    let (ckv_cache_ptr, _ckv_cache_sync) = ckv_cache.device_ptr_mut(stream);
    let (kpe_cache_ptr, _kpe_cache_sync) = kpe_cache.device_ptr_mut(stream);
    let (kv_indices_ptr, _kv_indices_sync) = kv_indices.device_ptr(stream);
    let (kv_indptr_ptr, _kv_indptr_sync) = kv_indptr.device_ptr(stream);
    let (kv_last_page_len_ptr, _kv_last_page_len_sync) = kv_last_page_len.device_ptr(stream);

    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit in i32"))?;

    let nnz_i64 = i64::try_from(nnz)
        .map_err(|_| FlashInferError::invalid_argument("nnz does not fit in i64"))?;
    let page_size_i64 = i64::try_from(page_size)
        .map_err(|_| FlashInferError::invalid_argument("page_size does not fit in i64"))?;
    let num_pages_i64 = i64::try_from(num_pages_ckv)
        .map_err(|_| FlashInferError::invalid_argument("num_pages does not fit in i64"))?;
    let kv_indices_len_i64 = i64::try_from(kv_indices.len())
        .map_err(|_| FlashInferError::invalid_argument("kv_indices length does not fit in i64"))?;
    let kv_indptr_len_i64 = i64::try_from(kv_indptr.len())
        .map_err(|_| FlashInferError::invalid_argument("kv_indptr length does not fit in i64"))?;
    let kv_last_page_len_len_i64 = i64::try_from(kv_last_page_len.len()).map_err(|_| {
        FlashInferError::invalid_argument("kv_last_page_len length does not fit in i64")
    })?;

    let ckv_stride0 = page_size_i64
        .checked_mul(512)
        .ok_or_else(|| FlashInferError::invalid_argument("ckv stride overflow"))?;
    let kpe_stride0 = page_size_i64
        .checked_mul(64)
        .ok_or_else(|| FlashInferError::invalid_argument("kpe stride overflow"))?;

    let params = PagedMlaKvAppendParams::new(
        PagedMlaTensor2DDesc {
            ptr: append_ckv_ptr as usize as *const c_void,
            rows: nnz_i64,
            cols: 512,
            stride_row: 512,
            stride_col: 1,
            dtype,
            device_id,
        },
        PagedMlaTensor2DDesc {
            ptr: append_kpe_ptr as usize as *const c_void,
            rows: nnz_i64,
            cols: 64,
            stride_row: 64,
            stride_col: 1,
            dtype,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: batch_indices_ptr as usize as *const c_void,
            len: nnz_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: positions_ptr as usize as *const c_void,
            len: nnz_i64,
            stride: 1,
            device_id,
        },
        MhaTensor3DDesc {
            ptr: ckv_cache_ptr as usize as *const c_void,
            dim0: num_pages_i64,
            dim1: page_size_i64,
            dim2: 512,
            stride0: ckv_stride0,
            stride1: 512,
            stride2: 1,
            dtype,
            device_id,
        },
        MhaTensor3DDesc {
            ptr: kpe_cache_ptr as usize as *const c_void,
            dim0: num_pages_i64,
            dim1: page_size_i64,
            dim2: 64,
            stride0: kpe_stride0,
            stride1: 64,
            stride2: 1,
            dtype,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: kv_indices_ptr as usize as *const c_void,
            len: kv_indices_len_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: kv_indptr_ptr as usize as *const c_void,
            len: kv_indptr_len_i64,
            stride: 1,
            device_id,
        },
        MhaTensor1DI32Desc {
            ptr: kv_last_page_len_ptr as usize as *const c_void,
            len: kv_last_page_len_len_i64,
            stride: 1,
            device_id,
        },
        stream.cu_stream().cast(),
    );

    append_paged_mla_kv_cache(&params)
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

fn decode_paged_layout(desc: MhaTensor4DDesc, layout: MhaQkvLayout) -> (i64, i64, i64, i64) {
    match layout {
        MhaQkvLayout::Nhd => (desc.dim0, desc.dim1, desc.dim2, desc.dim3),
        MhaQkvLayout::Hnd => (desc.dim0, desc.dim2, desc.dim1, desc.dim3),
    }
}

fn kv_layout_code(layout: MhaQkvLayout) -> i64 {
    match layout {
        MhaQkvLayout::Nhd => 0,
        MhaQkvLayout::Hnd => 1,
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

fn check_last_dim_contiguous(name: &str, stride: i64) -> Result<(), FlashInferError> {
    if stride != 1 {
        return Err(FlashInferError::invalid_argument(format!(
            "{name} last-dimension stride must be 1"
        )));
    }
    Ok(())
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

fn dl_dtype_i32() -> DLDataType {
    DLDataType {
        code: KDL_INT,
        bits: 32,
        lanes: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{KTVM_FFI_DL_TENSOR_PTR, KTVM_FFI_INT};

    fn non_null() -> *const c_void {
        std::ptr::NonNull::<u8>::dangling().as_ptr().cast()
    }

    fn valid_paged_kv_params() -> PagedKvAppendParams {
        PagedKvAppendParams::new(
            MhaTensor3DDesc {
                ptr: non_null(),
                dim0: 4,
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
                dim0: 4,
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
                len: 4,
                stride: 1,
                device_id: 0,
            },
            MhaTensor1DI32Desc {
                ptr: non_null(),
                len: 4,
                stride: 1,
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
            std::ptr::null_mut(),
        )
    }

    fn valid_paged_mla_params() -> PagedMlaKvAppendParams {
        PagedMlaKvAppendParams::new(
            PagedMlaTensor2DDesc {
                ptr: non_null(),
                rows: 4,
                cols: 512,
                stride_row: 512,
                stride_col: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            PagedMlaTensor2DDesc {
                ptr: non_null(),
                rows: 4,
                cols: 64,
                stride_row: 64,
                stride_col: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            MhaTensor1DI32Desc {
                ptr: non_null(),
                len: 4,
                stride: 1,
                device_id: 0,
            },
            MhaTensor1DI32Desc {
                ptr: non_null(),
                len: 4,
                stride: 1,
                device_id: 0,
            },
            MhaTensor3DDesc {
                ptr: non_null(),
                dim0: 3,
                dim1: 2,
                dim2: 512,
                stride0: 1024,
                stride1: 512,
                stride2: 1,
                dtype: DType::F16,
                device_id: 0,
            },
            MhaTensor3DDesc {
                ptr: non_null(),
                dim0: 3,
                dim1: 2,
                dim2: 64,
                stride0: 128,
                stride1: 64,
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
            MhaTensor1DI32Desc {
                ptr: non_null(),
                len: 2,
                stride: 1,
                device_id: 0,
            },
            std::ptr::null_mut(),
        )
    }

    fn dummy_tensor() -> DLTensor {
        DLTensor {
            data: std::ptr::null_mut(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: 0,
            },
            ndim: 0,
            dtype: dl_dtype_i32(),
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        }
    }

    #[test]
    fn validate_accepts_minimal_valid_paged_kv_append() {
        let params = valid_paged_kv_params();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn validate_rejects_kv_stride_mismatch() {
        let mut params = valid_paged_kv_params();
        params.paged_v_cache.stride1 = 256;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_layout_shape_mismatch() {
        let params = valid_paged_kv_params().with_kv_layout(MhaQkvLayout::Hnd);
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_indptr_length_mismatch() {
        let mut params = valid_paged_kv_params();
        params.kv_indptr.len = 4;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_dtype_mismatch() {
        let mut params = valid_paged_kv_params();
        params.append_value.dtype = DType::BF16;
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_accepts_minimal_valid_mla_append() {
        let params = valid_paged_mla_params();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn validate_rejects_mla_dim_not_512_64() {
        let mut params = valid_paged_mla_params();
        params.append_ckv.cols = 256;
        assert!(params.validate().is_err());
    }

    #[test]
    fn pack_paged_kv_args_have_expected_count_order_and_types() {
        let t0 = dummy_tensor();
        let t1 = dummy_tensor();
        let t2 = dummy_tensor();
        let t3 = dummy_tensor();
        let t4 = dummy_tensor();
        let t5 = dummy_tensor();
        let t6 = dummy_tensor();
        let t7 = dummy_tensor();
        let t8 = dummy_tensor();

        let mut args = [any_none(); 10];
        pack_paged_kv_append_args(
            &mut args,
            &t0,
            &t1,
            &t2,
            &t3,
            &t4,
            &t5,
            &t6,
            &t7,
            &t8,
            MhaQkvLayout::Nhd,
        );

        assert_eq!(args.len(), 10);
        for arg in args.iter().take(9) {
            assert_eq!(arg.type_index, KTVM_FFI_DL_TENSOR_PTR);
        }
        assert_eq!(args[9].type_index, KTVM_FFI_INT);

        let expected_ptrs = [
            (&t0 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t1 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t2 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t3 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t4 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t5 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t6 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t7 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t8 as *const DLTensor).cast_mut().cast::<c_void>(),
        ];
        for (idx, expected_ptr) in expected_ptrs.iter().enumerate() {
            // SAFETY: we set args[idx] with `any_dltensor_ptr` above.
            let got_ptr = unsafe { args[idx].value.v_ptr };
            assert_eq!(got_ptr, *expected_ptr);
        }
        // SAFETY: final arg is `any_i64(kv_layout_code(...))`.
        assert_eq!(unsafe { args[9].value.v_int64 }, 0);
    }

    #[test]
    fn pack_paged_mla_args_have_expected_count_order_and_types() {
        let t0 = dummy_tensor();
        let t1 = dummy_tensor();
        let t2 = dummy_tensor();
        let t3 = dummy_tensor();
        let t4 = dummy_tensor();
        let t5 = dummy_tensor();
        let t6 = dummy_tensor();
        let t7 = dummy_tensor();
        let t8 = dummy_tensor();

        let mut args = [any_none(); 9];
        pack_paged_mla_kv_append_args(&mut args, &t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8);

        assert_eq!(args.len(), 9);
        for arg in &args {
            assert_eq!(arg.type_index, KTVM_FFI_DL_TENSOR_PTR);
        }

        let expected_ptrs = [
            (&t0 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t1 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t2 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t3 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t4 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t5 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t6 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t7 as *const DLTensor).cast_mut().cast::<c_void>(),
            (&t8 as *const DLTensor).cast_mut().cast::<c_void>(),
        ];
        for (idx, expected_ptr) in expected_ptrs.iter().enumerate() {
            // SAFETY: all args are set by `any_dltensor_ptr` in the packer.
            let got_ptr = unsafe { args[idx].value.v_ptr };
            assert_eq!(got_ptr, *expected_ptr);
        }
    }

    #[test]
    fn pack_paged_kv_layout_code_hnd_is_one() {
        let t0 = dummy_tensor();
        let t1 = dummy_tensor();
        let t2 = dummy_tensor();
        let t3 = dummy_tensor();
        let t4 = dummy_tensor();
        let t5 = dummy_tensor();
        let t6 = dummy_tensor();
        let t7 = dummy_tensor();
        let t8 = dummy_tensor();

        let mut args = [any_none(); 10];
        pack_paged_kv_append_args(
            &mut args,
            &t0,
            &t1,
            &t2,
            &t3,
            &t4,
            &t5,
            &t6,
            &t7,
            &t8,
            MhaQkvLayout::Hnd,
        );

        // SAFETY: final arg is `any_i64(kv_layout_code(...))`.
        assert_eq!(unsafe { args[9].value.v_int64 }, 1);
    }
}
