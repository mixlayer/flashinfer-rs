//! Dense E4M3 GEMM bindings for FlashInfer's CUTLASS kernels.

use std::ffi::c_void;

use crate::error::FlashInferError;
use crate::ffi::{
    DLDataType, DLDevice, DLTensor, KDL_BFLOAT, KDL_CUDA, KDL_FLOAT, KDL_FLOAT8_E4M3FN, KDL_UINT,
    TVMFFIAny, any_dltensor_ptr, any_i64, any_none, any_object_handle,
};
use crate::runtime::FlashInferRuntime;

/// Scalar storage types accepted by dense FP8 GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8GemmDType {
    /// E4M3 finite-only FP8 input storage.
    F8E4M3FN,
    /// IEEE half-precision output storage.
    F16,
    /// Brain floating-point output storage.
    BF16,
    /// IEEE single-precision scalar scale storage.
    F32,
    /// Raw byte workspace storage.
    U8,
}

impl Fp8GemmDType {
    fn as_dl_dtype(self) -> DLDataType {
        let (code, bits) = match self {
            Self::F8E4M3FN => (KDL_FLOAT8_E4M3FN, 8),
            Self::F16 => (KDL_FLOAT, 16),
            Self::BF16 => (KDL_BFLOAT, 16),
            Self::F32 => (KDL_FLOAT, 32),
            Self::U8 => (KDL_UINT, 8),
        };
        DLDataType {
            code,
            bits,
            lanes: 1,
        }
    }
}

/// A contiguous rank-1 CUDA tensor descriptor.
///
/// The allocation must remain live until work on `stream` completes.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Tensor1DDesc {
    /// Device pointer at the first logical element.
    pub ptr: *const c_void,
    /// Element count.
    pub len: i64,
    /// Scalar storage type.
    pub dtype: Fp8GemmDType,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// A contiguous row-major rank-2 CUDA tensor descriptor.
///
/// The allocation must remain live until work on `stream` completes.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Tensor2DDesc {
    /// Device pointer at the first logical element.
    pub ptr: *const c_void,
    /// Number of rows.
    pub rows: i64,
    /// Number of columns.
    pub cols: i64,
    /// Scalar storage type.
    pub dtype: Fp8GemmDType,
    /// CUDA device ordinal.
    pub device_id: i32,
}

/// Arguments for dense row-major `[M,K] x [N,K]^T` E4M3 GEMM.
///
/// Scale tensors use FlashInfer's `MN`-major `(1, 128, 128)` grouping. Calls
/// are asynchronous on `stream`; all inputs, output, and workspace must remain
/// live until that stream completes.
#[derive(Debug, Clone, Copy)]
pub struct Fp8GemmParams {
    /// Contiguous E4M3 activations `[M,K]`.
    pub input: Fp8Tensor2DDesc,
    /// Contiguous E4M3 weights `[N,K]`.
    pub weight: Fp8Tensor2DDesc,
    /// F32 activation scales `[K / 128, M]`, contiguous in `MN`-major order.
    pub input_scale: Fp8Tensor2DDesc,
    /// F32 weight scales `[K / 128, N / 128]`, contiguous in `MN`-major order.
    pub weight_scale: Fp8Tensor2DDesc,
    /// Writable contiguous F16/BF16 output `[M,N]`.
    pub output: Fp8Tensor2DDesc,
    /// Writable U8 scratch allocation.
    pub workspace: Fp8Tensor1DDesc,
    /// CUDA stream used for asynchronous execution.
    pub stream: *mut c_void,
}

impl Fp8GemmParams {
    /// Validate pointer, shape, dtype, and device contracts without launching CUDA.
    pub fn validate(&self) -> Result<(), FlashInferError> {
        validate_ptrs(&[
            (self.input.ptr, "input"),
            (self.weight.ptr, "weight"),
            (self.input_scale.ptr, "input_scale"),
            (self.weight_scale.ptr, "weight_scale"),
            (self.output.ptr, "output"),
            (self.workspace.ptr, "workspace"),
        ])?;
        if self.input.rows <= 0
            || self.input.rows % 4 != 0
            || self.input.cols <= 0
            || self.input.cols % 128 != 0
        {
            return invalid(
                "input must be positive [M,K] E4M3 with M divisible by 4 and K divisible by 128",
            );
        }
        if self.input.dtype != Fp8GemmDType::F8E4M3FN || self.weight.dtype != Fp8GemmDType::F8E4M3FN
        {
            return invalid("input and weight must be F8E4M3FN");
        }
        if self.weight.rows <= 0
            || self.weight.rows % 128 != 0
            || self.weight.cols != self.input.cols
        {
            return invalid(
                "weight must be positive [N,K] with N divisible by 128 and the same K as input",
            );
        }
        let k_groups = self.input.cols / 128;
        if self.input_scale.dtype != Fp8GemmDType::F32
            || self.input_scale.rows != k_groups
            || self.input_scale.cols != self.input.rows
        {
            return invalid("input_scale must be contiguous F32 [K / 128,M]");
        }
        if self.weight_scale.dtype != Fp8GemmDType::F32
            || self.weight_scale.rows != k_groups
            || self.weight_scale.cols != self.weight.rows / 128
        {
            return invalid("weight_scale must be contiguous F32 [K / 128,N / 128]");
        }
        if !matches!(self.output.dtype, Fp8GemmDType::F16 | Fp8GemmDType::BF16)
            || self.output.rows != self.input.rows
            || self.output.cols != self.weight.rows
        {
            return invalid("output must be F16/BF16 [M,N]");
        }
        if self.workspace.dtype != Fp8GemmDType::U8 || self.workspace.len <= 0 {
            return invalid("workspace must be a non-empty U8 allocation");
        }
        validate_same_device(
            self.input.device_id,
            &[
                self.weight.device_id,
                self.input_scale.device_id,
                self.weight_scale.device_id,
                self.output.device_id,
                self.workspace.device_id,
            ],
        )
    }
}

/// Multiply E4M3 activations and weights with FlashInfer's dense CUTLASS kernel.
pub fn fp8_gemm(params: &Fp8GemmParams) -> Result<(), FlashInferError> {
    params.validate()?;
    let runtime = FlashInferRuntime::global()?;
    // SAFETY: validation establishes the typed FlashInfer ABI contract.
    unsafe { gemm_with_runtime(runtime, params) }
}

/// Execute dense FP8 GEMM from flat cudarc buffers.
///
/// Input is `[m,k]`, weight is `[n,k]`, and output is `[m,n]`. Work is
/// enqueued asynchronously on `stream`.
#[cfg(feature = "cudarc")]
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_cudarc<T, I, W, AS, BS, O, X>(
    stream: &cudarc::driver::CudaStream,
    input: &I,
    weight: &W,
    input_scale: &AS,
    weight_scale: &BS,
    output: &mut O,
    workspace: &mut X,
    m: usize,
    n: usize,
    k: usize,
    output_dtype: Fp8GemmDType,
) -> Result<(), FlashInferError>
where
    I: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    W: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtr<u8>,
    AS: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    BS: cudarc::driver::DeviceSlice<f32> + cudarc::driver::DevicePtr<f32>,
    O: cudarc::driver::DeviceSlice<T> + cudarc::driver::DevicePtrMut<T>,
    X: cudarc::driver::DeviceSlice<u8> + cudarc::driver::DevicePtrMut<u8>,
{
    let input_len = m
        .checked_mul(k)
        .ok_or_else(|| FlashInferError::invalid_argument("input size overflow"))?;
    let weight_len = n
        .checked_mul(k)
        .ok_or_else(|| FlashInferError::invalid_argument("weight size overflow"))?;
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| FlashInferError::invalid_argument("output size overflow"))?;
    if input.len() != input_len
        || weight.len() != weight_len
        || input_scale.len() != (k / 128) * m
        || weight_scale.len() != (k / 128) * (n / 128)
        || output.len() != output_len
        || workspace.len() == 0
    {
        return invalid("cudarc FP8 GEMM buffer length mismatch");
    }
    let workspace_len = workspace.len();
    let (input_ptr, _input_sync) = input.device_ptr(stream);
    let (weight_ptr, _weight_sync) = weight.device_ptr(stream);
    let (input_scale_ptr, _input_scale_sync) = input_scale.device_ptr(stream);
    let (weight_scale_ptr, _weight_scale_sync) = weight_scale.device_ptr(stream);
    let (output_ptr, _output_sync) = output.device_ptr_mut(stream);
    let (workspace_ptr, _workspace_sync) = workspace.device_ptr_mut(stream);
    let device_id = i32::try_from(stream.context().ordinal())
        .map_err(|_| FlashInferError::invalid_argument("device id does not fit i32"))?;
    fp8_gemm(&Fp8GemmParams {
        input: desc2(input_ptr, m, k, Fp8GemmDType::F8E4M3FN, device_id)?,
        weight: desc2(weight_ptr, n, k, Fp8GemmDType::F8E4M3FN, device_id)?,
        input_scale: desc2(input_scale_ptr, k / 128, m, Fp8GemmDType::F32, device_id)?,
        weight_scale: desc2(
            weight_scale_ptr,
            k / 128,
            n / 128,
            Fp8GemmDType::F32,
            device_id,
        )?,
        output: desc2(output_ptr, m, n, output_dtype, device_id)?,
        workspace: desc1(workspace_ptr, workspace_len, Fp8GemmDType::U8, device_id)?,
        stream: stream.cu_stream().cast(),
    })
}

#[cfg(feature = "cudarc")]
fn to_i64(value: usize, name: &str) -> Result<i64, FlashInferError> {
    i64::try_from(value)
        .map_err(|_| FlashInferError::invalid_argument(format!("{name} does not fit i64")))
}

#[cfg(feature = "cudarc")]
fn desc1(
    ptr: u64,
    len: usize,
    dtype: Fp8GemmDType,
    device_id: i32,
) -> Result<Fp8Tensor1DDesc, FlashInferError> {
    Ok(Fp8Tensor1DDesc {
        ptr: ptr as usize as *const c_void,
        len: to_i64(len, "descriptor length")?,
        dtype,
        device_id,
    })
}

#[cfg(feature = "cudarc")]
fn desc2(
    ptr: u64,
    rows: usize,
    cols: usize,
    dtype: Fp8GemmDType,
    device_id: i32,
) -> Result<Fp8Tensor2DDesc, FlashInferError> {
    Ok(Fp8Tensor2DDesc {
        ptr: ptr as usize as *const c_void,
        rows: to_i64(rows, "rows")?,
        cols: to_i64(cols, "cols")?,
        dtype,
        device_id,
    })
}

unsafe fn gemm_with_runtime(
    runtime: &FlashInferRuntime,
    params: &Fp8GemmParams,
) -> Result<(), FlashInferError> {
    let (input, mut input_shape, mut input_strides) = tensor2d(params.input);
    let (weight, mut weight_shape, mut weight_strides) = tensor2d(params.weight);
    let (input_scale, mut input_scale_shape, mut input_scale_strides) =
        tensor2d(params.input_scale);
    let (weight_scale, mut weight_scale_shape, mut weight_scale_strides) =
        tensor2d(params.weight_scale);
    let (output, mut output_shape, mut output_strides) = tensor2d(params.output);
    let (workspace, mut workspace_shape, mut workspace_strides) = tensor1d(params.workspace);
    let input = bind_tensor(input, &mut input_shape, &mut input_strides);
    let weight = bind_tensor(weight, &mut weight_shape, &mut weight_strides);
    let input_scale = bind_tensor(
        input_scale,
        &mut input_scale_shape,
        &mut input_scale_strides,
    );
    let weight_scale = bind_tensor(
        weight_scale,
        &mut weight_scale_shape,
        &mut weight_scale_strides,
    );
    let output = bind_tensor(output, &mut output_shape, &mut output_strides);
    let workspace = bind_tensor(workspace, &mut workspace_shape, &mut workspace_strides);
    // SAFETY: TVM-FFI returns an owned string object which is decref'd below.
    let scale_mode = unsafe { runtime.string_to_any("MN")? };
    let scale_mode_object = any_object_handle(&scale_mode);
    let args = [
        any_dltensor_ptr(&workspace),
        any_dltensor_ptr(&input),
        any_dltensor_ptr(&weight),
        any_dltensor_ptr(&input_scale),
        any_dltensor_ptr(&weight_scale),
        any_dltensor_ptr(&output),
        any_i64(1),
        any_i64(128),
        any_i64(128),
        scale_mode,
    ];
    let result = unsafe {
        call_on_stream(runtime, params.input.device_id, params.stream, |result| {
            runtime.call_gemm_fp8_nt_groupwise(args.as_ptr(), args.len() as i32, result)
        })
    };
    if let Some(object) = scale_mode_object {
        // SAFETY: `string_to_any` transferred one owned reference to this scope.
        unsafe { runtime.object_dec_ref(object) };
    }
    result
}

unsafe fn call_on_stream(
    runtime: &FlashInferRuntime,
    device_id: i32,
    stream: *mut c_void,
    call: impl FnOnce(*mut TVMFFIAny) -> Result<(), FlashInferError>,
) -> Result<(), FlashInferError> {
    let previous = unsafe { runtime.set_stream(device_id, stream)? };
    let mut result = any_none();
    let call_result = call(&mut result);
    let restore_result = unsafe { runtime.restore_stream(device_id, previous) };
    match (call_result, restore_result) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn tensor1d(desc: Fp8Tensor1DDesc) -> (DLTensor, [i64; 1], [i64; 1]) {
    (
        base_tensor(desc.ptr, desc.dtype, desc.device_id, 1),
        [desc.len],
        [1],
    )
}

fn tensor2d(desc: Fp8Tensor2DDesc) -> (DLTensor, [i64; 2], [i64; 2]) {
    (
        base_tensor(desc.ptr, desc.dtype, desc.device_id, 2),
        [desc.rows, desc.cols],
        [desc.cols, 1],
    )
}

fn base_tensor(ptr: *const c_void, dtype: Fp8GemmDType, device_id: i32, ndim: i32) -> DLTensor {
    DLTensor {
        data: ptr.cast_mut(),
        device: DLDevice {
            device_type: KDL_CUDA,
            device_id,
        },
        ndim,
        dtype: dtype.as_dl_dtype(),
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    }
}

fn bind_tensor<const N: usize>(
    mut tensor: DLTensor,
    shape: &mut [i64; N],
    strides: &mut [i64; N],
) -> DLTensor {
    tensor.shape = shape.as_mut_ptr();
    tensor.strides = strides.as_mut_ptr();
    tensor
}

fn validate_ptrs(ptrs: &[(*const c_void, &str)]) -> Result<(), FlashInferError> {
    for (ptr, name) in ptrs {
        if ptr.is_null() {
            return invalid(format!("{name} pointer is null"));
        }
    }
    Ok(())
}

fn validate_same_device(reference: i32, others: &[i32]) -> Result<(), FlashInferError> {
    if reference < 0 || others.iter().any(|device| *device != reference) {
        return invalid("all tensors must use the same non-negative CUDA device");
    }
    Ok(())
}

fn invalid<T>(message: impl Into<String>) -> Result<T, FlashInferError> {
    Err(FlashInferError::invalid_argument(message))
}

#[cfg(test)]
mod tests {
    use std::ptr::NonNull;

    use super::{Fp8GemmDType, Fp8GemmParams, Fp8Tensor1DDesc, Fp8Tensor2DDesc};

    fn ptr() -> *const std::ffi::c_void {
        NonNull::<u8>::dangling().as_ptr().cast()
    }

    fn desc1(dtype: Fp8GemmDType, len: i64) -> Fp8Tensor1DDesc {
        Fp8Tensor1DDesc {
            ptr: ptr(),
            len,
            dtype,
            device_id: 0,
        }
    }

    fn desc2(dtype: Fp8GemmDType, rows: i64, cols: i64) -> Fp8Tensor2DDesc {
        Fp8Tensor2DDesc {
            ptr: ptr(),
            rows,
            cols,
            dtype,
            device_id: 0,
        }
    }

    fn valid() -> Fp8GemmParams {
        Fp8GemmParams {
            input: desc2(Fp8GemmDType::F8E4M3FN, 4, 128),
            weight: desc2(Fp8GemmDType::F8E4M3FN, 128, 128),
            input_scale: desc2(Fp8GemmDType::F32, 1, 4),
            weight_scale: desc2(Fp8GemmDType::F32, 1, 1),
            output: desc2(Fp8GemmDType::BF16, 4, 128),
            workspace: desc1(Fp8GemmDType::U8, 4096),
            stream: std::ptr::null_mut(),
        }
    }

    #[test]
    fn validates_dense_fp8_contract() {
        valid().validate().expect("valid FP8 GEMM");
    }

    #[test]
    fn rejects_shape_and_dtype_mismatches() {
        let mut params = valid();
        params.weight.cols = 64;
        assert!(params.validate().is_err());
        let mut params = valid();
        params.output.dtype = Fp8GemmDType::F32;
        assert!(params.validate().is_err());
    }
}
