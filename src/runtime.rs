use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use fs2::FileExt;
use libloading::os::unix::Library;
use sha2::{Digest, Sha256};
use std::os::unix::fs::PermissionsExt;
use zip::ZipArchive;

use crate::error::FlashInferError;
use crate::ffi::{
    DLManagedTensorVersioned, DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION,
    DLPackManagedTensorAllocator, DLPackSetErrorFn, DLPackVersion, KDL_CUDA, TVMFFIAny,
    TVMFFIByteArray, TVMFFIObjectHandle, TVMFFIVersion, any_none, byte_array_to_string,
    error_cell_ptr,
};

include!(concat!(env!("OUT_DIR"), "/embedded_wheels.rs"));

const ENV_CACHE_DIR: &str = "FLASHINFER_RS_CACHE_DIR";
const ENV_SEED_CACHE_DIR: &str = "FLASHINFER_RS_SEED_CACHE_DIR";
const ENV_CUBIN_DIR: &str = "FLASHINFER_CUBIN_DIR";
const ENV_CUBIN_REPOSITORY: &str = "FLASHINFER_CUBINS_REPOSITORY";
const DEFAULT_CUBIN_REPOSITORY: &str =
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local";
const TRTLLM_GEN_MOE_SM100_URI: &str = "fused_moe_trtllm_sm100";

const FLASHINFER_NORM_SO_SUFFIX: &str = "flashinfer_jit_cache/jit_cache/norm/norm.so";
const FLASHINFER_GDN_PREFILL_SM90_SO_SUFFIX: &str =
    "flashinfer_jit_cache/jit_cache/gdn_prefill_sm90/gdn_prefill_sm90.so";
const FLASHINFER_PAGE_SO_SUFFIX: &str = "flashinfer_jit_cache/jit_cache/page/page.so";
const FLASHINFER_SAMPLING_SO_SUFFIX: &str = "flashinfer_jit_cache/jit_cache/sampling/sampling.so";
const FLASHINFER_TRTLLM_COMM_SO_SUFFIX: &str =
    "flashinfer_jit_cache/jit_cache/trtllm_comm/trtllm_comm.so";
const TVMFFI_SO_MEMBER: &str = "tvm_ffi/lib/libtvm_ffi.so";
const WHEEL_CACHE_DIR_NAME: &str = "wheels";
const SHA256SUM_PROGRAM: &str = "sha256sum";

const EXPECTED_TVMFFI_MAJOR: u32 = 0;
const EXPECTED_TVMFFI_MINOR: u32 = 1;

type TVMFFIGetVersionFn = unsafe extern "C" fn(*mut TVMFFIVersion);
type TVMFFIEnvSetStreamFn = unsafe extern "C" fn(i32, i32, *mut c_void, *mut *mut c_void) -> i32;
type TVMFFIEnvSetDLPackManagedTensorAllocatorFn = unsafe extern "C" fn(
    DLPackManagedTensorAllocator,
    i32,
    *mut DLPackManagedTensorAllocator,
) -> i32;
type TVMFFIEnvGetStreamFn = unsafe extern "C" fn(i32, i32) -> *mut c_void;
type TVMFFIErrorMoveFromRaisedFn = unsafe extern "C" fn(*mut TVMFFIObjectHandle);
type TVMFFIObjectDecRefFn = unsafe extern "C" fn(TVMFFIObjectHandle) -> i32;
type TVMFFIFunctionGetGlobalFn =
    unsafe extern "C" fn(*const TVMFFIByteArray, *mut TVMFFIObjectHandle) -> i32;
type TVMFFIFunctionCallFn =
    unsafe extern "C" fn(TVMFFIObjectHandle, *mut TVMFFIAny, i32, *mut TVMFFIAny) -> i32;
type TVMFFIStringFromByteArrayFn =
    unsafe extern "C" fn(*const TVMFFIByteArray, *mut TVMFFIAny) -> i32;
type TVMFFITensorFromDLPackVersionedFn =
    unsafe extern "C" fn(*mut DLManagedTensorVersioned, i32, i32, *mut TVMFFIObjectHandle) -> i32;
type TVMFFITensorToDLPackVersionedFn =
    unsafe extern "C" fn(TVMFFIObjectHandle, *mut *mut DLManagedTensorVersioned) -> i32;
type TVMFFISafeCallFn =
    unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;
type FlashInferCubinCallbackFn = unsafe extern "C" fn(*const c_char, *const c_char);
type FlashInferSetCubinCallbackFn = unsafe extern "C" fn(Option<FlashInferCubinCallbackFn>);
type FlashInferSetCurrentCubinFn = unsafe extern "C" fn(*const c_char, c_int);
type CudaMallocFn = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type CudaFreeFn = unsafe extern "C" fn(*mut c_void) -> i32;
type CudaMallocAsyncFn = unsafe extern "C" fn(*mut *mut c_void, usize, *mut c_void) -> i32;
type CudaFreeAsyncFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type CudaGetDeviceFn = unsafe extern "C" fn(*mut i32) -> i32;
type CudaSetDeviceFn = unsafe extern "C" fn(i32) -> i32;
type CudaGetErrorStringFn = unsafe extern "C" fn(i32) -> *const c_char;
type CudaMemcpyAsyncFn =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, *mut c_void) -> i32;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedRuntimeConfig {
    cache_dir: PathBuf,
    seed_cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CubinLoaderConfig {
    cache_dir: PathBuf,
    repository: String,
}

/// Runtime cache configuration for FlashInfer wheel and artifact loading.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// Writable cache root for downloaded wheels, extracted shared libraries, and default cubins.
    pub cache_dir: Option<PathBuf>,
    /// Optional read-only seed cache root containing pre-baked pinned wheels.
    pub seed_cache_dir: Option<PathBuf>,
}

impl RuntimeConfig {
    /// Reads runtime cache configuration from supported environment variables.
    pub fn from_env() -> Result<Self, FlashInferError> {
        Ok(Self {
            cache_dir: env_path(ENV_CACHE_DIR)?,
            seed_cache_dir: env_path(ENV_SEED_CACHE_DIR)?,
        })
    }

    /// Sets the writable runtime cache root.
    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Sets the optional read-only seed cache root for pre-baked pinned wheels.
    pub fn with_seed_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.seed_cache_dir = Some(path.into());
        self
    }

    fn resolve(&self) -> Result<ResolvedRuntimeConfig, FlashInferError> {
        let env_cfg = RuntimeConfig::from_env()?;

        let cache_dir = if let Some(path) = self.cache_dir.clone().or(env_cfg.cache_dir) {
            path
        } else {
            default_cache_dir()?
        };
        let seed_cache_dir = self.seed_cache_dir.clone().or(env_cfg.seed_cache_dir);

        Ok(ResolvedRuntimeConfig {
            cache_dir,
            seed_cache_dir,
        })
    }
}

/// Ensures the build-selected FlashInfer JIT-cache and TVM-FFI wheels are
/// locally available and SHA-256 valid without initializing CUDA or loading
/// shared libraries.
pub fn prefetch_pinned_wheels(config: RuntimeConfig) -> Result<(), FlashInferError> {
    let resolved = config.resolve()?;
    let _ = ensure_pinned_wheels_cached(&resolved)?;
    Ok(())
}

#[derive(Debug)]
struct ExtractedArtifacts {
    artifact_dir: PathBuf,
    norm_so_path: PathBuf,
    gdn_prefill_sm90_so_path: PathBuf,
    page_so_path: PathBuf,
    sampling_so_path: PathBuf,
    trtllm_comm_so_path: PathBuf,
    tvmffi_so_path: PathBuf,
}

struct MaterializedWheels {
    jit_cache_wheel_path: PathBuf,
    tvmffi_wheel_path: PathBuf,
}

#[derive(Clone, Copy)]
struct PinnedWheelMetadata<'a> {
    logical_name: &'static str,
    filename: &'a str,
    url: &'a str,
    sha256_hex: &'a str,
}

struct LoadedKernel {
    _lib: Library,
    run: TVMFFISafeCallFn,
}

struct LoadedFusedMoeKernel {
    _lib: Library,
    init: TVMFFISafeCallFn,
}

struct LoadedTrtllmGenMoeKernel {
    _lib: Library,
    fp8_block_scale_moe: TVMFFISafeCallFn,
}

#[derive(Clone, Copy)]
struct BatchPrefillKernelFns {
    plan: TVMFFISafeCallFn,
    ragged_run: TVMFFISafeCallFn,
    paged_run: TVMFFISafeCallFn,
}

struct LoadedBatchPrefillKernel {
    _lib: Library,
    fns: BatchPrefillKernelFns,
}

#[derive(Clone, Copy)]
struct BatchDecodeKernelFns {
    plan: TVMFFISafeCallFn,
    run: TVMFFISafeCallFn,
}

struct LoadedBatchDecodeKernel {
    _lib: Library,
    fns: BatchDecodeKernelFns,
}

#[derive(Clone, Copy)]
struct BatchMlaKernelFns {
    plan: TVMFFISafeCallFn,
    run: TVMFFISafeCallFn,
}

struct LoadedBatchMlaKernel {
    _lib: Library,
    fns: BatchMlaKernelFns,
}

#[derive(Clone, Copy)]
struct SamplingKernelFns {
    softmax: TVMFFISafeCallFn,
    sampling_from_probs: TVMFFISafeCallFn,
    sampling_from_logits: TVMFFISafeCallFn,
    top_p_sampling_from_probs: TVMFFISafeCallFn,
    top_k_sampling_from_probs: TVMFFISafeCallFn,
    min_p_sampling_from_probs: TVMFFISafeCallFn,
    top_k_top_p_sampling_from_probs: TVMFFISafeCallFn,
    top_p_renorm_probs: TVMFFISafeCallFn,
    top_k_renorm_probs: TVMFFISafeCallFn,
    top_k_mask_logits: TVMFFISafeCallFn,
    chain_speculative_sampling: TVMFFISafeCallFn,
}

#[derive(Clone, Copy)]
pub(crate) enum SamplingKernel {
    Softmax,
    SamplingFromProbs,
    SamplingFromLogits,
    TopPSamplingFromProbs,
    TopKSamplingFromProbs,
    MinPSamplingFromProbs,
    TopKTopPSamplingFromProbs,
    TopPRenormProbs,
    TopKRenormProbs,
    TopKMaskLogits,
    ChainSpeculativeSampling,
}

pub struct FlashInferRuntime {
    resolved: ResolvedRuntimeConfig,
    jit_cache_wheel_path: PathBuf,
    artifact_dir: PathBuf,
    _tvmffi_lib: Library,
    _norm_lib: Library,
    _gdn_prefill_sm90_lib: Library,
    _page_lib: Library,
    _sampling_lib: Library,
    _trtllm_comm_lib: Library,
    _tvmffi_get_version: TVMFFIGetVersionFn,
    tvmffi_env_set_stream: TVMFFIEnvSetStreamFn,
    tvmffi_error_move_from_raised: TVMFFIErrorMoveFromRaisedFn,
    tvmffi_object_dec_ref: TVMFFIObjectDecRefFn,
    tvmffi_function_get_global: TVMFFIFunctionGetGlobalFn,
    tvmffi_function_call: TVMFFIFunctionCallFn,
    tvmffi_string_from_byte_array: TVMFFIStringFromByteArrayFn,
    tvmffi_tensor_from_dlpack_versioned: TVMFFITensorFromDLPackVersionedFn,
    tvmffi_tensor_to_dlpack_versioned: TVMFFITensorToDLPackVersionedFn,
    tvm_ffi_rmsnorm: TVMFFISafeCallFn,
    tvm_ffi_gemma_rmsnorm: TVMFFISafeCallFn,
    tvm_ffi_gemma_fused_add_rmsnorm: TVMFFISafeCallFn,
    tvm_ffi_gdn_prefill: TVMFFISafeCallFn,
    tvm_ffi_append_paged_kv_cache: TVMFFISafeCallFn,
    tvm_ffi_append_paged_mla_kv_cache: TVMFFISafeCallFn,
    tvm_ffi_trtllm_allreduce_fusion: TVMFFISafeCallFn,
    tvm_ffi_trtllm_lamport_initialize: TVMFFISafeCallFn,
    sampling_fns: SamplingKernelFns,
    single_prefill_kernel_cache: Mutex<HashMap<String, LoadedKernel>>,
    batch_prefill_kernel_cache: Mutex<HashMap<String, LoadedBatchPrefillKernel>>,
    single_decode_kernel_cache: Mutex<HashMap<String, LoadedKernel>>,
    batch_decode_kernel_cache: Mutex<HashMap<String, LoadedBatchDecodeKernel>>,
    batch_mla_kernel_cache: Mutex<HashMap<String, LoadedBatchMlaKernel>>,
    fused_moe_kernel_cache: Mutex<HashMap<String, LoadedFusedMoeKernel>>,
    trtllm_gen_moe_kernel: Mutex<Option<LoadedTrtllmGenMoeKernel>>,
    cubin_loader_config: CubinLoaderConfig,
}

static GLOBAL_RUNTIME: OnceLock<FlashInferRuntime> = OnceLock::new();
static HOST_SHA256SUM: OnceLock<Option<PathBuf>> = OnceLock::new();
static RUNTIME_INIT_LOCK: Mutex<()> = Mutex::new(());
static CUDA_RUNTIME_FNS: OnceLock<Result<CudaRuntimeFns, String>> = OnceLock::new();
static TVM_ENV_GET_STREAM_FN: OnceLock<Option<TVMFFIEnvGetStreamFn>> = OnceLock::new();
static CUBIN_LOADER_CONFIG: OnceLock<CubinLoaderConfig> = OnceLock::new();
static SET_CURRENT_CUBIN_FN: OnceLock<FlashInferSetCurrentCubinFn> = OnceLock::new();
const RUNTIME_ERROR_KIND: &[u8] = b"RuntimeError\0";

thread_local! {
    static LAST_CUBIN_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

struct CudaRuntimeFns {
    _lib: Library,
    malloc: CudaMallocFn,
    free: CudaFreeFn,
    malloc_async: Option<CudaMallocAsyncFn>,
    free_async: Option<CudaFreeAsyncFn>,
    get_device: CudaGetDeviceFn,
    set_device: CudaSetDeviceFn,
    get_error_string: Option<CudaGetErrorStringFn>,
    memcpy_async: CudaMemcpyAsyncFn,
}

struct ManagedTensorContext {
    data: *mut c_void,
    device_id: i32,
    used_async_alloc: bool,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

impl FlashInferRuntime {
    pub fn initialize(config: RuntimeConfig) -> Result<&'static Self, FlashInferError> {
        let resolved = config.resolve()?;

        let _init_guard = RUNTIME_INIT_LOCK
            .lock()
            .map_err(|_| FlashInferError::invalid_argument("runtime lock is poisoned"))?;

        if let Some(runtime) = GLOBAL_RUNTIME.get() {
            if runtime.resolved == resolved {
                return Ok(runtime);
            }
            return Err(FlashInferError::RuntimeAlreadyInitialized);
        }

        // SAFETY: dynamic loading and symbol resolution are encapsulated and validated.
        let runtime = unsafe { Self::load(resolved)? };
        let _ = GLOBAL_RUNTIME.set(runtime);
        GLOBAL_RUNTIME
            .get()
            .ok_or_else(|| FlashInferError::invalid_argument("failed to initialize runtime"))
    }

    pub fn global() -> Result<&'static Self, FlashInferError> {
        Self::initialize(RuntimeConfig::default())
    }

    pub(crate) unsafe fn set_stream(
        &self,
        device_id: i32,
        stream: *mut c_void,
    ) -> Result<*mut c_void, FlashInferError> {
        let mut old_stream: *mut c_void = std::ptr::null_mut();
        // SAFETY: function pointer is resolved from trusted tvm_ffi C ABI.
        let code = unsafe {
            (self.tvmffi_env_set_stream)(KDL_CUDA, device_id, stream, &mut old_stream as *mut _)
        };
        if code != 0 {
            return Err(FlashInferError::StreamSet { device_id, code });
        }
        Ok(old_stream)
    }

    pub(crate) unsafe fn restore_stream(
        &self,
        device_id: i32,
        stream: *mut c_void,
    ) -> Result<(), FlashInferError> {
        // SAFETY: function pointer is resolved from trusted tvm_ffi C ABI.
        let code = unsafe {
            (self.tvmffi_env_set_stream)(KDL_CUDA, device_id, stream, std::ptr::null_mut())
        };
        if code != 0 {
            return Err(FlashInferError::StreamRestore { device_id, code });
        }
        Ok(())
    }

    pub(crate) unsafe fn call_gemma_rmsnorm(
        &self,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code =
            unsafe { (self.tvm_ffi_gemma_rmsnorm)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_gemma_fused_add_rmsnorm(
        &self,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        let code = unsafe {
            (self.tvm_ffi_gemma_fused_add_rmsnorm)(std::ptr::null_mut(), args, num_args, result)
        };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_rmsnorm(
        &self,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { (self.tvm_ffi_rmsnorm)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_gdn_prefill(
        &self,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code =
            unsafe { (self.tvm_ffi_gdn_prefill)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_append_paged_kv_cache(
        &self,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe {
            (self.tvm_ffi_append_paged_kv_cache)(std::ptr::null_mut(), args, num_args, result)
        };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_append_paged_mla_kv_cache(
        &self,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe {
            (self.tvm_ffi_append_paged_mla_kv_cache)(std::ptr::null_mut(), args, num_args, result)
        };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_trtllm_allreduce_fusion(
        &self,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe {
            (self.tvm_ffi_trtllm_allreduce_fusion)(std::ptr::null_mut(), args, num_args, result)
        };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_trtllm_lamport_initialize(
        &self,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe {
            (self.tvm_ffi_trtllm_lamport_initialize)(std::ptr::null_mut(), args, num_args, result)
        };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_sampling(
        &self,
        kernel: SamplingKernel,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        let function = match kernel {
            SamplingKernel::Softmax => self.sampling_fns.softmax,
            SamplingKernel::SamplingFromProbs => self.sampling_fns.sampling_from_probs,
            SamplingKernel::SamplingFromLogits => self.sampling_fns.sampling_from_logits,
            SamplingKernel::TopPSamplingFromProbs => self.sampling_fns.top_p_sampling_from_probs,
            SamplingKernel::TopKSamplingFromProbs => self.sampling_fns.top_k_sampling_from_probs,
            SamplingKernel::MinPSamplingFromProbs => self.sampling_fns.min_p_sampling_from_probs,
            SamplingKernel::TopKTopPSamplingFromProbs => {
                self.sampling_fns.top_k_top_p_sampling_from_probs
            }
            SamplingKernel::TopPRenormProbs => self.sampling_fns.top_p_renorm_probs,
            SamplingKernel::TopKRenormProbs => self.sampling_fns.top_k_renorm_probs,
            SamplingKernel::TopKMaskLogits => self.sampling_fns.top_k_mask_logits,
            SamplingKernel::ChainSpeculativeSampling => {
                self.sampling_fns.chain_speculative_sampling
            }
        };
        // SAFETY: every symbol has TVMFFISafeCallType and arguments are validated by callers.
        let code = unsafe { function(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_single_prefill(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let run = unsafe { self.resolve_single_prefill_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { run(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_batch_prefill_plan(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let fns = unsafe { self.resolve_batch_prefill_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { (fns.plan)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_batch_prefill_ragged(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let fns = unsafe { self.resolve_batch_prefill_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { (fns.ragged_run)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_batch_prefill_paged(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let fns = unsafe { self.resolve_batch_prefill_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { (fns.paged_run)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_single_decode(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let run = unsafe { self.resolve_single_decode_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { run(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_fused_moe_init(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let init = unsafe { self.resolve_fused_moe_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { init(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_trtllm_gen_fp8_block_scale_moe_sm100(
        &self,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let call = unsafe { self.resolve_trtllm_gen_moe_sm100()? };
        clear_last_cubin_error();
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { call(std::ptr::null_mut(), args, num_args, result) };
        let cubin_error = take_last_cubin_error();
        if code != 0 {
            let call_error = self.decode_raised_error(code);
            if let Some(message) = cubin_error {
                return Err(FlashInferError::invalid_argument(format!(
                    "FlashInfer cubin loading failed: {message}; kernel error: {call_error}"
                )));
            }
            return Err(call_error);
        }
        if let Some(message) = cubin_error {
            return Err(FlashInferError::invalid_argument(format!(
                "FlashInfer cubin loading failed: {message}"
            )));
        }
        Ok(())
    }

    pub(crate) unsafe fn call_batch_decode_plan(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let fns = unsafe { self.resolve_batch_decode_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { (fns.plan)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_batch_mla_plan(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let fns = unsafe { self.resolve_batch_mla_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { (fns.plan)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_batch_mla_run(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let fns = unsafe { self.resolve_batch_mla_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { (fns.run)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn call_batch_decode_run(
        &self,
        kernel_uri: &str,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let fns = unsafe { self.resolve_batch_decode_kernel(kernel_uri)? };
        // SAFETY: symbol signature follows TVMFFISafeCallType.
        let code = unsafe { (fns.run)(std::ptr::null_mut(), args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn get_global_function(
        &self,
        name: &str,
    ) -> Result<TVMFFIObjectHandle, FlashInferError> {
        let name_bytes = TVMFFIByteArray {
            data: name.as_ptr().cast::<c_char>(),
            size: name.len(),
        };
        let mut handle: TVMFFIObjectHandle = std::ptr::null_mut();
        // SAFETY: symbol pointer and argument layout match C API.
        let code =
            unsafe { (self.tvmffi_function_get_global)(&name_bytes as *const _, &mut handle) };
        if code != 0 {
            return Err(self.decode_raised_error(code));
        }
        if handle.is_null() {
            return Err(FlashInferError::invalid_argument(format!(
                "TVM-FFI global function `{name}` is not available"
            )));
        }
        Ok(handle)
    }

    pub(crate) unsafe fn call_function(
        &self,
        func: TVMFFIObjectHandle,
        args: *mut TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> Result<(), FlashInferError> {
        // SAFETY: symbol pointer and argument layout match C API.
        let code = unsafe { (self.tvmffi_function_call)(func, args, num_args, result) };
        if code == 0 {
            return Ok(());
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn string_to_any(&self, value: &str) -> Result<TVMFFIAny, FlashInferError> {
        let value_bytes = TVMFFIByteArray {
            data: value.as_ptr().cast::<c_char>(),
            size: value.len(),
        };
        let mut out = any_none();
        // SAFETY: symbol pointer and argument layout match C API.
        let code =
            unsafe { (self.tvmffi_string_from_byte_array)(&value_bytes as *const _, &mut out) };
        if code == 0 {
            return Ok(out);
        }
        Err(self.decode_raised_error(code))
    }

    pub(crate) unsafe fn tensor_from_dlpack_versioned(
        &self,
        from: *mut DLManagedTensorVersioned,
        require_alignment: i32,
        require_contiguous: bool,
    ) -> Result<TVMFFIObjectHandle, FlashInferError> {
        let mut out: TVMFFIObjectHandle = std::ptr::null_mut();
        let require_contiguous_i32 = if require_contiguous { 1 } else { 0 };
        // SAFETY: symbol pointer and argument layout match C API.
        let code = unsafe {
            (self.tvmffi_tensor_from_dlpack_versioned)(
                from,
                require_alignment,
                require_contiguous_i32,
                &mut out as *mut _,
            )
        };
        if code != 0 {
            return Err(self.decode_raised_error(code));
        }
        if out.is_null() {
            return Err(FlashInferError::invalid_argument(
                "TVMFFITensorFromDLPackVersioned returned a null tensor object",
            ));
        }
        Ok(out)
    }

    pub(crate) unsafe fn tensor_to_dlpack_versioned(
        &self,
        tensor: TVMFFIObjectHandle,
    ) -> Result<*mut DLManagedTensorVersioned, FlashInferError> {
        if tensor.is_null() {
            return Err(FlashInferError::invalid_argument(
                "cannot export a null TVM tensor to DLPack",
            ));
        }
        let mut out: *mut DLManagedTensorVersioned = std::ptr::null_mut();
        // SAFETY: tensor is an owned TVM tensor object returned by a safe call.
        let code = unsafe { (self.tvmffi_tensor_to_dlpack_versioned)(tensor, &mut out as *mut _) };
        if code != 0 {
            return Err(self.decode_raised_error(code));
        }
        if out.is_null() {
            return Err(FlashInferError::invalid_argument(
                "TVMFFITensorToDLPackVersioned returned a null managed tensor",
            ));
        }
        Ok(out)
    }

    pub(crate) unsafe fn copy_device_to_device_async(
        &self,
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> Result<(), FlashInferError> {
        let fns = cuda_runtime_fns().map_err(FlashInferError::invalid_argument)?;
        // cudaMemcpyDeviceToDevice is enum value 3 in the CUDA runtime API.
        let code = unsafe { (fns.memcpy_async)(destination, source, bytes, 3, stream) };
        if code == 0 {
            Ok(())
        } else {
            Err(FlashInferError::CudaCopy { code })
        }
    }

    pub(crate) unsafe fn object_dec_ref(&self, obj: TVMFFIObjectHandle) {
        if obj.is_null() {
            return;
        }
        // SAFETY: object handle was created by TVM-FFI APIs and may be decref'd here.
        let _ = unsafe { (self.tvmffi_object_dec_ref)(obj) };
    }

    unsafe fn resolve_single_prefill_kernel(
        &self,
        kernel_uri: &str,
    ) -> Result<TVMFFISafeCallFn, FlashInferError> {
        let mut cache = self.single_prefill_kernel_cache.lock().map_err(|_| {
            FlashInferError::invalid_argument("single prefill cache lock is poisoned")
        })?;

        if let Some(kernel) = cache.get(kernel_uri) {
            return Ok(kernel.run);
        }

        let kernel_path =
            extract_jit_kernel(&self.jit_cache_wheel_path, &self.artifact_dir, kernel_uri)?;

        let kernel_lib =
            unsafe { Library::open(Some(&kernel_path), libc::RTLD_NOW | libc::RTLD_LOCAL) }
                .map_err(|e| FlashInferError::LibraryLoad {
                    library: kernel_path.clone(),
                    message: e.to_string(),
                })?;

        let run: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_run\0",
                "__tvm_ffi_run",
            )?
        };

        cache.insert(
            kernel_uri.to_string(),
            LoadedKernel {
                _lib: kernel_lib,
                run,
            },
        );
        Ok(run)
    }

    unsafe fn resolve_single_decode_kernel(
        &self,
        kernel_uri: &str,
    ) -> Result<TVMFFISafeCallFn, FlashInferError> {
        let mut cache = self.single_decode_kernel_cache.lock().map_err(|_| {
            FlashInferError::invalid_argument("single decode cache lock is poisoned")
        })?;

        if let Some(kernel) = cache.get(kernel_uri) {
            return Ok(kernel.run);
        }

        let kernel_path =
            extract_jit_kernel(&self.jit_cache_wheel_path, &self.artifact_dir, kernel_uri)?;

        let kernel_lib =
            unsafe { Library::open(Some(&kernel_path), libc::RTLD_NOW | libc::RTLD_LOCAL) }
                .map_err(|e| FlashInferError::LibraryLoad {
                    library: kernel_path.clone(),
                    message: e.to_string(),
                })?;

        let run: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_run\0",
                "__tvm_ffi_run",
            )?
        };

        cache.insert(
            kernel_uri.to_string(),
            LoadedKernel {
                _lib: kernel_lib,
                run,
            },
        );
        Ok(run)
    }

    unsafe fn resolve_fused_moe_kernel(
        &self,
        kernel_uri: &str,
    ) -> Result<TVMFFISafeCallFn, FlashInferError> {
        let mut cache = self
            .fused_moe_kernel_cache
            .lock()
            .map_err(|_| FlashInferError::invalid_argument("fused moe cache lock is poisoned"))?;

        if let Some(kernel) = cache.get(kernel_uri) {
            return Ok(kernel.init);
        }

        let kernel_path =
            extract_jit_kernel(&self.jit_cache_wheel_path, &self.artifact_dir, kernel_uri)?;

        let kernel_lib =
            unsafe { Library::open(Some(&kernel_path), libc::RTLD_NOW | libc::RTLD_LOCAL) }
                .map_err(|e| FlashInferError::LibraryLoad {
                    library: kernel_path.clone(),
                    message: e.to_string(),
                })?;

        let init: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_init\0",
                "__tvm_ffi_init",
            )?
        };

        cache.insert(
            kernel_uri.to_string(),
            LoadedFusedMoeKernel {
                _lib: kernel_lib,
                init,
            },
        );
        Ok(init)
    }

    unsafe fn resolve_trtllm_gen_moe_sm100(&self) -> Result<TVMFFISafeCallFn, FlashInferError> {
        let mut loaded = self.trtllm_gen_moe_kernel.lock().map_err(|_| {
            FlashInferError::invalid_argument("TensorRT-LLM Gen MoE cache lock is poisoned")
        })?;

        if let Some(kernel) = loaded.as_ref() {
            return Ok(kernel.fp8_block_scale_moe);
        }

        let kernel_path = extract_jit_kernel(
            &self.jit_cache_wheel_path,
            &self.artifact_dir,
            TRTLLM_GEN_MOE_SM100_URI,
        )?;
        let kernel_lib =
            unsafe { Library::open(Some(&kernel_path), libc::RTLD_NOW | libc::RTLD_LOCAL) }
                .map_err(|e| FlashInferError::LibraryLoad {
                    library: kernel_path.clone(),
                    message: e.to_string(),
                })?;

        let fp8_block_scale_moe = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_trtllm_fp8_block_scale_moe\0",
                "__tvm_ffi_trtllm_fp8_block_scale_moe",
            )?
        };
        let set_cubin_callback: FlashInferSetCubinCallbackFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"FlashInferSetCubinCallback\0",
                "FlashInferSetCubinCallback",
            )?
        };
        let set_current_cubin: FlashInferSetCurrentCubinFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"FlashInferSetCurrentCubin\0",
                "FlashInferSetCurrentCubin",
            )?
        };

        if let Some(existing) = CUBIN_LOADER_CONFIG.get() {
            if existing != &self.cubin_loader_config {
                return Err(FlashInferError::invalid_argument(
                    "FlashInfer cubin loader was initialized with a different configuration",
                ));
            }
        } else {
            let _ = CUBIN_LOADER_CONFIG.set(self.cubin_loader_config.clone());
        }
        if let Some(existing) = SET_CURRENT_CUBIN_FN.get() {
            if *existing as usize != set_current_cubin as usize {
                return Err(FlashInferError::invalid_argument(
                    "a different FlashInfer cubin loader is already active",
                ));
            }
        } else {
            let _ = SET_CURRENT_CUBIN_FN.set(set_current_cubin);
        }

        // SAFETY: callback has the C ABI required by the loaded FlashInfer module and remains
        // process-global for at least as long as the library stored below.
        unsafe { set_cubin_callback(Some(load_cubin_callback)) };

        *loaded = Some(LoadedTrtllmGenMoeKernel {
            _lib: kernel_lib,
            fp8_block_scale_moe,
        });
        Ok(fp8_block_scale_moe)
    }

    unsafe fn resolve_batch_prefill_kernel(
        &self,
        kernel_uri: &str,
    ) -> Result<BatchPrefillKernelFns, FlashInferError> {
        let mut cache = self.batch_prefill_kernel_cache.lock().map_err(|_| {
            FlashInferError::invalid_argument("batch prefill cache lock is poisoned")
        })?;

        if let Some(kernel) = cache.get(kernel_uri) {
            return Ok(kernel.fns);
        }

        let kernel_path =
            extract_jit_kernel(&self.jit_cache_wheel_path, &self.artifact_dir, kernel_uri)?;

        let kernel_lib =
            unsafe { Library::open(Some(&kernel_path), libc::RTLD_NOW | libc::RTLD_LOCAL) }
                .map_err(|e| FlashInferError::LibraryLoad {
                    library: kernel_path.clone(),
                    message: e.to_string(),
                })?;

        let plan: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_plan\0",
                "__tvm_ffi_plan",
            )?
        };
        let ragged_run: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_ragged_run\0",
                "__tvm_ffi_ragged_run",
            )?
        };
        let paged_run: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_paged_run\0",
                "__tvm_ffi_paged_run",
            )?
        };
        let fns = BatchPrefillKernelFns {
            plan,
            ragged_run,
            paged_run,
        };

        cache.insert(
            kernel_uri.to_string(),
            LoadedBatchPrefillKernel {
                _lib: kernel_lib,
                fns,
            },
        );
        Ok(fns)
    }

    unsafe fn resolve_batch_decode_kernel(
        &self,
        kernel_uri: &str,
    ) -> Result<BatchDecodeKernelFns, FlashInferError> {
        let mut cache = self.batch_decode_kernel_cache.lock().map_err(|_| {
            FlashInferError::invalid_argument("batch decode cache lock is poisoned")
        })?;

        if let Some(kernel) = cache.get(kernel_uri) {
            return Ok(kernel.fns);
        }

        let kernel_path =
            extract_jit_kernel(&self.jit_cache_wheel_path, &self.artifact_dir, kernel_uri)?;

        let kernel_lib =
            unsafe { Library::open(Some(&kernel_path), libc::RTLD_NOW | libc::RTLD_LOCAL) }
                .map_err(|e| FlashInferError::LibraryLoad {
                    library: kernel_path.clone(),
                    message: e.to_string(),
                })?;

        let plan: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_plan\0",
                "__tvm_ffi_plan",
            )?
        };
        let run: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_run\0",
                "__tvm_ffi_run",
            )?
        };
        let fns = BatchDecodeKernelFns { plan, run };

        cache.insert(
            kernel_uri.to_string(),
            LoadedBatchDecodeKernel {
                _lib: kernel_lib,
                fns,
            },
        );
        Ok(fns)
    }

    unsafe fn resolve_batch_mla_kernel(
        &self,
        kernel_uri: &str,
    ) -> Result<BatchMlaKernelFns, FlashInferError> {
        let mut cache = self
            .batch_mla_kernel_cache
            .lock()
            .map_err(|_| FlashInferError::invalid_argument("batch MLA cache lock is poisoned"))?;

        if let Some(kernel) = cache.get(kernel_uri) {
            return Ok(kernel.fns);
        }

        let kernel_path =
            extract_jit_kernel(&self.jit_cache_wheel_path, &self.artifact_dir, kernel_uri)?;

        let kernel_lib =
            unsafe { Library::open(Some(&kernel_path), libc::RTLD_NOW | libc::RTLD_LOCAL) }
                .map_err(|e| FlashInferError::LibraryLoad {
                    library: kernel_path.clone(),
                    message: e.to_string(),
                })?;

        let plan: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_plan\0",
                "__tvm_ffi_plan",
            )?
        };
        let run: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &kernel_lib,
                &kernel_path,
                b"__tvm_ffi_run\0",
                "__tvm_ffi_run",
            )?
        };
        let fns = BatchMlaKernelFns { plan, run };

        cache.insert(
            kernel_uri.to_string(),
            LoadedBatchMlaKernel {
                _lib: kernel_lib,
                fns,
            },
        );
        Ok(fns)
    }

    unsafe fn load(resolved: ResolvedRuntimeConfig) -> Result<Self, FlashInferError> {
        let cubin_loader_config = resolve_cubin_loader_config(&resolved)?;
        let materialized_wheels = ensure_pinned_wheels_cached(&resolved)?;
        let artifacts = extract_artifacts(&resolved, &materialized_wheels)?;

        let tvmffi_lib = load_or_reuse_tvmffi(&artifacts.tvmffi_so_path)?;

        let norm_lib = unsafe {
            Library::open(
                Some(&artifacts.norm_so_path),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        }
        .map_err(|e| FlashInferError::LibraryLoad {
            library: artifacts.norm_so_path.clone(),
            message: e.to_string(),
        })?;

        let gdn_prefill_sm90_lib = unsafe {
            Library::open(
                Some(&artifacts.gdn_prefill_sm90_so_path),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        }
        .map_err(|e| FlashInferError::LibraryLoad {
            library: artifacts.gdn_prefill_sm90_so_path.clone(),
            message: e.to_string(),
        })?;

        let page_lib = unsafe {
            Library::open(
                Some(&artifacts.page_so_path),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        }
        .map_err(|e| FlashInferError::LibraryLoad {
            library: artifacts.page_so_path.clone(),
            message: e.to_string(),
        })?;

        let sampling_lib = unsafe {
            Library::open(
                Some(&artifacts.sampling_so_path),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        }
        .map_err(|e| FlashInferError::LibraryLoad {
            library: artifacts.sampling_so_path.clone(),
            message: e.to_string(),
        })?;

        let trtllm_comm_lib = unsafe {
            Library::open(
                Some(&artifacts.trtllm_comm_so_path),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        }
        .map_err(|e| FlashInferError::LibraryLoad {
            library: artifacts.trtllm_comm_so_path.clone(),
            message: e.to_string(),
        })?;

        let tvmffi_get_version: TVMFFIGetVersionFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFIGetVersion\0",
                "TVMFFIGetVersion",
            )?
        };

        let tvmffi_env_set_stream: TVMFFIEnvSetStreamFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFIEnvSetStream\0",
                "TVMFFIEnvSetStream",
            )?
        };

        let tvmffi_env_set_dlpack_managed_tensor_allocator:
            TVMFFIEnvSetDLPackManagedTensorAllocatorFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFIEnvSetDLPackManagedTensorAllocator\0",
                "TVMFFIEnvSetDLPackManagedTensorAllocator",
            )?
        };

        let tvmffi_error_move_from_raised: TVMFFIErrorMoveFromRaisedFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFIErrorMoveFromRaised\0",
                "TVMFFIErrorMoveFromRaised",
            )?
        };

        let tvmffi_object_dec_ref: TVMFFIObjectDecRefFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFIObjectDecRef\0",
                "TVMFFIObjectDecRef",
            )?
        };

        let tvmffi_function_get_global: TVMFFIFunctionGetGlobalFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFIFunctionGetGlobal\0",
                "TVMFFIFunctionGetGlobal",
            )?
        };

        let tvmffi_function_call: TVMFFIFunctionCallFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFIFunctionCall\0",
                "TVMFFIFunctionCall",
            )?
        };

        let tvmffi_string_from_byte_array: TVMFFIStringFromByteArrayFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFIStringFromByteArray\0",
                "TVMFFIStringFromByteArray",
            )?
        };

        let tvmffi_tensor_from_dlpack_versioned: TVMFFITensorFromDLPackVersionedFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFITensorFromDLPackVersioned\0",
                "TVMFFITensorFromDLPackVersioned",
            )?
        };

        let tvmffi_tensor_to_dlpack_versioned: TVMFFITensorToDLPackVersionedFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFITensorToDLPackVersioned\0",
                "TVMFFITensorToDLPackVersioned",
            )?
        };

        let tvm_ffi_gemma_rmsnorm: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &norm_lib,
                &artifacts.norm_so_path,
                b"__tvm_ffi_gemma_rmsnorm\0",
                "__tvm_ffi_gemma_rmsnorm",
            )?
        };

        let tvm_ffi_gemma_fused_add_rmsnorm: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &norm_lib,
                &artifacts.norm_so_path,
                b"__tvm_ffi_gemma_fused_add_rmsnorm\0",
                "__tvm_ffi_gemma_fused_add_rmsnorm",
            )?
        };

        let tvm_ffi_rmsnorm: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &norm_lib,
                &artifacts.norm_so_path,
                b"__tvm_ffi_rmsnorm\0",
                "__tvm_ffi_rmsnorm",
            )?
        };

        let tvm_ffi_gdn_prefill: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &gdn_prefill_sm90_lib,
                &artifacts.gdn_prefill_sm90_so_path,
                b"__tvm_ffi_gdn_prefill\0",
                "__tvm_ffi_gdn_prefill",
            )?
        };

        let tvm_ffi_append_paged_kv_cache: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &page_lib,
                &artifacts.page_so_path,
                b"__tvm_ffi_append_paged_kv_cache\0",
                "__tvm_ffi_append_paged_kv_cache",
            )?
        };

        let tvm_ffi_append_paged_mla_kv_cache: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &page_lib,
                &artifacts.page_so_path,
                b"__tvm_ffi_append_paged_mla_kv_cache\0",
                "__tvm_ffi_append_paged_mla_kv_cache",
            )?
        };

        let tvm_ffi_trtllm_allreduce_fusion: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &trtllm_comm_lib,
                &artifacts.trtllm_comm_so_path,
                b"__tvm_ffi_trtllm_allreduce_fusion\0",
                "__tvm_ffi_trtllm_allreduce_fusion",
            )?
        };

        let tvm_ffi_trtllm_lamport_initialize: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &trtllm_comm_lib,
                &artifacts.trtllm_comm_so_path,
                b"__tvm_ffi_trtllm_lamport_initialize\0",
                "__tvm_ffi_trtllm_lamport_initialize",
            )?
        };

        let sampling_fns = SamplingKernelFns {
            softmax: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_softmax\0",
                    "__tvm_ffi_softmax",
                )?
            },
            sampling_from_probs: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_sampling_from_probs\0",
                    "__tvm_ffi_sampling_from_probs",
                )?
            },
            sampling_from_logits: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_sampling_from_logits\0",
                    "__tvm_ffi_sampling_from_logits",
                )?
            },
            top_p_sampling_from_probs: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_top_p_sampling_from_probs\0",
                    "__tvm_ffi_top_p_sampling_from_probs",
                )?
            },
            top_k_sampling_from_probs: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_top_k_sampling_from_probs\0",
                    "__tvm_ffi_top_k_sampling_from_probs",
                )?
            },
            min_p_sampling_from_probs: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_min_p_sampling_from_probs\0",
                    "__tvm_ffi_min_p_sampling_from_probs",
                )?
            },
            top_k_top_p_sampling_from_probs: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_top_k_top_p_sampling_from_probs\0",
                    "__tvm_ffi_top_k_top_p_sampling_from_probs",
                )?
            },
            top_p_renorm_probs: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_top_p_renorm_probs\0",
                    "__tvm_ffi_top_p_renorm_probs",
                )?
            },
            top_k_renorm_probs: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_top_k_renorm_probs\0",
                    "__tvm_ffi_top_k_renorm_probs",
                )?
            },
            top_k_mask_logits: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_top_k_mask_logits\0",
                    "__tvm_ffi_top_k_mask_logits",
                )?
            },
            chain_speculative_sampling: unsafe {
                resolve_symbol(
                    &sampling_lib,
                    &artifacts.sampling_so_path,
                    b"__tvm_ffi_chain_speculative_sampling\0",
                    "__tvm_ffi_chain_speculative_sampling",
                )?
            },
        };

        let mut version = TVMFFIVersion {
            major: 0,
            minor: 0,
            patch: 0,
        };
        // SAFETY: symbol resolved from c_api ABI and pointer is valid.
        unsafe { tvmffi_get_version(&mut version as *mut _) };

        if version.major != EXPECTED_TVMFFI_MAJOR || version.minor != EXPECTED_TVMFFI_MINOR {
            return Err(FlashInferError::AbiVersionMismatch {
                found_major: version.major,
                found_minor: version.minor,
                found_patch: version.patch,
            });
        }

        // SAFETY: function pointer is resolved from trusted tvm_ffi C ABI.
        let set_allocator_code = unsafe {
            (tvmffi_env_set_dlpack_managed_tensor_allocator)(
                dlpack_managed_tensor_allocator,
                1,
                std::ptr::null_mut(),
            )
        };
        if set_allocator_code != 0 {
            return Err(FlashInferError::DLPackManagedTensorAllocatorSet {
                code: set_allocator_code,
            });
        }

        Ok(Self {
            resolved,
            jit_cache_wheel_path: materialized_wheels.jit_cache_wheel_path,
            artifact_dir: artifacts.artifact_dir,
            _tvmffi_lib: tvmffi_lib,
            _norm_lib: norm_lib,
            _gdn_prefill_sm90_lib: gdn_prefill_sm90_lib,
            _page_lib: page_lib,
            _sampling_lib: sampling_lib,
            _trtllm_comm_lib: trtllm_comm_lib,
            _tvmffi_get_version: tvmffi_get_version,
            tvmffi_env_set_stream,
            tvmffi_error_move_from_raised,
            tvmffi_object_dec_ref,
            tvmffi_function_get_global,
            tvmffi_function_call,
            tvmffi_string_from_byte_array,
            tvmffi_tensor_from_dlpack_versioned,
            tvmffi_tensor_to_dlpack_versioned,
            tvm_ffi_rmsnorm,
            tvm_ffi_gemma_rmsnorm,
            tvm_ffi_gemma_fused_add_rmsnorm,
            tvm_ffi_gdn_prefill,
            tvm_ffi_append_paged_kv_cache,
            tvm_ffi_append_paged_mla_kv_cache,
            tvm_ffi_trtllm_allreduce_fusion,
            tvm_ffi_trtllm_lamport_initialize,
            sampling_fns,
            single_prefill_kernel_cache: Mutex::new(HashMap::new()),
            batch_prefill_kernel_cache: Mutex::new(HashMap::new()),
            single_decode_kernel_cache: Mutex::new(HashMap::new()),
            batch_decode_kernel_cache: Mutex::new(HashMap::new()),
            batch_mla_kernel_cache: Mutex::new(HashMap::new()),
            fused_moe_kernel_cache: Mutex::new(HashMap::new()),
            trtllm_gen_moe_kernel: Mutex::new(None),
            cubin_loader_config,
        })
    }

    fn decode_raised_error(&self, code: i32) -> FlashInferError {
        let mut error_obj: TVMFFIObjectHandle = std::ptr::null_mut();

        // SAFETY: symbol resolved from c_api ABI and output pointer is valid.
        unsafe {
            (self.tvmffi_error_move_from_raised)(&mut error_obj as *mut _);
        }

        if error_obj.is_null() {
            return FlashInferError::TvmFfiCallNoDetails { code };
        }

        // SAFETY: runtime sets raised error to ffi.Error object.
        let (kind, message, backtrace) = unsafe {
            let cell = &*error_cell_ptr(error_obj);
            (
                byte_array_to_string(cell.kind),
                byte_array_to_string(cell.message),
                byte_array_to_string(cell.backtrace),
            )
        };

        // SAFETY: object came from TVMFFIErrorMoveFromRaised and should be decref'd by caller.
        unsafe {
            let _ = (self.tvmffi_object_dec_ref)(error_obj);
        }

        FlashInferError::tvm_ffi_call(code, kind, message, backtrace)
    }
}

unsafe fn resolve_symbol<T: Copy>(
    lib: &Library,
    library: &Path,
    symbol_bytes: &'static [u8],
    symbol_name: &'static str,
) -> Result<T, FlashInferError> {
    // SAFETY: caller provides the concrete symbol type and this function only copies fn ptr values.
    let symbol =
        unsafe { lib.get::<T>(symbol_bytes) }.map_err(|e| FlashInferError::SymbolResolve {
            library: library.to_path_buf(),
            symbol: symbol_name,
            message: e.to_string(),
        })?;
    Ok(*symbol)
}

fn load_or_reuse_tvmffi(path: &Path) -> Result<Library, FlashInferError> {
    let process = Library::this();
    // Reuse a TVM FFI runtime that another kernel stack intentionally made
    // process-global. ABI compatibility is checked after all required symbols
    // are resolved, just as it is for the pinned runtime below.
    if unsafe {
        process
            .get::<TVMFFIGetVersionFn>(b"TVMFFIGetVersion\0")
            .is_ok()
    } {
        return Ok(process);
    }

    // SAFETY: the path names the checksum-validated pinned TVM FFI runtime and
    // its handle remains owned by FlashInferRuntime for the process lifetime.
    unsafe { Library::open(Some(path), libc::RTLD_NOW | libc::RTLD_GLOBAL) }.map_err(|error| {
        FlashInferError::LibraryLoad {
            library: path.to_path_buf(),
            message: error.to_string(),
        }
    })
}

fn cuda_runtime_fns() -> Result<&'static CudaRuntimeFns, String> {
    CUDA_RUNTIME_FNS
        .get_or_init(|| {
            // SAFETY: symbol resolution only reads process symbol table.
            unsafe { load_cuda_runtime_fns() }
        })
        .as_ref()
        .map_err(Clone::clone)
}

unsafe fn load_cuda_runtime_fns() -> Result<CudaRuntimeFns, String> {
    // Keep an explicit handle because cudarc uses the driver API and does not make CUDA Runtime
    // symbols globally visible. FlashInfer's shared objects are deliberately loaded RTLD_LOCAL.
    let runtime_lib = unsafe {
        Library::open(
            Some(Path::new("libcudart.so.13")),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        )
    }
    .map_err(|error| format!("failed to load libcudart.so.13: {error}"))?;
    // SAFETY: symbol signatures match CUDA Runtime API.
    let malloc = unsafe {
        resolve_runtime_symbol::<CudaMallocFn>(&runtime_lib, "cudaMalloc", b"cudaMalloc\0")?
    };
    // SAFETY: symbol signatures match CUDA Runtime API.
    let free =
        unsafe { resolve_runtime_symbol::<CudaFreeFn>(&runtime_lib, "cudaFree", b"cudaFree\0")? };
    // SAFETY: symbol signatures match CUDA Runtime API.
    let get_device = unsafe {
        resolve_runtime_symbol::<CudaGetDeviceFn>(
            &runtime_lib,
            "cudaGetDevice",
            b"cudaGetDevice\0",
        )?
    };
    // SAFETY: symbol signatures match CUDA Runtime API.
    let set_device = unsafe {
        resolve_runtime_symbol::<CudaSetDeviceFn>(
            &runtime_lib,
            "cudaSetDevice",
            b"cudaSetDevice\0",
        )?
    };
    // SAFETY: optional symbol; null means we fall back to `cudaMalloc`.
    let malloc_async = unsafe {
        runtime_lib
            .get::<CudaMallocAsyncFn>(b"cudaMallocAsync\0")
            .ok()
            .map(|symbol| *symbol)
    };
    // SAFETY: optional symbol; null means we fall back to `cudaFree`.
    let free_async = unsafe {
        runtime_lib
            .get::<CudaFreeAsyncFn>(b"cudaFreeAsync\0")
            .ok()
            .map(|symbol| *symbol)
    };
    // SAFETY: optional symbol; null means we fall back to numeric code messages.
    let get_error_string = unsafe {
        runtime_lib
            .get::<CudaGetErrorStringFn>(b"cudaGetErrorString\0")
            .ok()
            .map(|symbol| *symbol)
    };
    // SAFETY: symbol signature matches the CUDA Runtime API.
    let memcpy_async = unsafe {
        resolve_runtime_symbol::<CudaMemcpyAsyncFn>(
            &runtime_lib,
            "cudaMemcpyAsync",
            b"cudaMemcpyAsync\0",
        )?
    };

    Ok(CudaRuntimeFns {
        _lib: runtime_lib,
        malloc,
        free,
        malloc_async,
        free_async,
        get_device,
        set_device,
        get_error_string,
        memcpy_async,
    })
}

unsafe fn resolve_runtime_symbol<T: Copy>(
    runtime_lib: &Library,
    symbol_name: &'static str,
    symbol_bytes: &'static [u8],
) -> Result<T, String> {
    // SAFETY: caller provides the matching CUDA Runtime function-pointer type.
    unsafe { runtime_lib.get::<T>(symbol_bytes) }
        .map(|symbol| *symbol)
        .map_err(|error| format!("failed to resolve `{symbol_name}` from libcudart.so.13: {error}"))
}

fn cuda_error_message(fns: &CudaRuntimeFns, code: i32) -> String {
    let mut message = format!("CUDA error code {code}");
    if let Some(get_error_string) = fns.get_error_string {
        // SAFETY: CUDA runtime returns a static C string for valid error codes.
        let ptr = unsafe { get_error_string(code) };
        if !ptr.is_null() {
            // SAFETY: CUDA runtime provides a null-terminated string.
            let detail = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
            message = format!("{message} ({detail})");
        }
    }
    message
}

fn current_tvm_stream(device_id: i32) -> *mut c_void {
    let get_stream = TVM_ENV_GET_STREAM_FN.get_or_init(|| {
        // SAFETY: direct process-wide symbol lookup for optional tvm_ffi helper.
        unsafe {
            let ptr = libc::dlsym(libc::RTLD_DEFAULT, b"TVMFFIEnvGetStream\0".as_ptr().cast());
            if ptr.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut c_void, TVMFFIEnvGetStreamFn>(
                    ptr,
                ))
            }
        }
    });
    match get_stream {
        Some(get_stream) => {
            // SAFETY: function pointer is resolved from trusted tvm_ffi C ABI.
            unsafe { get_stream(KDL_CUDA, device_id) }
        }
        None => std::ptr::null_mut(),
    }
}

unsafe extern "C" fn dlpack_managed_tensor_allocator(
    prototype: *mut crate::ffi::DLTensor,
    out: *mut *mut DLManagedTensorVersioned,
    error_ctx: *mut c_void,
    set_error: DLPackSetErrorFn,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        // SAFETY: pointer validation is handled inside `allocate_managed_tensor`.
        unsafe { allocate_managed_tensor(prototype, out) }
    });

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(message)) => {
            // SAFETY: callback contract allows setting an error on failure.
            unsafe { report_allocator_error(set_error, error_ctx, &message) };
            -1
        }
        Err(_) => {
            // SAFETY: callback contract allows setting an error on failure.
            unsafe {
                report_allocator_error(
                    set_error,
                    error_ctx,
                    "panic while allocating DLPack managed tensor",
                )
            };
            -1
        }
    }
}

unsafe fn allocate_managed_tensor(
    prototype: *mut crate::ffi::DLTensor,
    out: *mut *mut DLManagedTensorVersioned,
) -> Result<(), String> {
    if prototype.is_null() {
        return Err("allocator received null prototype tensor".to_string());
    }
    if out.is_null() {
        return Err("allocator received null output pointer".to_string());
    }

    // SAFETY: caller guarantees `prototype` is valid for the duration of the call.
    let prototype = unsafe { &*prototype };
    if prototype.device.device_type != KDL_CUDA {
        return Err(format!(
            "unsupported device_type {} in DLPack allocator; expected kDLCUDA ({KDL_CUDA})",
            prototype.device.device_type
        ));
    }

    let ndim = usize::try_from(prototype.ndim)
        .map_err(|_| format!("negative ndim {} in prototype tensor", prototype.ndim))?;

    let shape = if ndim == 0 {
        Vec::new()
    } else {
        if prototype.shape.is_null() {
            return Err("prototype shape pointer is null for ndim > 0".to_string());
        }
        // SAFETY: shape pointer is expected to have `ndim` elements.
        let shape_slice = unsafe { std::slice::from_raw_parts(prototype.shape, ndim) };
        let mut shape = Vec::with_capacity(ndim);
        for (idx, dim) in shape_slice.iter().copied().enumerate() {
            if dim < 0 {
                return Err(format!("negative shape dim at index {idx}: {dim}"));
            }
            shape.push(dim);
        }
        shape
    };

    let element_bits = usize::from(prototype.dtype.bits)
        .checked_mul(usize::from(prototype.dtype.lanes))
        .ok_or_else(|| "dtype bit-size overflow".to_string())?;
    if element_bits == 0 {
        return Err("dtype bit-size is zero".to_string());
    }
    let element_bytes = (element_bits + 7) / 8;

    let num_elements = if shape.is_empty() {
        1_usize
    } else {
        shape.iter().try_fold(1_usize, |acc, &dim| {
            let dim_usize = usize::try_from(dim)
                .map_err(|_| format!("shape dim {dim} does not fit in usize"))?;
            acc.checked_mul(dim_usize)
                .ok_or_else(|| "tensor element-count overflow".to_string())
        })?
    };
    let num_bytes = num_elements
        .checked_mul(element_bytes)
        .ok_or_else(|| "tensor byte-size overflow".to_string())?;

    let strides = compute_contiguous_strides(&shape)?;
    let (data, used_async_alloc) = if num_bytes == 0 {
        (std::ptr::null_mut(), false)
    } else {
        let fns = cuda_runtime_fns()?;
        allocate_cuda_buffer_on_device(fns, prototype.device.device_id, num_bytes)?
    };

    let mut ctx = Box::new(ManagedTensorContext {
        data,
        device_id: prototype.device.device_id,
        used_async_alloc,
        shape: shape.into_boxed_slice(),
        strides: strides.into_boxed_slice(),
    });
    let shape_ptr = if ndim == 0 {
        std::ptr::null_mut()
    } else {
        ctx.shape.as_mut_ptr()
    };
    let strides_ptr = if ndim == 0 {
        std::ptr::null_mut()
    } else {
        ctx.strides.as_mut_ptr()
    };
    let ctx_ptr = Box::into_raw(ctx);

    let managed_tensor = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion {
            major: DLPACK_MAJOR_VERSION,
            minor: DLPACK_MINOR_VERSION,
        },
        manager_ctx: ctx_ptr.cast(),
        deleter: Some(dlpack_managed_tensor_deleter),
        flags: 0,
        dl_tensor: crate::ffi::DLTensor {
            data,
            device: prototype.device,
            ndim: prototype.ndim,
            dtype: prototype.dtype,
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: 0,
        },
    });

    // SAFETY: `out` is validated non-null and points to caller-owned storage.
    unsafe { *out = Box::into_raw(managed_tensor) };
    Ok(())
}

fn compute_contiguous_strides(shape: &[i64]) -> Result<Vec<i64>, String> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }

    let mut strides = vec![0_i64; shape.len()];
    strides[shape.len() - 1] = 1;
    for idx in (0..shape.len() - 1).rev() {
        strides[idx] = strides[idx + 1]
            .checked_mul(shape[idx + 1])
            .ok_or_else(|| "stride overflow while computing contiguous layout".to_string())?;
    }
    Ok(strides)
}

fn allocate_cuda_buffer_on_device(
    fns: &CudaRuntimeFns,
    device_id: i32,
    num_bytes: usize,
) -> Result<(*mut c_void, bool), String> {
    let mut previous_device = 0_i32;
    // SAFETY: CUDA runtime symbol signatures are validated during resolution.
    let get_device_code = unsafe { (fns.get_device)(&mut previous_device as *mut _) };
    if get_device_code != 0 {
        return Err(format!(
            "cudaGetDevice failed: {}",
            cuda_error_message(fns, get_device_code)
        ));
    }

    let switched = previous_device != device_id;
    if switched {
        // SAFETY: CUDA runtime symbol signatures are validated during resolution.
        let set_device_code = unsafe { (fns.set_device)(device_id) };
        if set_device_code != 0 {
            return Err(format!(
                "cudaSetDevice({device_id}) failed: {}",
                cuda_error_message(fns, set_device_code)
            ));
        }
    }

    let stream = current_tvm_stream(device_id);
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let (malloc_code, used_async_alloc) = if let Some(malloc_async) = fns.malloc_async {
        // SAFETY: CUDA runtime symbol signatures are validated during resolution.
        let code = unsafe { malloc_async(&mut ptr as *mut _, num_bytes, stream) };
        (code, true)
    } else {
        // SAFETY: CUDA runtime symbol signatures are validated during resolution.
        let code = unsafe { (fns.malloc)(&mut ptr as *mut _, num_bytes) };
        (code, false)
    };

    let mut restore_error = None;
    if switched {
        // SAFETY: CUDA runtime symbol signatures are validated during resolution.
        let restore_code = unsafe { (fns.set_device)(previous_device) };
        if restore_code != 0 {
            restore_error = Some(format!(
                "cudaSetDevice({previous_device}) restore failed: {}",
                cuda_error_message(fns, restore_code)
            ));
        }
    }

    if malloc_code != 0 {
        let alloc_kind = if used_async_alloc {
            "cudaMallocAsync"
        } else {
            "cudaMalloc"
        };
        return Err(format!(
            "{alloc_kind}({num_bytes}) failed: {}",
            cuda_error_message(fns, malloc_code)
        ));
    }
    if let Some(restore_error) = restore_error {
        // SAFETY: pointer was allocated by `cudaMalloc` above and must be released.
        let _ = unsafe { (fns.free)(ptr) };
        return Err(restore_error);
    }
    Ok((ptr, used_async_alloc))
}

fn free_cuda_buffer_on_device(
    fns: &CudaRuntimeFns,
    device_id: i32,
    data: *mut c_void,
    used_async_alloc: bool,
) -> Result<(), String> {
    let mut previous_device = 0_i32;
    // SAFETY: CUDA runtime symbol signatures are validated during resolution.
    let get_device_code = unsafe { (fns.get_device)(&mut previous_device as *mut _) };
    if get_device_code != 0 {
        return Err(format!(
            "cudaGetDevice failed during free: {}",
            cuda_error_message(fns, get_device_code)
        ));
    }

    let switched = previous_device != device_id;
    if switched {
        // SAFETY: CUDA runtime symbol signatures are validated during resolution.
        let set_device_code = unsafe { (fns.set_device)(device_id) };
        if set_device_code != 0 {
            return Err(format!(
                "cudaSetDevice({device_id}) failed during free: {}",
                cuda_error_message(fns, set_device_code)
            ));
        }
    }

    let stream = current_tvm_stream(device_id);
    let (free_code, free_kind) = if used_async_alloc {
        if let Some(free_async) = fns.free_async {
            // SAFETY: pointer originated from `cudaMallocAsync`.
            (unsafe { free_async(data, stream) }, "cudaFreeAsync")
        } else {
            // SAFETY: fallback path for runtimes where async free is unavailable.
            (unsafe { (fns.free)(data) }, "cudaFree")
        }
    } else {
        // SAFETY: pointer originated from `cudaMalloc`.
        (unsafe { (fns.free)(data) }, "cudaFree")
    };

    let mut restore_error = None;
    if switched {
        // SAFETY: CUDA runtime symbol signatures are validated during resolution.
        let restore_code = unsafe { (fns.set_device)(previous_device) };
        if restore_code != 0 {
            restore_error = Some(format!(
                "cudaSetDevice({previous_device}) restore failed after free: {}",
                cuda_error_message(fns, restore_code)
            ));
        }
    }

    if free_code != 0 {
        return Err(format!(
            "{free_kind} failed: {}",
            cuda_error_message(fns, free_code)
        ));
    }
    if let Some(restore_error) = restore_error {
        return Err(restore_error);
    }
    Ok(())
}

unsafe extern "C" fn dlpack_managed_tensor_deleter(managed_tensor: *mut DLManagedTensorVersioned) {
    if managed_tensor.is_null() {
        return;
    }

    // SAFETY: deleter receives ownership of `managed_tensor`.
    let managed_tensor = unsafe { Box::from_raw(managed_tensor) };
    let ctx_ptr = managed_tensor.manager_ctx.cast::<ManagedTensorContext>();
    if ctx_ptr.is_null() {
        return;
    }

    // SAFETY: manager_ctx is owned by this managed tensor.
    let ctx = unsafe { Box::from_raw(ctx_ptr) };
    if ctx.data.is_null() {
        return;
    }

    if let Ok(fns) = cuda_runtime_fns() {
        let _ = free_cuda_buffer_on_device(fns, ctx.device_id, ctx.data, ctx.used_async_alloc);
    }
}

unsafe fn report_allocator_error(
    set_error: DLPackSetErrorFn,
    error_ctx: *mut c_void,
    message: &str,
) {
    let Some(set_error) = set_error else {
        return;
    };

    let sanitized = message.replace('\0', " ");
    let message_cstr = match CString::new(sanitized) {
        Ok(msg) => msg,
        Err(_) => match CString::new("allocator error") {
            Ok(msg) => msg,
            Err(_) => return,
        },
    };

    // SAFETY: callback contract allows producer to report a string error.
    unsafe {
        set_error(
            error_ctx,
            RUNTIME_ERROR_KIND.as_ptr().cast(),
            message_cstr.as_ptr(),
        )
    };
}

fn resolve_cubin_loader_config(
    resolved: &ResolvedRuntimeConfig,
) -> Result<CubinLoaderConfig, FlashInferError> {
    let cache_dir = env_path(ENV_CUBIN_DIR)?.unwrap_or_else(|| resolved.cache_dir.join("cubins"));
    let repository = match env::var(ENV_CUBIN_REPOSITORY) {
        Ok(value) if !value.trim().is_empty() => value,
        Ok(_) => {
            return Err(FlashInferError::InvalidEnvironment {
                name: ENV_CUBIN_REPOSITORY,
                message: "value must not be empty".to_string(),
            });
        }
        Err(env::VarError::NotPresent) => DEFAULT_CUBIN_REPOSITORY.to_string(),
        Err(env::VarError::NotUnicode(_)) => {
            return Err(FlashInferError::InvalidEnvironment {
                name: ENV_CUBIN_REPOSITORY,
                message: "value is not valid Unicode".to_string(),
            });
        }
    };
    Ok(CubinLoaderConfig {
        cache_dir,
        repository,
    })
}

fn clear_last_cubin_error() {
    LAST_CUBIN_ERROR.with(|slot| *slot.borrow_mut() = None);
}

fn take_last_cubin_error() -> Option<String> {
    LAST_CUBIN_ERROR.with(|slot| slot.borrow_mut().take())
}

fn record_last_cubin_error(message: String) {
    LAST_CUBIN_ERROR.with(|slot| *slot.borrow_mut() = Some(message));
}

unsafe extern "C" fn load_cubin_callback(name: *const c_char, sha256: *const c_char) {
    const EMPTY_CUBIN: &[u8] = b"\0";
    let result = std::panic::catch_unwind(|| {
        if name.is_null() || sha256.is_null() {
            return Err(FlashInferError::invalid_argument(
                "FlashInfer requested a cubin with a null name or checksum",
            ));
        }
        // SAFETY: FlashInfer passes null-terminated strings for the duration of the callback.
        let name = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| FlashInferError::invalid_argument("cubin name is not valid UTF-8"))?;
        // SAFETY: FlashInfer passes null-terminated strings for the duration of the callback.
        let sha256 = unsafe { CStr::from_ptr(sha256) }
            .to_str()
            .map_err(|_| FlashInferError::invalid_argument("cubin checksum is not valid UTF-8"))?;
        materialize_cubin(name, sha256)
    });

    let set_current = SET_CURRENT_CUBIN_FN.get().copied();
    match result {
        Ok(Ok(bytes)) => {
            let Ok(size) = c_int::try_from(bytes.len()) else {
                record_last_cubin_error("FlashInfer cubin is too large for the loader ABI".into());
                if let Some(set_current) = set_current {
                    // SAFETY: the module copies the provided bytes before this call returns.
                    unsafe { set_current(EMPTY_CUBIN.as_ptr().cast(), 0) };
                }
                return;
            };
            if let Some(set_current) = set_current {
                // SAFETY: the module copies the provided bytes before this call returns.
                unsafe { set_current(bytes.as_ptr().cast(), size) };
            } else {
                record_last_cubin_error("FlashInfer cubin setter is not initialized".into());
            }
        }
        Ok(Err(error)) => {
            record_last_cubin_error(error.to_string());
            if let Some(set_current) = set_current {
                // SAFETY: an empty cubin signals callback failure to the module.
                unsafe { set_current(EMPTY_CUBIN.as_ptr().cast(), 0) };
            }
        }
        Err(_) => {
            record_last_cubin_error("panic while loading a FlashInfer cubin".into());
            if let Some(set_current) = set_current {
                // SAFETY: an empty cubin signals callback failure to the module.
                unsafe { set_current(EMPTY_CUBIN.as_ptr().cast(), 0) };
            }
        }
    }
}

fn materialize_cubin(name: &str, expected_sha256: &str) -> Result<Vec<u8>, FlashInferError> {
    if expected_sha256.len() != 64 || !expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(FlashInferError::invalid_argument(format!(
            "invalid SHA-256 supplied for FlashInfer cubin `{name}`"
        )));
    }
    let relative = safe_cubin_relative_path(name)?;
    let config = CUBIN_LOADER_CONFIG.get().ok_or_else(|| {
        FlashInferError::invalid_argument("FlashInfer cubin loader is not initialized")
    })?;
    let path = config.cache_dir.join(relative);

    if let Ok(bytes) = fs::read(&path) {
        let found = format!("{:x}", Sha256::digest(&bytes));
        if found.eq_ignore_ascii_case(expected_sha256) {
            return Ok(bytes);
        }
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| FlashInferError::CubinCache {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut lock_name = path.as_os_str().to_os_string();
    lock_name.push(".lock");
    let lock_path = PathBuf::from(lock_name);
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|source| FlashInferError::CubinCache {
            path: lock_path.clone(),
            source,
        })?;
    lock.lock_exclusive()
        .map_err(|source| FlashInferError::CubinCache {
            path: lock_path,
            source,
        })?;

    if let Ok(bytes) = fs::read(&path) {
        let found = format!("{:x}", Sha256::digest(&bytes));
        if found.eq_ignore_ascii_case(expected_sha256) {
            let _ = lock.unlock();
            return Ok(bytes);
        }
    }

    let url = format!(
        "{}/{}",
        config.repository.trim_end_matches('/'),
        name.trim_start_matches('/')
    );
    let response = ureq::get(&url)
        .call()
        .map_err(|error| FlashInferError::CubinDownload {
            url: url.clone(),
            message: error.to_string(),
        })?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    {
        let mut output =
            File::create(&temporary).map_err(|source| FlashInferError::CubinCache {
                path: temporary.clone(),
                source,
            })?;
        io::copy(&mut response.into_reader(), &mut output).map_err(|source| {
            FlashInferError::CubinCache {
                path: temporary.clone(),
                source,
            }
        })?;
        output
            .sync_all()
            .map_err(|source| FlashInferError::CubinCache {
                path: temporary.clone(),
                source,
            })?;
    }
    let bytes = fs::read(&temporary).map_err(|source| FlashInferError::CubinCache {
        path: temporary.clone(),
        source,
    })?;
    let found = format!("{:x}", Sha256::digest(&bytes));
    if !found.eq_ignore_ascii_case(expected_sha256) {
        let _ = fs::remove_file(&temporary);
        let _ = lock.unlock();
        return Err(FlashInferError::CubinChecksumMismatch {
            path,
            expected: expected_sha256.to_ascii_lowercase(),
            found,
        });
    }
    fs::rename(&temporary, &path).map_err(|source| FlashInferError::CubinCache {
        path: path.clone(),
        source,
    })?;
    let _ = lock.unlock();
    Ok(bytes)
}

fn safe_cubin_relative_path(name: &str) -> Result<PathBuf, FlashInferError> {
    let path = Path::new(name);
    if name.is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(FlashInferError::invalid_argument(format!(
            "unsafe FlashInfer cubin path `{name}`"
        )));
    }
    Ok(path.to_path_buf())
}

fn extract_artifacts(
    resolved: &ResolvedRuntimeConfig,
    materialized_wheels: &MaterializedWheels,
) -> Result<ExtractedArtifacts, FlashInferError> {
    let artifact_dir = artifact_dir_for(resolved)?;

    let lock_path = artifact_dir.join(".extract.lock");
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| FlashInferError::CacheLock {
            path: lock_path.clone(),
            source: e,
        })?;
    lock_file
        .lock_exclusive()
        .map_err(|e| FlashInferError::CacheLock {
            path: lock_path.clone(),
            source: e,
        })?;

    let norm_so_path = artifact_dir.join("norm.so");
    if !norm_so_path.exists() {
        extract_member_from_wheel_by_suffix(
            &materialized_wheels.jit_cache_wheel_path,
            FLASHINFER_NORM_SO_SUFFIX,
            &norm_so_path,
        )?;
    }

    let gdn_prefill_sm90_so_path = artifact_dir.join("gdn_prefill_sm90.so");
    if !gdn_prefill_sm90_so_path.exists() {
        extract_member_from_wheel_by_suffix(
            &materialized_wheels.jit_cache_wheel_path,
            FLASHINFER_GDN_PREFILL_SM90_SO_SUFFIX,
            &gdn_prefill_sm90_so_path,
        )?;
    }

    let page_so_path = artifact_dir.join("page.so");
    if !page_so_path.exists() {
        extract_member_from_wheel_by_suffix(
            &materialized_wheels.jit_cache_wheel_path,
            FLASHINFER_PAGE_SO_SUFFIX,
            &page_so_path,
        )?;
    }

    let sampling_so_path = artifact_dir.join("sampling.so");
    if !sampling_so_path.exists() {
        extract_member_from_wheel_by_suffix(
            &materialized_wheels.jit_cache_wheel_path,
            FLASHINFER_SAMPLING_SO_SUFFIX,
            &sampling_so_path,
        )?;
    }

    let trtllm_comm_so_path = artifact_dir.join("trtllm_comm.so");
    if !trtllm_comm_so_path.exists() {
        extract_member_from_wheel_by_suffix(
            &materialized_wheels.jit_cache_wheel_path,
            FLASHINFER_TRTLLM_COMM_SO_SUFFIX,
            &trtllm_comm_so_path,
        )?;
    }

    let tvmffi_so_path = artifact_dir.join("libtvm_ffi.so");
    if !tvmffi_so_path.exists() {
        extract_member_from_wheel_exact(
            &materialized_wheels.tvmffi_wheel_path,
            TVMFFI_SO_MEMBER,
            &tvmffi_so_path,
        )?;
    }

    let _ = lock_file.unlock();

    Ok(ExtractedArtifacts {
        artifact_dir,
        norm_so_path,
        gdn_prefill_sm90_so_path,
        page_so_path,
        sampling_so_path,
        trtllm_comm_so_path,
        tvmffi_so_path,
    })
}

fn pinned_flashinfer_jit_cache_wheel() -> PinnedWheelMetadata<'static> {
    PinnedWheelMetadata {
        logical_name: "flashinfer_jit_cache",
        filename: PINNED_FLASHINFER_JIT_CACHE_WHEEL_FILENAME,
        url: PINNED_FLASHINFER_JIT_CACHE_WHEEL_URL,
        sha256_hex: PINNED_FLASHINFER_JIT_CACHE_WHEEL_SHA256,
    }
}

fn pinned_apache_tvm_ffi_wheel() -> PinnedWheelMetadata<'static> {
    PinnedWheelMetadata {
        logical_name: "apache_tvm_ffi",
        filename: PINNED_APACHE_TVM_FFI_WHEEL_FILENAME,
        url: PINNED_APACHE_TVM_FFI_WHEEL_URL,
        sha256_hex: PINNED_APACHE_TVM_FFI_WHEEL_SHA256,
    }
}

fn ensure_pinned_wheels_cached(
    resolved: &ResolvedRuntimeConfig,
) -> Result<MaterializedWheels, FlashInferError> {
    let primary_wheels_dir = resolved.cache_dir.join(WHEEL_CACHE_DIR_NAME);
    fs::create_dir_all(&primary_wheels_dir).map_err(|e| FlashInferError::CreateCacheDir {
        path: primary_wheels_dir.clone(),
        source: e,
    })?;
    let seed_wheels_dir = resolved
        .seed_cache_dir
        .as_ref()
        .map(|seed_cache_dir| seed_cache_dir.join(WHEEL_CACHE_DIR_NAME));

    let jit_cache_wheel_path = ensure_pinned_wheel_cached(
        &primary_wheels_dir,
        seed_wheels_dir.as_deref(),
        pinned_flashinfer_jit_cache_wheel(),
    )?;
    let tvmffi_wheel_path = ensure_pinned_wheel_cached(
        &primary_wheels_dir,
        seed_wheels_dir.as_deref(),
        pinned_apache_tvm_ffi_wheel(),
    )?;

    Ok(MaterializedWheels {
        jit_cache_wheel_path,
        tvmffi_wheel_path,
    })
}

fn ensure_pinned_wheel_cached(
    primary_wheels_dir: &Path,
    seed_wheels_dir: Option<&Path>,
    wheel: PinnedWheelMetadata<'_>,
) -> Result<PathBuf, FlashInferError> {
    ensure_pinned_wheel_cached_with_downloader(
        primary_wheels_dir,
        seed_wheels_dir,
        wheel,
        download_pinned_wheel,
    )
}

fn ensure_pinned_wheel_cached_with_downloader<F>(
    primary_wheels_dir: &Path,
    seed_wheels_dir: Option<&Path>,
    wheel: PinnedWheelMetadata<'_>,
    mut downloader: F,
) -> Result<PathBuf, FlashInferError>
where
    F: FnMut(&Path, PinnedWheelMetadata<'_>) -> Result<(), FlashInferError>,
{
    cleanup_stale_download_temps(primary_wheels_dir, wheel);
    let output_filename = format!("{}-{}", wheel.sha256_hex, wheel.filename);
    let output_path = primary_wheels_dir.join(&output_filename);
    let lock_path = output_path.with_extension("lock");
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| FlashInferError::CacheLock {
            path: lock_path.clone(),
            source: e,
        })?;
    lock_file
        .lock_exclusive()
        .map_err(|e| FlashInferError::CacheLock {
            path: lock_path.clone(),
            source: e,
        })?;

    if output_path.exists() {
        let found = sha256_file_hex(&output_path, wheel.logical_name)?;
        if found == wheel.sha256_hex {
            let _ = lock_file.unlock();
            return Ok(output_path);
        }
        fs::remove_file(&output_path).map_err(|e| FlashInferError::EmbeddedWheelCache {
            wheel: wheel.logical_name,
            path: output_path.clone(),
            source: e,
        })?;
    }

    if let Some(seed_wheels_dir) = seed_wheels_dir {
        let seed_path = seed_wheels_dir.join(output_filename);
        if seed_path.exists() {
            if sha256_file_hex(&seed_path, wheel.logical_name)
                .map(|found| found == wheel.sha256_hex)
                .unwrap_or(false)
            {
                let _ = lock_file.unlock();
                return Ok(seed_path);
            }
        }
    }

    match downloader(&output_path, wheel) {
        Ok(()) => {
            let _ = lock_file.unlock();
            Ok(output_path)
        }
        Err(err) => {
            let _ = fs::remove_file(&output_path);
            let _ = lock_file.unlock();
            Err(err)
        }
    }
}

fn cleanup_stale_download_temps(wheels_dir: &Path, wheel: PinnedWheelMetadata<'_>) {
    let Ok(entries) = fs::read_dir(wheels_dir) else {
        return;
    };
    let modern_prefix = format!("{}-{}.tmp-", wheel.sha256_hex, wheel.filename);
    let legacy_prefix = format!(
        "{}-{}.tmp-",
        wheel.sha256_hex,
        wheel.filename.trim_end_matches(".whl")
    );
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with(&modern_prefix) || name.starts_with(&legacy_prefix) {
            let _ = fs::remove_file(path);
        }
    }
}

fn download_pinned_wheel(
    output_path: &Path,
    wheel: PinnedWheelMetadata<'_>,
) -> Result<(), FlashInferError> {
    let response =
        ureq::get(wheel.url)
            .call()
            .map_err(|e| FlashInferError::EmbeddedWheelCache {
                wheel: wheel.logical_name,
                path: output_path.to_path_buf(),
                source: io::Error::other(format!(
                    "failed to download `{}` from `{}`: {e}",
                    wheel.logical_name, wheel.url
                )),
            })?;
    let mut reader = response.into_reader();
    write_wheel_from_reader(&mut reader, output_path, wheel)
}

fn write_wheel_from_reader<R: Read>(
    reader: &mut R,
    output_path: &Path,
    wheel: PinnedWheelMetadata<'_>,
) -> Result<(), FlashInferError> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| FlashInferError::CreateCacheDir {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }

    let temp_path = output_path.with_extension(format!("tmp-{}", std::process::id()));
    if temp_path.exists() {
        let _ = fs::remove_file(&temp_path);
    }

    {
        let mut out =
            File::create(&temp_path).map_err(|e| FlashInferError::EmbeddedWheelCache {
                wheel: wheel.logical_name,
                path: output_path.to_path_buf(),
                source: e,
            })?;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read =
                reader
                    .read(&mut buffer)
                    .map_err(|e| FlashInferError::EmbeddedWheelCache {
                        wheel: wheel.logical_name,
                        path: output_path.to_path_buf(),
                        source: e,
                    })?;
            if read == 0 {
                break;
            }
            let chunk = &buffer[..read];
            out.write_all(chunk)
                .map_err(|e| FlashInferError::EmbeddedWheelCache {
                    wheel: wheel.logical_name,
                    path: output_path.to_path_buf(),
                    source: e,
                })?;
        }
        out.sync_all()
            .map_err(|e| FlashInferError::EmbeddedWheelCache {
                wheel: wheel.logical_name,
                path: output_path.to_path_buf(),
                source: e,
            })?;
    }

    let found = match sha256_file_hex(&temp_path, wheel.logical_name) {
        Ok(found) => found,
        Err(err) => {
            let _ = fs::remove_file(&temp_path);
            return Err(err);
        }
    };
    if found != wheel.sha256_hex {
        let _ = fs::remove_file(&temp_path);
        return Err(FlashInferError::EmbeddedWheelChecksumMismatch {
            wheel: wheel.logical_name,
            path: output_path.to_path_buf(),
            expected: wheel.sha256_hex.to_string(),
            found,
        });
    }

    match fs::rename(&temp_path, output_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
            let _ = fs::remove_file(&temp_path);
            let existing = sha256_file_hex(output_path, wheel.logical_name)?;
            if existing == wheel.sha256_hex {
                Ok(())
            } else {
                Err(FlashInferError::EmbeddedWheelChecksumMismatch {
                    wheel: wheel.logical_name,
                    path: output_path.to_path_buf(),
                    expected: wheel.sha256_hex.to_string(),
                    found: existing,
                })
            }
        }
        Err(err) => {
            let _ = fs::remove_file(&temp_path);
            Err(FlashInferError::EmbeddedWheelCache {
                wheel: wheel.logical_name,
                path: output_path.to_path_buf(),
                source: err,
            })
        }
    }
}

fn extract_jit_kernel(
    jit_cache_wheel: &Path,
    artifact_dir: &Path,
    kernel_uri: &str,
) -> Result<PathBuf, FlashInferError> {
    let member_suffix = format!("flashinfer_jit_cache/jit_cache/{kernel_uri}/{kernel_uri}.so");
    let output_path = artifact_dir
        .join("jit_cache")
        .join(kernel_uri)
        .join(format!("{kernel_uri}.so"));

    if output_path.exists() {
        return Ok(output_path);
    }

    fs::create_dir_all(artifact_dir).map_err(|e| FlashInferError::CreateCacheDir {
        path: artifact_dir.to_path_buf(),
        source: e,
    })?;

    let lock_path = artifact_dir.join(".extract.lock");
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| FlashInferError::CacheLock {
            path: lock_path.clone(),
            source: e,
        })?;
    lock_file
        .lock_exclusive()
        .map_err(|e| FlashInferError::CacheLock {
            path: lock_path.clone(),
            source: e,
        })?;

    if !output_path.exists() {
        extract_member_from_wheel_by_suffix(jit_cache_wheel, &member_suffix, &output_path)?;
    }

    let _ = lock_file.unlock();
    Ok(output_path)
}

fn artifact_dir_for(resolved: &ResolvedRuntimeConfig) -> Result<PathBuf, FlashInferError> {
    fs::create_dir_all(&resolved.cache_dir).map_err(|e| FlashInferError::CreateCacheDir {
        path: resolved.cache_dir.clone(),
        source: e,
    })?;

    let artifact_hash = artifact_hash();
    let artifact_dir = resolved.cache_dir.join(artifact_hash);
    fs::create_dir_all(&artifact_dir).map_err(|e| FlashInferError::CreateCacheDir {
        path: artifact_dir.clone(),
        source: e,
    })?;
    Ok(artifact_dir)
}

fn artifact_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(PINNED_FLASHINFER_JIT_CACHE_WHEEL_SHA256.as_bytes());
    hasher.update(PINNED_APACHE_TVM_FFI_WHEEL_SHA256.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn sha256_file_hex(path: &Path, wheel: &'static str) -> Result<String, FlashInferError> {
    // Hashing the ~1.8 GiB JIT-cache wheel with `sha2` takes over 100 seconds
    // in debug builds. Prefer the host's optimized `sha256sum` executable so
    // non-release test and development builds do not pay that cost.
    if let Some(program) = host_sha256sum() {
        return sha256_file_hex_with_command(program, path, wheel);
    }
    sha256_file_hex_rust(path, wheel)
}

fn host_sha256sum() -> Option<&'static Path> {
    HOST_SHA256SUM
        .get_or_init(|| {
            env::var_os("PATH")
                .as_deref()
                .and_then(|path| find_executable_in_path(SHA256SUM_PROGRAM, path))
        })
        .as_deref()
}

fn find_executable_in_path(program: &str, path: &std::ffi::OsStr) -> Option<PathBuf> {
    env::split_paths(path).find_map(|directory| {
        let candidate = directory.join(program);
        let metadata = candidate.metadata().ok()?;
        if metadata.is_file() && metadata.permissions().mode() & 0o111 != 0 {
            Some(candidate)
        } else {
            None
        }
    })
}

fn sha256_file_hex_with_command(
    program: &Path,
    path: &Path,
    wheel: &'static str,
) -> Result<String, FlashInferError> {
    let output = Command::new(program)
        .arg("--")
        .arg(path)
        .output()
        .map_err(|e| FlashInferError::EmbeddedWheelCache {
            wheel,
            path: path.to_path_buf(),
            source: e,
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(FlashInferError::EmbeddedWheelCache {
            wheel,
            path: path.to_path_buf(),
            source: io::Error::other(format!(
                "`{}` failed with status {}: {}",
                program.display(),
                output.status,
                stderr.trim()
            )),
        });
    }
    parse_sha256sum_output(&output.stdout).map_err(|e| FlashInferError::EmbeddedWheelCache {
        wheel,
        path: path.to_path_buf(),
        source: e,
    })
}

fn parse_sha256sum_output(output: &[u8]) -> io::Result<String> {
    let digest = output
        .split(|byte| byte.is_ascii_whitespace())
        .next()
        .filter(|digest| !digest.is_empty())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "sha256sum returned no digest")
        })?;
    if digest.len() != 64 || !digest.iter().all(u8::is_ascii_hexdigit) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "sha256sum returned an invalid SHA-256 digest",
        ));
    }
    Ok(String::from_utf8_lossy(digest).to_ascii_lowercase())
}

fn sha256_file_hex_rust(path: &Path, wheel: &'static str) -> Result<String, FlashInferError> {
    let mut file = File::open(path).map_err(|e| FlashInferError::EmbeddedWheelCache {
        wheel,
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| FlashInferError::EmbeddedWheelCache {
                wheel,
                path: path.to_path_buf(),
                source: e,
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_member_from_wheel_by_suffix(
    wheel_path: &Path,
    member_suffix: &str,
    output_path: &Path,
) -> Result<(), FlashInferError> {
    let file = File::open(wheel_path).map_err(|e| FlashInferError::WheelOpen {
        wheel: wheel_path.to_path_buf(),
        source: e,
    })?;
    let mut archive = ZipArchive::new(file).map_err(|e| FlashInferError::WheelRead {
        wheel: wheel_path.to_path_buf(),
        source: e,
    })?;

    let mut found = None;
    for i in 0..archive.len() {
        let entry_name = archive
            .by_index(i)
            .map_err(|e| FlashInferError::WheelRead {
                wheel: wheel_path.to_path_buf(),
                source: e,
            })?
            .name()
            .to_string();
        if entry_name.ends_with(member_suffix) {
            found = Some(entry_name);
            break;
        }
    }

    let Some(member_name) = found else {
        return Err(FlashInferError::WheelEntryMissing {
            wheel: wheel_path.to_path_buf(),
            entry: member_suffix.to_string(),
        });
    };

    extract_member_from_open_archive(&mut archive, wheel_path, &member_name, output_path)
}

fn extract_member_from_wheel_exact(
    wheel_path: &Path,
    member_name: &str,
    output_path: &Path,
) -> Result<(), FlashInferError> {
    let file = File::open(wheel_path).map_err(|e| FlashInferError::WheelOpen {
        wheel: wheel_path.to_path_buf(),
        source: e,
    })?;
    let mut archive = ZipArchive::new(file).map_err(|e| FlashInferError::WheelRead {
        wheel: wheel_path.to_path_buf(),
        source: e,
    })?;

    if archive.by_name(member_name).is_err() {
        return Err(FlashInferError::WheelEntryMissing {
            wheel: wheel_path.to_path_buf(),
            entry: member_name.to_string(),
        });
    }

    extract_member_from_open_archive(&mut archive, wheel_path, member_name, output_path)
}

fn extract_member_from_open_archive<R: io::Read + io::Seek>(
    archive: &mut ZipArchive<R>,
    wheel_path: &Path,
    member_name: &str,
    output_path: &Path,
) -> Result<(), FlashInferError> {
    let mut entry = archive
        .by_name(member_name)
        .map_err(|e| FlashInferError::WheelRead {
            wheel: wheel_path.to_path_buf(),
            source: e,
        })?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|e| FlashInferError::CreateCacheDir {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }

    let temp_path = output_path.with_extension("tmp");
    {
        let mut out = File::create(&temp_path).map_err(|e| FlashInferError::WheelExtract {
            wheel: wheel_path.to_path_buf(),
            entry: member_name.to_string(),
            output: output_path.to_path_buf(),
            source: e,
        })?;
        io::copy(&mut entry, &mut out).map_err(|e| FlashInferError::WheelExtract {
            wheel: wheel_path.to_path_buf(),
            entry: member_name.to_string(),
            output: output_path.to_path_buf(),
            source: e,
        })?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mode = entry.unix_mode().unwrap_or(0o755);
        fs::set_permissions(&temp_path, fs::Permissions::from_mode(mode)).map_err(|e| {
            FlashInferError::WheelExtract {
                wheel: wheel_path.to_path_buf(),
                entry: member_name.to_string(),
                output: output_path.to_path_buf(),
                source: e,
            }
        })?;
    }

    fs::rename(&temp_path, output_path).map_err(|e| FlashInferError::WheelExtract {
        wheel: wheel_path.to_path_buf(),
        entry: member_name.to_string(),
        output: output_path.to_path_buf(),
        source: e,
    })
}

fn env_path(name: &'static str) -> Result<Option<PathBuf>, FlashInferError> {
    let Some(value) = env::var_os(name) else {
        return Ok(None);
    };

    if value.is_empty() {
        return Err(FlashInferError::InvalidEnvironment {
            name,
            message: "value is empty".to_string(),
        });
    }

    Ok(Some(PathBuf::from(value)))
}

fn default_cache_dir() -> Result<PathBuf, FlashInferError> {
    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home).join(".cache").join("flashinfer-rs"));
    }

    if let Some(xdg) = env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(xdg).join("flashinfer-rs"));
    }

    Err(FlashInferError::invalid_argument(
        "unable to resolve cache directory; set FLASHINFER_RS_CACHE_DIR",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{
        DLDataType, DLDevice, DLTensor, KDL_CUDA, KDL_FLOAT, TVMFFIAny, any_bool, any_dltensor_ptr,
        any_f64, any_none,
    };
    use std::cell::Cell;
    use std::io::Cursor;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Mutex;
    use std::time::Duration;

    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn sha256_bytes_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    fn pinned_test_wheel<'a>(filename: &'a str, sha256_hex: &'a str) -> PinnedWheelMetadata<'a> {
        PinnedWheelMetadata {
            logical_name: "test_wheel",
            filename,
            url: "https://unused.invalid/test.whl",
            sha256_hex,
        }
    }

    fn cached_wheel_path(wheels_dir: &Path, wheel: PinnedWheelMetadata<'_>) -> PathBuf {
        wheels_dir.join(format!("{}-{}", wheel.sha256_hex, wheel.filename))
    }

    #[test]
    fn env_path_empty_is_error() {
        let _guard = ENV_TEST_LOCK.lock().expect("env lock");
        unsafe {
            env::set_var("FLASHINFER_RS_TEST_EMPTY", "");
        }
        let result = env_path("FLASHINFER_RS_TEST_EMPTY");
        unsafe {
            env::remove_var("FLASHINFER_RS_TEST_EMPTY");
        }
        assert!(result.is_err());
    }

    #[test]
    fn artifact_hash_is_deterministic_from_pinned_shas() {
        let mut hasher = Sha256::new();
        hasher.update(PINNED_FLASHINFER_JIT_CACHE_WHEEL_SHA256.as_bytes());
        hasher.update(PINNED_APACHE_TVM_FFI_WHEEL_SHA256.as_bytes());
        let expected = format!("{:x}", hasher.finalize());
        assert_eq!(artifact_hash(), expected);
    }

    #[test]
    fn parse_sha256sum_output_accepts_and_normalizes_digest() {
        let uppercase =
            b"ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789  wheel.whl\n";
        assert_eq!(
            parse_sha256sum_output(uppercase).expect("parse digest"),
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        );
    }

    #[test]
    fn parse_sha256sum_output_rejects_invalid_digest() {
        assert!(parse_sha256sum_output(b"not-a-digest  wheel.whl\n").is_err());
        assert!(parse_sha256sum_output(b"").is_err());
    }

    #[test]
    fn find_executable_in_path_requires_executable_file() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let program = tmpdir.path().join(SHA256SUM_PROGRAM);
        fs::write(&program, b"#!/bin/sh\nexit 0\n").expect("write program");

        let mut permissions = fs::metadata(&program).expect("metadata").permissions();
        permissions.set_mode(0o644);
        fs::set_permissions(&program, permissions).expect("set non-executable");
        assert_eq!(
            find_executable_in_path(SHA256SUM_PROGRAM, tmpdir.path().as_os_str()),
            None
        );

        let mut permissions = fs::metadata(&program).expect("metadata").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&program, permissions).expect("set executable");
        assert_eq!(
            find_executable_in_path(SHA256SUM_PROGRAM, tmpdir.path().as_os_str()),
            Some(program)
        );
    }

    #[test]
    fn command_and_rust_sha256_paths_match() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let wheel_path = tmpdir.path().join("wheel with spaces.whl");
        let bytes = b"wheel-hash-test";
        fs::write(&wheel_path, bytes).expect("write wheel");
        let expected = sha256_bytes_hex(bytes);

        let program = tmpdir.path().join("mock-sha256sum");
        fs::write(
            &program,
            format!("#!/bin/sh\nprintf '%s  %s\\n' '{expected}' \"$2\"\n"),
        )
        .expect("write mock sha256sum");
        let mut permissions = fs::metadata(&program).expect("metadata").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&program, permissions).expect("set executable");

        assert_eq!(
            sha256_file_hex_with_command(&program, &wheel_path, "test_wheel")
                .expect("command hash"),
            expected
        );
        assert_eq!(
            sha256_file_hex_rust(&wheel_path, "test_wheel").expect("Rust hash"),
            expected
        );
    }

    #[test]
    fn sha256sum_command_failure_is_not_silently_ignored() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let wheel_path = tmpdir.path().join("wheel.whl");
        fs::write(&wheel_path, b"wheel").expect("write wheel");

        let program = tmpdir.path().join("failing-sha256sum");
        fs::write(&program, b"#!/bin/sh\necho failed >&2\nexit 17\n")
            .expect("write mock sha256sum");
        let mut permissions = fs::metadata(&program).expect("metadata").permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&program, permissions).expect("set executable");

        let error = sha256_file_hex_with_command(&program, &wheel_path, "test_wheel")
            .expect_err("command failure");
        match error {
            FlashInferError::EmbeddedWheelCache { source, .. } => {
                assert!(source.to_string().contains("status"));
                assert!(source.to_string().contains("failed"));
            }
            other => panic!("unexpected error variant: {other}"),
        }
    }

    #[test]
    fn ensure_pinned_wheel_cached_creates_missing_file() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let wheels_dir = tmpdir.path().join("wheels");
        fs::create_dir_all(&wheels_dir).expect("create wheels dir");

        let bytes = b"wheel-bytes-v1";
        let sha = sha256_bytes_hex(bytes);
        let wheel = pinned_test_wheel("test.whl", &sha);

        let output_path = ensure_pinned_wheel_cached_with_downloader(
            &wheels_dir,
            None,
            wheel,
            |output_path, wheel| {
                let mut reader = Cursor::new(bytes.as_slice());
                write_wheel_from_reader(&mut reader, output_path, wheel)
            },
        )
        .expect("cache wheel");
        let written = fs::read(&output_path).expect("read output");
        assert_eq!(written, bytes);
    }

    #[test]
    fn ensure_pinned_wheel_cached_reuses_when_checksum_matches() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let wheels_dir = tmpdir.path().join("wheels");
        fs::create_dir_all(&wheels_dir).expect("create wheels dir");

        let bytes = b"wheel-bytes-v2";
        let sha = sha256_bytes_hex(bytes);
        let wheel = pinned_test_wheel("reuse.whl", &sha);

        let output_path = ensure_pinned_wheel_cached_with_downloader(
            &wheels_dir,
            None,
            wheel,
            |output_path, wheel| {
                let mut reader = Cursor::new(bytes.as_slice());
                write_wheel_from_reader(&mut reader, output_path, wheel)
            },
        )
        .expect("cache wheel 1");
        let before = fs::metadata(&output_path)
            .expect("metadata 1")
            .modified()
            .expect("modified 1");
        std::thread::sleep(Duration::from_millis(1100));
        let output_path_2 = ensure_pinned_wheel_cached_with_downloader(
            &wheels_dir,
            None,
            wheel,
            |_output_path, _wheel| {
                panic!("downloader should not be called on cache hit");
            },
        )
        .expect("cache wheel 2");
        let after = fs::metadata(&output_path_2)
            .expect("metadata 2")
            .modified()
            .expect("modified 2");

        assert_eq!(output_path, output_path_2);
        assert_eq!(before, after);
    }

    #[test]
    fn ensure_pinned_wheel_cached_rewrites_when_checksum_mismatch() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let wheels_dir = tmpdir.path().join("wheels");
        fs::create_dir_all(&wheels_dir).expect("create wheels dir");

        let bytes = b"wheel-bytes-v3";
        let sha = sha256_bytes_hex(bytes);
        let wheel = pinned_test_wheel("rewrite.whl", &sha);
        let target_path = cached_wheel_path(&wheels_dir, wheel);
        fs::write(&target_path, b"corrupt-data").expect("write corrupt");

        let output_path = ensure_pinned_wheel_cached_with_downloader(
            &wheels_dir,
            None,
            wheel,
            |output_path, wheel| {
                let mut reader = Cursor::new(bytes.as_slice());
                write_wheel_from_reader(&mut reader, output_path, wheel)
            },
        )
        .expect("cache wheel");
        let written = fs::read(&output_path).expect("read output");
        assert_eq!(written, bytes);
    }

    #[test]
    fn ensure_pinned_wheel_cached_primary_hit_wins_over_seed() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let primary_wheels_dir = tmpdir.path().join("primary").join("wheels");
        let seed_wheels_dir = tmpdir.path().join("seed").join("wheels");
        fs::create_dir_all(&primary_wheels_dir).expect("create primary wheels dir");
        fs::create_dir_all(&seed_wheels_dir).expect("create seed wheels dir");

        let bytes = b"wheel-bytes-primary";
        let sha = sha256_bytes_hex(bytes);
        let wheel = pinned_test_wheel("primary-hit.whl", &sha);
        fs::write(cached_wheel_path(&primary_wheels_dir, wheel), bytes).expect("write primary");
        fs::write(cached_wheel_path(&seed_wheels_dir, wheel), b"corrupt-seed")
            .expect("write corrupt seed");

        let output_path = ensure_pinned_wheel_cached_with_downloader(
            &primary_wheels_dir,
            Some(&seed_wheels_dir),
            wheel,
            |_output_path, _wheel| {
                panic!("downloader should not be called on primary cache hit");
            },
        )
        .expect("cache wheel");

        assert_eq!(output_path, cached_wheel_path(&primary_wheels_dir, wheel));
    }

    #[test]
    fn ensure_pinned_wheel_cached_returns_valid_seed_when_primary_missing() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let primary_wheels_dir = tmpdir.path().join("primary").join("wheels");
        let seed_wheels_dir = tmpdir.path().join("seed").join("wheels");
        fs::create_dir_all(&primary_wheels_dir).expect("create primary wheels dir");
        fs::create_dir_all(&seed_wheels_dir).expect("create seed wheels dir");

        let bytes = b"wheel-bytes-seed";
        let sha = sha256_bytes_hex(bytes);
        let wheel = pinned_test_wheel("seed-hit.whl", &sha);
        fs::write(cached_wheel_path(&seed_wheels_dir, wheel), bytes).expect("write seed");

        let output_path = ensure_pinned_wheel_cached_with_downloader(
            &primary_wheels_dir,
            Some(&seed_wheels_dir),
            wheel,
            |_output_path, _wheel| {
                panic!("downloader should not be called on seed cache hit");
            },
        )
        .expect("cache wheel");

        assert_eq!(output_path, cached_wheel_path(&seed_wheels_dir, wheel));
        assert!(
            !cached_wheel_path(&primary_wheels_dir, wheel).exists(),
            "seed hit should not be copied into primary cache"
        );
    }

    #[test]
    fn ensure_pinned_wheel_cached_corrupt_primary_can_use_valid_seed() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let primary_wheels_dir = tmpdir.path().join("primary").join("wheels");
        let seed_wheels_dir = tmpdir.path().join("seed").join("wheels");
        fs::create_dir_all(&primary_wheels_dir).expect("create primary wheels dir");
        fs::create_dir_all(&seed_wheels_dir).expect("create seed wheels dir");

        let bytes = b"wheel-bytes-valid-seed";
        let sha = sha256_bytes_hex(bytes);
        let wheel = pinned_test_wheel("corrupt-primary.whl", &sha);
        let primary_path = cached_wheel_path(&primary_wheels_dir, wheel);
        let seed_path = cached_wheel_path(&seed_wheels_dir, wheel);
        fs::write(&primary_path, b"corrupt-primary").expect("write corrupt primary");
        fs::write(&seed_path, bytes).expect("write seed");

        let output_path = ensure_pinned_wheel_cached_with_downloader(
            &primary_wheels_dir,
            Some(&seed_wheels_dir),
            wheel,
            |_output_path, _wheel| {
                panic!("downloader should not be called on valid seed cache hit");
            },
        )
        .expect("cache wheel");

        assert_eq!(output_path, seed_path);
        assert!(!primary_path.exists(), "corrupt primary should be removed");
    }

    #[test]
    fn ensure_pinned_wheel_cached_corrupt_seed_downloads_primary() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let primary_wheels_dir = tmpdir.path().join("primary").join("wheels");
        let seed_wheels_dir = tmpdir.path().join("seed").join("wheels");
        fs::create_dir_all(&primary_wheels_dir).expect("create primary wheels dir");
        fs::create_dir_all(&seed_wheels_dir).expect("create seed wheels dir");

        let bytes = b"wheel-bytes-download";
        let sha = sha256_bytes_hex(bytes);
        let wheel = pinned_test_wheel("corrupt-seed.whl", &sha);
        let primary_path = cached_wheel_path(&primary_wheels_dir, wheel);
        let seed_path = cached_wheel_path(&seed_wheels_dir, wheel);
        fs::write(&seed_path, b"corrupt-seed").expect("write corrupt seed");
        let downloaded = Cell::new(false);

        let output_path = ensure_pinned_wheel_cached_with_downloader(
            &primary_wheels_dir,
            Some(&seed_wheels_dir),
            wheel,
            |output_path, wheel| {
                downloaded.set(true);
                let mut reader = Cursor::new(bytes.as_slice());
                write_wheel_from_reader(&mut reader, output_path, wheel)
            },
        )
        .expect("cache wheel");

        assert!(downloaded.get(), "downloader should be called");
        assert_eq!(output_path, primary_path);
        assert_eq!(fs::read(&primary_path).expect("read primary"), bytes);
        assert_eq!(
            fs::read(&seed_path).expect("read seed"),
            b"corrupt-seed",
            "corrupt seed should not be modified"
        );
    }

    #[test]
    fn write_wheel_from_reader_checksum_mismatch_leaves_no_output() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let wheels_dir = tmpdir.path().join("wheels");
        fs::create_dir_all(&wheels_dir).expect("create wheels dir");

        let good_bytes = b"wheel-bytes-expected";
        let bad_bytes = b"wheel-bytes-actual";
        let sha = sha256_bytes_hex(good_bytes);
        let output_path = wheels_dir.join(format!("{sha}-bad.whl"));
        let wheel = pinned_test_wheel("bad.whl", &sha);

        let mut reader = Cursor::new(bad_bytes.as_slice());
        let err = write_wheel_from_reader(&mut reader, &output_path, wheel)
            .expect_err("checksum mismatch expected");

        match err {
            FlashInferError::EmbeddedWheelChecksumMismatch { .. } => {}
            other => panic!("unexpected error variant: {other}"),
        }

        assert!(
            !output_path.exists(),
            "output should not exist on checksum mismatch"
        );
    }

    #[test]
    fn seed_cache_env_empty_is_error() {
        let _guard = ENV_TEST_LOCK.lock().expect("env lock");
        let prev_seed = env::var_os(ENV_SEED_CACHE_DIR);

        unsafe {
            env::set_var(ENV_SEED_CACHE_DIR, "");
        }

        let result = RuntimeConfig::from_env();

        unsafe {
            match prev_seed {
                Some(v) => env::set_var(ENV_SEED_CACHE_DIR, v),
                None => env::remove_var(ENV_SEED_CACHE_DIR),
            }
        }

        match result {
            Err(FlashInferError::InvalidEnvironment { name, .. }) => {
                assert_eq!(name, ENV_SEED_CACHE_DIR);
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn explicit_seed_cache_dir_overrides_env_seed_cache_dir() {
        let _guard = ENV_TEST_LOCK.lock().expect("env lock");
        let prev_seed = env::var_os(ENV_SEED_CACHE_DIR);
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let env_seed = tmpdir.path().join("env-seed");
        let explicit_seed = tmpdir.path().join("explicit-seed");

        unsafe {
            env::set_var(ENV_SEED_CACHE_DIR, &env_seed);
        }

        let resolved = RuntimeConfig::default()
            .with_seed_cache_dir(&explicit_seed)
            .resolve()
            .expect("resolve");

        unsafe {
            match prev_seed {
                Some(v) => env::set_var(ENV_SEED_CACHE_DIR, v),
                None => env::remove_var(ENV_SEED_CACHE_DIR),
            }
        }

        assert_eq!(resolved.seed_cache_dir, Some(explicit_seed));
    }

    #[test]
    fn legacy_wheel_env_vars_are_ignored() {
        let _guard = ENV_TEST_LOCK.lock().expect("env lock");
        let prev_jit = env::var_os("FLASHINFER_RS_JIT_CACHE_WHEEL");
        let prev_tvm = env::var_os("FLASHINFER_RS_TVMFFI_WHEEL");
        let prev_cache = env::var_os(ENV_CACHE_DIR);
        let prev_seed = env::var_os(ENV_SEED_CACHE_DIR);

        unsafe {
            env::set_var("FLASHINFER_RS_JIT_CACHE_WHEEL", "/tmp/legacy-jit.whl");
            env::set_var("FLASHINFER_RS_TVMFFI_WHEEL", "/tmp/legacy-tvm.whl");
            env::remove_var(ENV_CACHE_DIR);
            env::remove_var(ENV_SEED_CACHE_DIR);
        }

        let cfg = RuntimeConfig::from_env().expect("from env");
        assert_eq!(cfg.cache_dir, None);
        assert_eq!(cfg.seed_cache_dir, None);

        unsafe {
            match prev_jit {
                Some(v) => env::set_var("FLASHINFER_RS_JIT_CACHE_WHEEL", v),
                None => env::remove_var("FLASHINFER_RS_JIT_CACHE_WHEEL"),
            }
            match prev_tvm {
                Some(v) => env::set_var("FLASHINFER_RS_TVMFFI_WHEEL", v),
                None => env::remove_var("FLASHINFER_RS_TVMFFI_WHEEL"),
            }
            match prev_cache {
                Some(v) => env::set_var(ENV_CACHE_DIR, v),
                None => env::remove_var(ENV_CACHE_DIR),
            }
            match prev_seed {
                Some(v) => env::set_var(ENV_SEED_CACHE_DIR, v),
                None => env::remove_var(ENV_SEED_CACHE_DIR),
            }
        }
    }

    #[test]
    fn gpu_ffi_error_path_decodes_raised_error() {
        if env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() != Some("1") {
            eprintln!("skipping GPU ffi error test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
            return;
        }

        let runtime = FlashInferRuntime::global().expect("initialize runtime");

        let mut out_shape = [2_i64, 4_i64];
        let mut out_strides = [4_i64, 1_i64];
        let out = DLTensor {
            data: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: KDL_FLOAT,
                bits: 16,
                lanes: 1,
            },
            shape: out_shape.as_mut_ptr(),
            strides: out_strides.as_mut_ptr(),
            byte_offset: 0,
        };

        // Intentionally rank-3 to trigger CHECK_DIM(2, input) inside gemma_rmsnorm.
        let mut input_shape = [2_i64, 1_i64, 4_i64];
        let mut input_strides = [4_i64, 4_i64, 1_i64];
        let input = DLTensor {
            data: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: 0,
            },
            ndim: 3,
            dtype: DLDataType {
                code: KDL_FLOAT,
                bits: 16,
                lanes: 1,
            },
            shape: input_shape.as_mut_ptr(),
            strides: input_strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let mut weight_shape = [4_i64];
        let mut weight_strides = [1_i64];
        let weight = DLTensor {
            data: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
            device: DLDevice {
                device_type: KDL_CUDA,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: KDL_FLOAT,
                bits: 16,
                lanes: 1,
            },
            shape: weight_shape.as_mut_ptr(),
            strides: weight_strides.as_mut_ptr(),
            byte_offset: 0,
        };

        let args: [TVMFFIAny; 5] = [
            any_dltensor_ptr(&out),
            any_dltensor_ptr(&input),
            any_dltensor_ptr(&weight),
            any_f64(1e-6),
            any_bool(false),
        ];
        let mut result = any_none();

        let err = unsafe {
            runtime
                .call_gemma_rmsnorm(args.as_ptr(), args.len() as i32, &mut result as *mut _)
                .expect_err("expected rank mismatch error")
        };

        match err {
            FlashInferError::TvmFfiCall { message, .. } => {
                assert!(
                    message.contains("dimension")
                        || message.contains("dim")
                        || message.contains("ndim"),
                    "unexpected error message: {message}"
                );
            }
            other => panic!("unexpected error variant: {other}"),
        }
    }
}
