use std::collections::HashMap;
use std::env;
use std::ffi::c_char;
use std::ffi::c_void;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use fs2::FileExt;
use libloading::os::unix::Library;
use sha2::{Digest, Sha256};
use zip::ZipArchive;

use crate::error::FlashInferError;
use crate::ffi::{
    KDL_CUDA, TVMFFIAny, TVMFFIByteArray, TVMFFIObjectHandle, TVMFFIVersion, any_none,
    byte_array_to_string, error_cell_ptr,
};

include!(concat!(env!("OUT_DIR"), "/embedded_wheels.rs"));

const ENV_CACHE_DIR: &str = "FLASHINFER_RS_CACHE_DIR";

const FLASHINFER_NORM_SO_SUFFIX: &str = "flashinfer_jit_cache/jit_cache/norm/norm.so";
const FLASHINFER_GDN_PREFILL_SM90_SO_SUFFIX: &str =
    "flashinfer_jit_cache/jit_cache/gdn_prefill_sm90/gdn_prefill_sm90.so";
const FLASHINFER_PAGE_SO_SUFFIX: &str = "flashinfer_jit_cache/jit_cache/page/page.so";
const TVMFFI_SO_MEMBER: &str = "tvm_ffi/lib/libtvm_ffi.so";
const WHEEL_CACHE_DIR_NAME: &str = "wheels";

const EXPECTED_TVMFFI_MAJOR: u32 = 0;
const EXPECTED_TVMFFI_MINOR: u32 = 1;

type TVMFFIGetVersionFn = unsafe extern "C" fn(*mut TVMFFIVersion);
type TVMFFIEnvSetStreamFn = unsafe extern "C" fn(i32, i32, *mut c_void, *mut *mut c_void) -> i32;
type TVMFFIErrorMoveFromRaisedFn = unsafe extern "C" fn(*mut TVMFFIObjectHandle);
type TVMFFIObjectDecRefFn = unsafe extern "C" fn(TVMFFIObjectHandle) -> i32;
type TVMFFIAnyViewToOwnedAnyFn = unsafe extern "C" fn(*const TVMFFIAny, *mut TVMFFIAny) -> i32;
type TVMFFIFunctionGetGlobalFn =
    unsafe extern "C" fn(*const TVMFFIByteArray, *mut TVMFFIObjectHandle) -> i32;
type TVMFFIFunctionCallFn =
    unsafe extern "C" fn(TVMFFIObjectHandle, *mut TVMFFIAny, i32, *mut TVMFFIAny) -> i32;
type TVMFFIStringFromByteArrayFn =
    unsafe extern "C" fn(*const TVMFFIByteArray, *mut TVMFFIAny) -> i32;
type TVMFFISafeCallFn =
    unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedRuntimeConfig {
    cache_dir: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub cache_dir: Option<PathBuf>,
}

impl RuntimeConfig {
    pub fn from_env() -> Result<Self, FlashInferError> {
        Ok(Self {
            cache_dir: env_path(ENV_CACHE_DIR)?,
        })
    }

    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    fn resolve(&self) -> Result<ResolvedRuntimeConfig, FlashInferError> {
        let env_cfg = RuntimeConfig::from_env()?;

        let cache_dir = if let Some(path) = self.cache_dir.clone().or(env_cfg.cache_dir) {
            path
        } else {
            default_cache_dir()?
        };

        Ok(ResolvedRuntimeConfig { cache_dir })
    }
}

#[derive(Debug)]
struct ExtractedArtifacts {
    artifact_dir: PathBuf,
    norm_so_path: PathBuf,
    gdn_prefill_sm90_so_path: PathBuf,
    page_so_path: PathBuf,
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

pub struct FlashInferRuntime {
    resolved: ResolvedRuntimeConfig,
    jit_cache_wheel_path: PathBuf,
    artifact_dir: PathBuf,
    _tvmffi_lib: Library,
    _norm_lib: Library,
    _gdn_prefill_sm90_lib: Library,
    _page_lib: Library,
    _tvmffi_get_version: TVMFFIGetVersionFn,
    tvmffi_env_set_stream: TVMFFIEnvSetStreamFn,
    tvmffi_error_move_from_raised: TVMFFIErrorMoveFromRaisedFn,
    tvmffi_object_dec_ref: TVMFFIObjectDecRefFn,
    tvmffi_any_view_to_owned_any: TVMFFIAnyViewToOwnedAnyFn,
    tvmffi_function_get_global: TVMFFIFunctionGetGlobalFn,
    tvmffi_function_call: TVMFFIFunctionCallFn,
    tvmffi_string_from_byte_array: TVMFFIStringFromByteArrayFn,
    tvm_ffi_rmsnorm: TVMFFISafeCallFn,
    tvm_ffi_gemma_rmsnorm: TVMFFISafeCallFn,
    tvm_ffi_gemma_fused_add_rmsnorm: TVMFFISafeCallFn,
    tvm_ffi_gdn_prefill: TVMFFISafeCallFn,
    tvm_ffi_append_paged_kv_cache: TVMFFISafeCallFn,
    tvm_ffi_append_paged_mla_kv_cache: TVMFFISafeCallFn,
    single_prefill_kernel_cache: Mutex<HashMap<String, LoadedKernel>>,
    batch_prefill_kernel_cache: Mutex<HashMap<String, LoadedBatchPrefillKernel>>,
    single_decode_kernel_cache: Mutex<HashMap<String, LoadedKernel>>,
    batch_decode_kernel_cache: Mutex<HashMap<String, LoadedBatchDecodeKernel>>,
    fused_moe_kernel_cache: Mutex<HashMap<String, LoadedFusedMoeKernel>>,
}

static GLOBAL_RUNTIME: OnceLock<FlashInferRuntime> = OnceLock::new();
static RUNTIME_INIT_LOCK: Mutex<()> = Mutex::new(());

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

    pub(crate) unsafe fn any_view_to_owned(
        &self,
        any_view: &TVMFFIAny,
    ) -> Result<TVMFFIAny, FlashInferError> {
        let mut owned = any_none();
        // SAFETY: symbol pointer and arguments follow TVM-FFI C ABI.
        let code = unsafe {
            (self.tvmffi_any_view_to_owned_any)(any_view as *const _, &mut owned as *mut _)
        };
        if code == 0 {
            return Ok(owned);
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

    unsafe fn load(resolved: ResolvedRuntimeConfig) -> Result<Self, FlashInferError> {
        let materialized_wheels = ensure_pinned_wheels_cached(&resolved.cache_dir)?;
        let artifacts = extract_artifacts(&resolved, &materialized_wheels)?;

        let tvmffi_lib = unsafe {
            Library::open(
                Some(&artifacts.tvmffi_so_path),
                libc::RTLD_NOW | libc::RTLD_GLOBAL,
            )
        }
        .map_err(|e| FlashInferError::LibraryLoad {
            library: artifacts.tvmffi_so_path.clone(),
            message: e.to_string(),
        })?;

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

        let tvmffi_any_view_to_owned_any: TVMFFIAnyViewToOwnedAnyFn = unsafe {
            resolve_symbol(
                &tvmffi_lib,
                &artifacts.tvmffi_so_path,
                b"TVMFFIAnyViewToOwnedAny\0",
                "TVMFFIAnyViewToOwnedAny",
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

        Ok(Self {
            resolved,
            jit_cache_wheel_path: materialized_wheels.jit_cache_wheel_path,
            artifact_dir: artifacts.artifact_dir,
            _tvmffi_lib: tvmffi_lib,
            _norm_lib: norm_lib,
            _gdn_prefill_sm90_lib: gdn_prefill_sm90_lib,
            _page_lib: page_lib,
            _tvmffi_get_version: tvmffi_get_version,
            tvmffi_env_set_stream,
            tvmffi_error_move_from_raised,
            tvmffi_object_dec_ref,
            tvmffi_any_view_to_owned_any,
            tvmffi_function_get_global,
            tvmffi_function_call,
            tvmffi_string_from_byte_array,
            tvm_ffi_rmsnorm,
            tvm_ffi_gemma_rmsnorm,
            tvm_ffi_gemma_fused_add_rmsnorm,
            tvm_ffi_gdn_prefill,
            tvm_ffi_append_paged_kv_cache,
            tvm_ffi_append_paged_mla_kv_cache,
            single_prefill_kernel_cache: Mutex::new(HashMap::new()),
            batch_prefill_kernel_cache: Mutex::new(HashMap::new()),
            single_decode_kernel_cache: Mutex::new(HashMap::new()),
            batch_decode_kernel_cache: Mutex::new(HashMap::new()),
            fused_moe_kernel_cache: Mutex::new(HashMap::new()),
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

fn ensure_pinned_wheels_cached(cache_dir: &Path) -> Result<MaterializedWheels, FlashInferError> {
    let wheels_dir = cache_dir.join(WHEEL_CACHE_DIR_NAME);
    fs::create_dir_all(&wheels_dir).map_err(|e| FlashInferError::CreateCacheDir {
        path: wheels_dir.clone(),
        source: e,
    })?;

    let jit_cache_wheel_path =
        ensure_pinned_wheel_cached(&wheels_dir, pinned_flashinfer_jit_cache_wheel())?;
    let tvmffi_wheel_path = ensure_pinned_wheel_cached(&wheels_dir, pinned_apache_tvm_ffi_wheel())?;

    Ok(MaterializedWheels {
        jit_cache_wheel_path,
        tvmffi_wheel_path,
    })
}

fn ensure_pinned_wheel_cached(
    wheels_dir: &Path,
    wheel: PinnedWheelMetadata<'_>,
) -> Result<PathBuf, FlashInferError> {
    ensure_pinned_wheel_cached_with_downloader(wheels_dir, wheel, download_pinned_wheel)
}

fn ensure_pinned_wheel_cached_with_downloader<F>(
    wheels_dir: &Path,
    wheel: PinnedWheelMetadata<'_>,
    mut downloader: F,
) -> Result<PathBuf, FlashInferError>
where
    F: FnMut(&Path, PinnedWheelMetadata<'_>) -> Result<(), FlashInferError>,
{
    cleanup_stale_download_temps(wheels_dir, wheel);
    let output_path = wheels_dir.join(format!("{}-{}", wheel.sha256_hex, wheel.filename));
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

    let mut needs_write = !output_path.exists();
    if !needs_write {
        let found = sha256_file_hex(&output_path, wheel.logical_name)?;
        needs_write = found != wheel.sha256_hex;
    }

    if needs_write {
        if output_path.exists() {
            fs::remove_file(&output_path).map_err(|e| FlashInferError::EmbeddedWheelCache {
                wheel: wheel.logical_name,
                path: output_path.clone(),
                source: e,
            })?;
        }
        downloader(&output_path, wheel)?;
    }

    let _ = lock_file.unlock();
    Ok(output_path)
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

    let mut hasher = Sha256::new();
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
            hasher.update(chunk);
        }
        out.sync_all()
            .map_err(|e| FlashInferError::EmbeddedWheelCache {
                wheel: wheel.logical_name,
                path: output_path.to_path_buf(),
                source: e,
            })?;
    }

    let found = format!("{:x}", hasher.finalize());
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

fn sha256_bytes_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
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
    use std::io::Cursor;
    use std::sync::Mutex;
    use std::time::Duration;

    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

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
    fn ensure_pinned_wheel_cached_creates_missing_file() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let wheels_dir = tmpdir.path().join("wheels");
        fs::create_dir_all(&wheels_dir).expect("create wheels dir");

        let bytes = b"wheel-bytes-v1";
        let sha = sha256_bytes_hex(bytes);
        let wheel = PinnedWheelMetadata {
            logical_name: "test_wheel",
            filename: "test.whl",
            url: "https://unused.invalid/test.whl",
            sha256_hex: &sha,
        };

        let output_path =
            ensure_pinned_wheel_cached_with_downloader(&wheels_dir, wheel, |output_path, wheel| {
                let mut reader = Cursor::new(bytes.as_slice());
                write_wheel_from_reader(&mut reader, output_path, wheel)
            })
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
        let wheel = PinnedWheelMetadata {
            logical_name: "test_wheel",
            filename: "reuse.whl",
            url: "https://unused.invalid/reuse.whl",
            sha256_hex: &sha,
        };

        let output_path =
            ensure_pinned_wheel_cached_with_downloader(&wheels_dir, wheel, |output_path, wheel| {
                let mut reader = Cursor::new(bytes.as_slice());
                write_wheel_from_reader(&mut reader, output_path, wheel)
            })
            .expect("cache wheel 1");
        let before = fs::metadata(&output_path)
            .expect("metadata 1")
            .modified()
            .expect("modified 1");
        std::thread::sleep(Duration::from_millis(1100));
        let output_path_2 = ensure_pinned_wheel_cached_with_downloader(
            &wheels_dir,
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
        let target_path = wheels_dir.join(format!("{sha}-rewrite.whl"));
        fs::write(&target_path, b"corrupt-data").expect("write corrupt");

        let wheel = PinnedWheelMetadata {
            logical_name: "test_wheel",
            filename: "rewrite.whl",
            url: "https://unused.invalid/rewrite.whl",
            sha256_hex: &sha,
        };

        let output_path =
            ensure_pinned_wheel_cached_with_downloader(&wheels_dir, wheel, |output_path, wheel| {
                let mut reader = Cursor::new(bytes.as_slice());
                write_wheel_from_reader(&mut reader, output_path, wheel)
            })
            .expect("cache wheel");
        let written = fs::read(&output_path).expect("read output");
        assert_eq!(written, bytes);
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
        let wheel = PinnedWheelMetadata {
            logical_name: "test_wheel",
            filename: "bad.whl",
            url: "https://unused.invalid/bad.whl",
            sha256_hex: &sha,
        };

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
    fn legacy_wheel_env_vars_are_ignored() {
        let _guard = ENV_TEST_LOCK.lock().expect("env lock");
        let prev_jit = env::var_os("FLASHINFER_RS_JIT_CACHE_WHEEL");
        let prev_tvm = env::var_os("FLASHINFER_RS_TVMFFI_WHEEL");
        let prev_cache = env::var_os(ENV_CACHE_DIR);

        unsafe {
            env::set_var("FLASHINFER_RS_JIT_CACHE_WHEEL", "/tmp/legacy-jit.whl");
            env::set_var("FLASHINFER_RS_TVMFFI_WHEEL", "/tmp/legacy-tvm.whl");
            env::remove_var(ENV_CACHE_DIR);
        }

        let cfg = RuntimeConfig::from_env().expect("from env");
        assert_eq!(cfg.cache_dir, None);

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
