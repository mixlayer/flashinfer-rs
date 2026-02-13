use std::collections::HashMap;
use std::env;
use std::ffi::c_void;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use fs2::FileExt;
use libloading::os::unix::Library;
use sha2::{Digest, Sha256};
use zip::ZipArchive;

use crate::error::FlashInferError;
use crate::ffi::{
    KDL_CUDA, TVMFFIAny, TVMFFIObjectHandle, TVMFFIVersion, any_none, byte_array_to_string,
    error_cell_ptr,
};

const ENV_JIT_CACHE_WHEEL: &str = "FLASHINFER_RS_JIT_CACHE_WHEEL";
const ENV_TVMFFI_WHEEL: &str = "FLASHINFER_RS_TVMFFI_WHEEL";
const ENV_CACHE_DIR: &str = "FLASHINFER_RS_CACHE_DIR";

const FLASHINFER_NORM_SO_SUFFIX: &str = "flashinfer_jit_cache/jit_cache/norm/norm.so";
const FLASHINFER_GDN_PREFILL_SM90_SO_SUFFIX: &str =
    "flashinfer_jit_cache/jit_cache/gdn_prefill_sm90/gdn_prefill_sm90.so";
const TVMFFI_SO_MEMBER: &str = "tvm_ffi/lib/libtvm_ffi.so";

const EXPECTED_TVMFFI_MAJOR: u32 = 0;
const EXPECTED_TVMFFI_MINOR: u32 = 1;

type TVMFFIGetVersionFn = unsafe extern "C" fn(*mut TVMFFIVersion);
type TVMFFIEnvSetStreamFn = unsafe extern "C" fn(i32, i32, *mut c_void, *mut *mut c_void) -> i32;
type TVMFFIErrorMoveFromRaisedFn = unsafe extern "C" fn(*mut TVMFFIObjectHandle);
type TVMFFIObjectDecRefFn = unsafe extern "C" fn(TVMFFIObjectHandle) -> i32;
type TVMFFIAnyViewToOwnedAnyFn = unsafe extern "C" fn(*const TVMFFIAny, *mut TVMFFIAny) -> i32;
type TVMFFISafeCallFn =
    unsafe extern "C" fn(*mut c_void, *const TVMFFIAny, i32, *mut TVMFFIAny) -> i32;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedRuntimeConfig {
    jit_cache_wheel: PathBuf,
    tvmffi_wheel: PathBuf,
    cache_dir: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub jit_cache_wheel: Option<PathBuf>,
    pub tvmffi_wheel: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
}

impl RuntimeConfig {
    pub fn from_env() -> Result<Self, FlashInferError> {
        Ok(Self {
            jit_cache_wheel: env_path(ENV_JIT_CACHE_WHEEL)?,
            tvmffi_wheel: env_path(ENV_TVMFFI_WHEEL)?,
            cache_dir: env_path(ENV_CACHE_DIR)?,
        })
    }

    pub fn with_jit_cache_wheel(mut self, path: impl Into<PathBuf>) -> Self {
        self.jit_cache_wheel = Some(path.into());
        self
    }

    pub fn with_tvmffi_wheel(mut self, path: impl Into<PathBuf>) -> Self {
        self.tvmffi_wheel = Some(path.into());
        self
    }

    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    fn resolve(&self) -> Result<ResolvedRuntimeConfig, FlashInferError> {
        let env_cfg = RuntimeConfig::from_env()?;

        let jit_cache_wheel = resolve_required_wheel(
            self.jit_cache_wheel
                .clone()
                .or(env_cfg.jit_cache_wheel)
                .as_ref(),
            "flashinfer_jit_cache",
            "flashinfer_jit_cache wheel",
            ENV_JIT_CACHE_WHEEL,
        )?;

        let tvmffi_wheel = resolve_required_wheel(
            self.tvmffi_wheel.clone().or(env_cfg.tvmffi_wheel).as_ref(),
            "apache_tvm_ffi",
            "apache_tvm_ffi wheel",
            ENV_TVMFFI_WHEEL,
        )?;

        let cache_dir = if let Some(path) = self.cache_dir.clone().or(env_cfg.cache_dir) {
            path
        } else {
            default_cache_dir()?
        };

        Ok(ResolvedRuntimeConfig {
            jit_cache_wheel,
            tvmffi_wheel,
            cache_dir,
        })
    }
}

#[derive(Debug)]
struct ExtractedArtifacts {
    artifact_dir: PathBuf,
    norm_so_path: PathBuf,
    gdn_prefill_sm90_so_path: PathBuf,
    tvmffi_so_path: PathBuf,
}

struct LoadedKernel {
    _lib: Library,
    run: TVMFFISafeCallFn,
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

pub struct FlashInferRuntime {
    resolved: ResolvedRuntimeConfig,
    artifact_dir: PathBuf,
    _tvmffi_lib: Library,
    _norm_lib: Library,
    _gdn_prefill_sm90_lib: Library,
    _tvmffi_get_version: TVMFFIGetVersionFn,
    tvmffi_env_set_stream: TVMFFIEnvSetStreamFn,
    tvmffi_error_move_from_raised: TVMFFIErrorMoveFromRaisedFn,
    tvmffi_object_dec_ref: TVMFFIObjectDecRefFn,
    tvmffi_any_view_to_owned_any: TVMFFIAnyViewToOwnedAnyFn,
    tvm_ffi_gemma_rmsnorm: TVMFFISafeCallFn,
    tvm_ffi_gdn_prefill: TVMFFISafeCallFn,
    single_prefill_kernel_cache: Mutex<HashMap<String, LoadedKernel>>,
    batch_prefill_kernel_cache: Mutex<HashMap<String, LoadedBatchPrefillKernel>>,
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

        let kernel_path = extract_jit_kernel(
            &self.resolved.jit_cache_wheel,
            &self.artifact_dir,
            kernel_uri,
        )?;

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

        let kernel_path = extract_jit_kernel(
            &self.resolved.jit_cache_wheel,
            &self.artifact_dir,
            kernel_uri,
        )?;

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

    unsafe fn load(resolved: ResolvedRuntimeConfig) -> Result<Self, FlashInferError> {
        let artifacts = extract_artifacts(&resolved)?;

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

        let tvm_ffi_gemma_rmsnorm: TVMFFISafeCallFn = unsafe {
            resolve_symbol(
                &norm_lib,
                &artifacts.norm_so_path,
                b"__tvm_ffi_gemma_rmsnorm\0",
                "__tvm_ffi_gemma_rmsnorm",
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
            artifact_dir: artifacts.artifact_dir,
            _tvmffi_lib: tvmffi_lib,
            _norm_lib: norm_lib,
            _gdn_prefill_sm90_lib: gdn_prefill_sm90_lib,
            _tvmffi_get_version: tvmffi_get_version,
            tvmffi_env_set_stream,
            tvmffi_error_move_from_raised,
            tvmffi_object_dec_ref,
            tvmffi_any_view_to_owned_any,
            tvm_ffi_gemma_rmsnorm,
            tvm_ffi_gdn_prefill,
            single_prefill_kernel_cache: Mutex::new(HashMap::new()),
            batch_prefill_kernel_cache: Mutex::new(HashMap::new()),
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
            &resolved.jit_cache_wheel,
            FLASHINFER_NORM_SO_SUFFIX,
            &norm_so_path,
        )?;
    }

    let gdn_prefill_sm90_so_path = artifact_dir.join("gdn_prefill_sm90.so");
    if !gdn_prefill_sm90_so_path.exists() {
        extract_member_from_wheel_by_suffix(
            &resolved.jit_cache_wheel,
            FLASHINFER_GDN_PREFILL_SM90_SO_SUFFIX,
            &gdn_prefill_sm90_so_path,
        )?;
    }

    let tvmffi_so_path = artifact_dir.join("libtvm_ffi.so");
    if !tvmffi_so_path.exists() {
        extract_member_from_wheel_exact(&resolved.tvmffi_wheel, TVMFFI_SO_MEMBER, &tvmffi_so_path)?;
    }

    let _ = lock_file.unlock();

    Ok(ExtractedArtifacts {
        artifact_dir,
        norm_so_path,
        gdn_prefill_sm90_so_path,
        tvmffi_so_path,
    })
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

    let artifact_hash = artifact_hash(&resolved.jit_cache_wheel, &resolved.tvmffi_wheel)?;
    let artifact_dir = resolved.cache_dir.join(artifact_hash);
    fs::create_dir_all(&artifact_dir).map_err(|e| FlashInferError::CreateCacheDir {
        path: artifact_dir.clone(),
        source: e,
    })?;
    Ok(artifact_dir)
}

fn artifact_hash(jit_cache_wheel: &Path, tvmffi_wheel: &Path) -> Result<String, FlashInferError> {
    let mut hasher = Sha256::new();
    hash_file_fingerprint(&mut hasher, jit_cache_wheel)?;
    hash_file_fingerprint(&mut hasher, tvmffi_wheel)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_file_fingerprint(hasher: &mut Sha256, path: &Path) -> Result<(), FlashInferError> {
    let metadata = fs::metadata(path).map_err(|e| FlashInferError::Metadata {
        path: path.to_path_buf(),
        source: e,
    })?;

    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(metadata.len().to_le_bytes());

    let modified = metadata
        .modified()
        .map_err(|e| FlashInferError::Metadata {
            path: path.to_path_buf(),
            source: e,
        })?
        .duration_since(UNIX_EPOCH)
        .map_err(|e| {
            FlashInferError::invalid_argument(format!(
                "invalid modified time for `{}`: {e}",
                path.display()
            ))
        })?;
    hasher.update(modified.as_secs().to_le_bytes());
    hasher.update(modified.subsec_nanos().to_le_bytes());
    Ok(())
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

fn resolve_required_wheel(
    explicit: Option<&PathBuf>,
    filename_prefix: &str,
    what: &'static str,
    env_name: &'static str,
) -> Result<PathBuf, FlashInferError> {
    if let Some(path) = explicit {
        return ensure_existing_file(path.clone(), what);
    }

    let mut candidates = Vec::new();
    gather_matching_wheels(
        &mut candidates,
        &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        filename_prefix,
    );
    gather_matching_wheels(
        &mut candidates,
        Path::new(env!("CARGO_MANIFEST_DIR")),
        filename_prefix,
    );

    candidates.sort();
    candidates.dedup();

    if let Some(path) = candidates.into_iter().next() {
        return ensure_existing_file(path, what);
    }

    Err(FlashInferError::ArtifactNotFound {
        what,
        hint: format!(
            "set `{env_name}` or place a `{filename_prefix}-*.whl` file in the current directory"
        ),
    })
}

fn ensure_existing_file(path: PathBuf, what: &'static str) -> Result<PathBuf, FlashInferError> {
    if !path.exists() {
        return Err(FlashInferError::ArtifactNotFound {
            what,
            hint: format!("file `{}` does not exist", path.display()),
        });
    }
    if !path.is_file() {
        return Err(FlashInferError::invalid_argument(format!(
            "`{}` is not a file",
            path.display()
        )));
    }
    Ok(path)
}

fn gather_matching_wheels(target: &mut Vec<PathBuf>, dir: &Path, prefix: &str) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with(prefix) && name.ends_with(".whl") {
            target.push(path);
        }
    }
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

    #[test]
    fn env_path_empty_is_error() {
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
    fn hash_changes_when_file_changes() {
        let tmpdir = tempfile::tempdir().expect("tempdir");
        let a = tmpdir.path().join("a.whl");
        let b = tmpdir.path().join("b.whl");
        fs::write(&a, b"one").expect("write a");
        fs::write(&b, b"two").expect("write b");

        let h1 = artifact_hash(&a, &b).expect("hash 1");
        fs::write(&a, b"three").expect("write a2");
        let h2 = artifact_hash(&a, &b).expect("hash 2");
        assert_ne!(h1, h2);
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
