use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FlashInferError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("invalid environment variable `{name}`: {message}")]
    InvalidEnvironment { name: &'static str, message: String },

    #[error("failed to open wheel `{wheel}`")]
    WheelOpen {
        wheel: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("entry `{entry}` not found in wheel `{wheel}`")]
    WheelEntryMissing { wheel: PathBuf, entry: String },

    #[error("failed to read wheel `{wheel}`")]
    WheelRead {
        wheel: PathBuf,
        #[source]
        source: zip::result::ZipError,
    },

    #[error("failed to extract `{entry}` from wheel `{wheel}` to `{output}`")]
    WheelExtract {
        wheel: PathBuf,
        entry: String,
        output: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to load shared library `{library}`: {message}")]
    LibraryLoad { library: PathBuf, message: String },

    #[error("failed to resolve symbol `{symbol}` from `{library}`: {message}")]
    SymbolResolve {
        library: PathBuf,
        symbol: &'static str,
        message: String,
    },

    #[error(
        "unsupported TVM-FFI ABI version {found_major}.{found_minor}.{found_patch}; expected 0.1.x"
    )]
    AbiVersionMismatch {
        found_major: u32,
        found_minor: u32,
        found_patch: u32,
    },

    #[error("TVM-FFI call failed (code {code}) kind={kind} message={message}{backtrace_suffix}")]
    TvmFfiCall {
        code: i32,
        kind: String,
        message: String,
        backtrace_suffix: String,
    },

    #[error("TVM-FFI did not provide raised error details (code {code})")]
    TvmFfiCallNoDetails { code: i32 },

    #[error("failed to set CUDA stream context for device {device_id} (code {code})")]
    StreamSet { device_id: i32, code: i32 },

    #[error("failed to restore CUDA stream context for device {device_id} (code {code})")]
    StreamRestore { device_id: i32, code: i32 },

    #[error("failed to create cache directory `{path}`")]
    CreateCacheDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to lock extraction file `{path}`")]
    CacheLock {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("embedded wheel cache I/O failed for `{wheel}` at `{path}`")]
    EmbeddedWheelCache {
        wheel: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "embedded wheel checksum mismatch for `{wheel}` at `{path}`: expected {expected}, found {found}"
    )]
    EmbeddedWheelChecksumMismatch {
        wheel: &'static str,
        path: PathBuf,
        expected: String,
        found: String,
    },

    #[error("runtime has already been initialized with a different configuration")]
    RuntimeAlreadyInitialized,
}

impl FlashInferError {
    pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    pub(crate) fn tvm_ffi_call(
        code: i32,
        kind: String,
        message: String,
        backtrace: String,
    ) -> Self {
        let backtrace_suffix = if backtrace.is_empty() {
            String::new()
        } else {
            format!(" backtrace={backtrace}")
        };
        Self::TvmFfiCall {
            code,
            kind,
            message,
            backtrace_suffix,
        }
    }
}
