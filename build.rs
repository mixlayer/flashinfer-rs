use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use fs2::FileExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const MANIFEST_FILE: &str = "Cargo.toml";
const EMBEDDED_WHEELS_RS: &str = "embedded_wheels.rs";
const PINNED_DIR_NAME: &str = "pinned";
const DOWNLOAD_BUFFER_SIZE: usize = 1024 * 1024;

fn main() {
    if let Err(error) = run() {
        panic!("build script failed: {error}");
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed={MANIFEST_FILE}");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let manifest_path = manifest_dir.join(MANIFEST_FILE);

    let pinned = parse_pinned_wheels(&manifest_path)?;
    let target_arch = current_target_arch()?;
    let flashinfer_wheel = select_wheel_metadata(
        "flashinfer_jit_cache",
        &pinned.flashinfer_jit_cache,
        &target_arch,
    )?;
    let tvmffi_wheel =
        select_wheel_metadata("apache_tvm_ffi", &pinned.apache_tvm_ffi, &target_arch)?;

    let cache_dir = build_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;

    let cached_flashinfer =
        ensure_cached_wheel(&cache_dir, "flashinfer_jit_cache", flashinfer_wheel)?;
    let cached_tvmffi = ensure_cached_wheel(&cache_dir, "apache_tvm_ffi", tvmffi_wheel)?;

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let pinned_out_dir = out_dir.join(PINNED_DIR_NAME);
    fs::create_dir_all(&pinned_out_dir)?;

    let staged_flashinfer =
        stage_wheel_for_include(&cached_flashinfer, &pinned_out_dir, flashinfer_wheel)?;
    let staged_tvmffi = stage_wheel_for_include(&cached_tvmffi, &pinned_out_dir, tvmffi_wheel)?;

    let module_path = out_dir.join(EMBEDDED_WHEELS_RS);
    write_embedded_wheels_module(
        &module_path,
        flashinfer_wheel,
        &staged_flashinfer,
        tvmffi_wheel,
        &staged_tvmffi,
    )?;

    Ok(())
}

#[derive(Debug, Deserialize)]
struct ManifestToml {
    package: ManifestPackage,
}

#[derive(Debug, Deserialize)]
struct ManifestPackage {
    metadata: Option<ManifestMetadata>,
}

#[derive(Debug, Deserialize)]
struct ManifestMetadata {
    flashinfer_rs: Option<FlashInferRsMetadata>,
}

#[derive(Debug, Deserialize)]
struct FlashInferRsMetadata {
    pinned_wheels: Option<PinnedWheelsMetadata>,
}

#[derive(Debug, Deserialize)]
struct PinnedWheelsMetadata {
    flashinfer_jit_cache: ArchWheelMetadata,
    apache_tvm_ffi: ArchWheelMetadata,
}

#[derive(Debug, Deserialize)]
struct ArchWheelMetadata {
    x86_64: WheelMetadata,
    aarch64: WheelMetadata,
}

#[derive(Debug, Clone, Deserialize)]
struct WheelMetadata {
    filename: String,
    url: String,
    sha256: String,
}

fn parse_pinned_wheels(
    manifest_path: &Path,
) -> Result<PinnedWheelsMetadata, Box<dyn std::error::Error>> {
    let manifest = fs::read_to_string(manifest_path)?;
    let parsed: ManifestToml = toml::from_str(&manifest)?;
    let pinned = parsed
        .package
        .metadata
        .and_then(|m| m.flashinfer_rs)
        .and_then(|m| m.pinned_wheels)
        .ok_or("missing [package.metadata.flashinfer_rs.pinned_wheels] configuration")?;

    validate_arch_wheel_metadata("flashinfer_jit_cache", &pinned.flashinfer_jit_cache)?;
    validate_arch_wheel_metadata("apache_tvm_ffi", &pinned.apache_tvm_ffi)?;

    Ok(pinned)
}

fn validate_arch_wheel_metadata(
    name: &'static str,
    wheels: &ArchWheelMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_wheel_metadata(&format!("{name}.x86_64"), &wheels.x86_64)?;
    validate_wheel_metadata(&format!("{name}.aarch64"), &wheels.aarch64)?;
    Ok(())
}

fn validate_wheel_metadata(
    name: &str,
    wheel: &WheelMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    if wheel.filename.trim().is_empty() {
        return Err(format!("metadata for `{name}` has empty filename").into());
    }
    if wheel.url.trim().is_empty() {
        return Err(format!("metadata for `{name}` has empty url").into());
    }
    if wheel.sha256.len() != 64 || !wheel.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!(
            "metadata for `{name}` has invalid sha256 `{}` (expected 64 hex chars)",
            wheel.sha256
        )
        .into());
    }
    Ok(())
}

fn current_target_arch() -> Result<String, Box<dyn std::error::Error>> {
    let arch = env::var("CARGO_CFG_TARGET_ARCH")
        .unwrap_or_else(|_| env::consts::ARCH.to_string())
        .trim()
        .to_string();
    if arch.is_empty() {
        return Err("unable to resolve target architecture".into());
    }
    Ok(arch)
}

fn select_wheel_metadata<'a>(
    wheel_name: &'static str,
    wheels: &'a ArchWheelMetadata,
    target_arch: &str,
) -> Result<&'a WheelMetadata, Box<dyn std::error::Error>> {
    match target_arch {
        "x86_64" => Ok(&wheels.x86_64),
        "aarch64" => Ok(&wheels.aarch64),
        _ => Err(format!(
            "unsupported target architecture `{target_arch}` for `{wheel_name}`; supported: x86_64, aarch64"
        )
        .into()),
    }
}

fn build_cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(home) = env::var_os("HOME") {
        return Ok(PathBuf::from(home)
            .join(".cache")
            .join("flashinfer-rs")
            .join("build-wheels"));
    }

    if let Some(xdg) = env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(xdg)
            .join("flashinfer-rs")
            .join("build-wheels"));
    }

    Err("unable to resolve build cache directory from HOME or XDG_CACHE_HOME".into())
}

fn ensure_cached_wheel(
    cache_dir: &Path,
    wheel_name: &'static str,
    wheel: &WheelMetadata,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    cleanup_stale_download_temps(cache_dir, wheel);
    let cache_path = cache_dir.join(format!("{}-{}", wheel.sha256, wheel.filename));

    let lock_path = cache_path.with_extension("lock");
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)?;
    lock_file.lock_exclusive()?;

    if cache_path.exists() {
        let found = sha256_file_hex(&cache_path)?;
        if found == wheel.sha256 {
            let _ = lock_file.unlock();
            return Ok(cache_path);
        }
        fs::remove_file(&cache_path)?;
    }

    download_wheel(cache_dir, &cache_path, wheel_name, wheel)?;
    let _ = lock_file.unlock();
    Ok(cache_path)
}

fn cleanup_stale_download_temps(cache_dir: &Path, wheel: &WheelMetadata) {
    let Ok(entries) = fs::read_dir(cache_dir) else {
        return;
    };
    let modern_prefix = format!("{}-{}.tmp-", wheel.sha256, wheel.filename);
    let legacy_prefix = format!(
        "{}-{}.tmp-",
        wheel.sha256,
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

fn download_wheel(
    cache_dir: &Path,
    cache_path: &Path,
    wheel_name: &'static str,
    wheel: &WheelMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(cache_dir)?;

    let tmp_path = cache_path.with_extension("tmp");
    if tmp_path.exists() {
        let _ = fs::remove_file(&tmp_path);
    }

    let response = ureq::get(&wheel.url).call().map_err(|e| {
        format!(
            "failed to download `{wheel_name}` wheel from `{}`: {e}",
            wheel.url
        )
    })?;

    let mut reader = response.into_reader();
    let mut file = File::create(&tmp_path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_SIZE];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let chunk = &buffer[..read];
        file.write_all(chunk)?;
        hasher.update(chunk);
    }
    file.sync_all()?;

    let found = format!("{:x}", hasher.finalize());
    if found != wheel.sha256 {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!(
            "sha256 mismatch for downloaded `{wheel_name}` wheel: expected {}, found {found}",
            wheel.sha256
        )
        .into());
    }

    match fs::rename(&tmp_path, cache_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
            let _ = fs::remove_file(&tmp_path);
            let existing = sha256_file_hex(cache_path)?;
            if existing == wheel.sha256 {
                Ok(())
            } else {
                Err(format!(
                    "cached wheel exists with unexpected checksum at `{}`: expected {}, found {existing}",
                    cache_path.display(),
                    wheel.sha256
                )
                .into())
            }
        }
        Err(err) => {
            let _ = fs::remove_file(&tmp_path);
            Err(err.into())
        }
    }
}

fn stage_wheel_for_include(
    cached_path: &Path,
    pinned_out_dir: &Path,
    wheel: &WheelMetadata,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = pinned_out_dir.join(&wheel.filename);
    if output_path.exists() {
        let found = sha256_file_hex(&output_path)?;
        if found == wheel.sha256 {
            return Ok(output_path);
        }
    }

    let tmp_path = output_path.with_extension("tmp");
    if tmp_path.exists() {
        let _ = fs::remove_file(&tmp_path);
    }

    fs::copy(cached_path, &tmp_path)?;
    let copied_sha = sha256_file_hex(&tmp_path)?;
    if copied_sha != wheel.sha256 {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!(
            "sha256 mismatch after staging wheel `{}` to `{}`: expected {}, found {copied_sha}",
            wheel.filename,
            output_path.display(),
            wheel.sha256
        )
        .into());
    }

    match fs::rename(&tmp_path, &output_path) {
        Ok(()) => Ok(output_path),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
            let _ = fs::remove_file(&tmp_path);
            let existing = sha256_file_hex(&output_path)?;
            if existing == wheel.sha256 {
                Ok(output_path)
            } else {
                Err(format!(
                    "staged wheel `{}` exists with unexpected checksum: expected {}, found {existing}",
                    output_path.display(),
                    wheel.sha256
                )
                .into())
            }
        }
        Err(err) => {
            let _ = fs::remove_file(&tmp_path);
            Err(err.into())
        }
    }
}

fn write_embedded_wheels_module(
    output_path: &Path,
    flashinfer: &WheelMetadata,
    flashinfer_staged_path: &Path,
    tvmffi: &WheelMetadata,
    tvmffi_staged_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let flashinfer_path = fs::canonicalize(flashinfer_staged_path)?;
    let tvmffi_path = fs::canonicalize(tvmffi_staged_path)?;

    let source = format!(
        "// @generated by build.rs.\n\
pub(crate) const PINNED_FLASHINFER_JIT_CACHE_WHEEL_FILENAME: &str = {flashinfer_filename};\n\
pub(crate) const PINNED_FLASHINFER_JIT_CACHE_WHEEL_SHA256: &str = {flashinfer_sha};\n\
pub(crate) static PINNED_FLASHINFER_JIT_CACHE_WHEEL_BYTES: &[u8] = include_bytes!({flashinfer_path});\n\
\n\
pub(crate) const PINNED_APACHE_TVM_FFI_WHEEL_FILENAME: &str = {tvmffi_filename};\n\
pub(crate) const PINNED_APACHE_TVM_FFI_WHEEL_SHA256: &str = {tvmffi_sha};\n\
pub(crate) static PINNED_APACHE_TVM_FFI_WHEEL_BYTES: &[u8] = include_bytes!({tvmffi_path});\n",
        flashinfer_filename = rust_str_literal(&flashinfer.filename),
        flashinfer_sha = rust_str_literal(&flashinfer.sha256),
        flashinfer_path = rust_str_literal(flashinfer_path.to_string_lossy().as_ref()),
        tvmffi_filename = rust_str_literal(&tvmffi.filename),
        tvmffi_sha = rust_str_literal(&tvmffi.sha256),
        tvmffi_path = rust_str_literal(tvmffi_path.to_string_lossy().as_ref()),
    );

    fs::write(output_path, source)?;
    Ok(())
}

fn rust_str_literal(value: &str) -> String {
    format!("{value:?}")
}

fn sha256_file_hex(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; DOWNLOAD_BUFFER_SIZE];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
