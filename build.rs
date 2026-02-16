//! Build script for pinning and embedding FlashInfer runtime wheel artifacts.
//!
//! High-level flow:
//! 1. Read pinned wheel metadata from `Cargo.toml`, keyed by CUDA variant
//!    (`cu130`/`cu131`) and architecture (`x86_64`/`aarch64`).
//! 2. Detect build-host CUDA toolkit version (supports CUDA 13.0 and 13.1) and
//!    detect Cargo target architecture.
//! 3. Select matching FlashInfer and Apache TVM-FFI wheel entries.
//! 4. Download wheels into a persistent build cache (`~/.cache/flashinfer-rs/build-wheels`),
//!    reusing existing files when checksums match.
//! 5. Verify SHA256 for all wheel bytes before use.
//! 6. Stage verified wheels into `OUT_DIR/pinned` and generate
//!    `OUT_DIR/embedded_wheels.rs`, which exposes `include_bytes!` constants used
//!    by runtime initialization.
//!
//! The build fails if required wheels cannot be detected, fetched, or verified,
//! ensuring the crate remains pinned to explicit, validated artifacts.

use std::collections::HashSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

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
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let manifest_path = manifest_dir.join(MANIFEST_FILE);

    let pinned = parse_pinned_wheels(&manifest_path)?;
    let target_arch = current_target_arch()?;
    let cuda_tag = detect_build_host_cuda_tag()?;
    let flashinfer_wheel = select_wheel_metadata(
        "flashinfer_jit_cache",
        &pinned.flashinfer_jit_cache,
        cuda_tag,
        &target_arch,
    )?;
    let tvmffi_wheel = select_wheel_metadata(
        "apache_tvm_ffi",
        &pinned.apache_tvm_ffi,
        cuda_tag,
        &target_arch,
    )?;

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
    flashinfer_jit_cache: CudaWheelMetadata,
    apache_tvm_ffi: CudaWheelMetadata,
}

#[derive(Debug, Deserialize)]
struct CudaWheelMetadata {
    cu130: ArchWheelMetadata,
    cu131: ArchWheelMetadata,
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

#[derive(Clone, Copy, Debug)]
enum CudaTag {
    Cu130,
    Cu131,
}

impl CudaTag {
    fn metadata_key(self) -> &'static str {
        match self {
            Self::Cu130 => "cu130",
            Self::Cu131 => "cu131",
        }
    }
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

    validate_cuda_wheel_metadata("flashinfer_jit_cache", &pinned.flashinfer_jit_cache)?;
    validate_cuda_wheel_metadata("apache_tvm_ffi", &pinned.apache_tvm_ffi)?;

    Ok(pinned)
}

fn validate_cuda_wheel_metadata(
    name: &'static str,
    wheels: &CudaWheelMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_arch_wheel_metadata(&format!("{name}.cu130"), &wheels.cu130)?;
    validate_arch_wheel_metadata(&format!("{name}.cu131"), &wheels.cu131)?;
    Ok(())
}

fn validate_arch_wheel_metadata(
    name: &str,
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
    wheels: &'a CudaWheelMetadata,
    cuda_tag: CudaTag,
    target_arch: &str,
) -> Result<&'a WheelMetadata, Box<dyn std::error::Error>> {
    let by_arch = match cuda_tag {
        CudaTag::Cu130 => &wheels.cu130,
        CudaTag::Cu131 => &wheels.cu131,
    };

    match target_arch {
        "x86_64" => Ok(&by_arch.x86_64),
        "aarch64" => Ok(&by_arch.aarch64),
        _ => Err(format!(
            "unsupported target architecture `{target_arch}` for `{wheel_name}` {cuda}; supported: x86_64, aarch64",
            cuda = cuda_tag.metadata_key(),
        )
        .into()),
    }
}

fn detect_build_host_cuda_tag() -> Result<CudaTag, Box<dyn std::error::Error>> {
    if let Some((version, source)) = detect_cuda_version_from_version_json()? {
        if let Some(tag) = cuda_tag_from_version(&version) {
            return Ok(tag);
        }
        return Err(format!(
            "unsupported CUDA version `{version}` detected via `{source}`; supported build-host versions: 13.0, 13.1"
        )
        .into());
    }

    if let Some(version) = detect_cuda_version_from_nvcc()? {
        if let Some(tag) = cuda_tag_from_version(&version) {
            return Ok(tag);
        }
        return Err(format!(
            "unsupported CUDA version `{version}` detected from `nvcc --version`; supported build-host versions: 13.0, 13.1"
        )
        .into());
    }

    Err("unable to detect CUDA build-host version; expected CUDA version.json under CUDA_HOME/CUDA_PATH/CUDA_ROOT or /usr/local/cuda, or a working `nvcc --version`".into())
}

fn detect_cuda_version_from_version_json()
-> Result<Option<(String, String)>, Box<dyn std::error::Error>> {
    for path in cuda_version_json_candidates() {
        if !path.is_file() {
            continue;
        }

        let content = fs::read_to_string(&path).map_err(|e| {
            format!(
                "failed reading CUDA version metadata from `{}`: {e}",
                path.display()
            )
        })?;
        if let Some(version) = extract_cuda_version_from_version_json(&content) {
            return Ok(Some((version, path.display().to_string())));
        }
    }
    Ok(None)
}

fn detect_cuda_version_from_nvcc() -> Result<Option<String>, Box<dyn std::error::Error>> {
    let output = match Command::new("nvcc").arg("--version").output() {
        Ok(output) => output,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            println!("cargo:warning=failed to run `nvcc --version`: {err}");
            return Ok(None);
        }
    };

    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.stderr.is_empty() {
        text.push('\n');
        text.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    Ok(extract_cuda_version_from_nvcc_output(&text))
}

fn cuda_version_json_candidates() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for key in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Some(value) = env::var_os(key) {
            let value = PathBuf::from(value);
            if !value.as_os_str().is_empty() {
                roots.push(value);
            }
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for root in roots {
        let candidate = root.join("version.json");
        if seen.insert(candidate.clone()) {
            out.push(candidate);
        }
    }
    out
}

fn extract_cuda_version_from_version_json(content: &str) -> Option<String> {
    if let Some(cuda_pos) = content.find("\"cuda\"") {
        let in_cuda_object = &content[cuda_pos..];
        if let Some(version) = extract_json_string_value(in_cuda_object, "version") {
            return Some(version);
        }
    }
    extract_json_string_value(content, "version")
}

fn extract_json_string_value(content: &str, key: &str) -> Option<String> {
    let key_marker = format!("\"{key}\"");
    let key_idx = content.find(&key_marker)?;
    let after_key = &content[key_idx + key_marker.len()..];
    let colon_idx = after_key.find(':')?;
    let value = after_key[colon_idx + 1..].trim_start();
    if !value.starts_with('"') {
        return None;
    }
    let value = &value[1..];
    let end_quote = value.find('"')?;
    Some(value[..end_quote].to_string())
}

fn extract_cuda_version_from_nvcc_output(output: &str) -> Option<String> {
    if let Some(version) = extract_version_after_marker(output, "release ") {
        return Some(version);
    }
    extract_version_after_marker(output, "V")
}

fn extract_version_after_marker(text: &str, marker: &str) -> Option<String> {
    for (index, _) in text.match_indices(marker) {
        let tail = &text[index + marker.len()..];
        let version: String = tail
            .chars()
            .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
            .collect();
        if version.is_empty() {
            continue;
        }
        if version.chars().any(|ch| ch == '.') {
            return Some(version);
        }
    }
    None
}

fn cuda_tag_from_version(version: &str) -> Option<CudaTag> {
    let mut components = version.split('.');
    let major = components.next()?.parse::<u32>().ok()?;
    let minor = components
        .next()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0);

    match (major, minor) {
        (13, 0) => Some(CudaTag::Cu130),
        (13, 1) => Some(CudaTag::Cu131),
        _ => None,
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
