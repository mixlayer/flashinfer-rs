# flashinfer-rs

Rust-first integration for calling precompiled FlashInfer kernels through TVM-FFI, without a C++ shim.

## Current Scope

- `gemma_rmsnorm` from `norm.so`
- `gdn_prefill` from `gdn_prefill_sm90.so` (SM90A path)
- Pure Rust TVM-FFI ABI packing and dynamic loading
- Optional `cudarc` convenience wrappers

This repository is currently optimized for local/internal Linux x86_64 deployment.

## Requirements

- NVIDIA GPU + CUDA runtime compatible with the selected wheel set
- `libcudart.so.13` available at runtime for the CUDA 13 wheel path
- Rust toolchain (edition 2024)

## Artifact Sources

TVM-FFI wheel source (`libtvm_ffi.so` lives inside this wheel):

- https://pypi.org/project/apache-tvm-ffi/
- https://pypi.org/simple/apache-tvm-ffi/

FlashInfer prebuilt JIT cache wheels:

- Stable CUDA 13.0 index: https://flashinfer.ai/whl/cu130/
- Stable wheel list: https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/
- Nightly CUDA 13.0 index: https://flashinfer.ai/whl/nightly/cu130/
- Nightly wheel list: https://flashinfer.ai/whl/nightly/cu130/flashinfer-jit-cache/

Also see FlashInfer installation docs:

- https://docs.flashinfer.ai/installation.html

## Artifact Discovery and Cache

By default the runtime searches for wheels by filename prefix in the current directory:

- `flashinfer_jit_cache-*.whl`
- `apache_tvm_ffi-*.whl`

Or set explicit paths:

- `FLASHINFER_RS_JIT_CACHE_WHEEL`
- `FLASHINFER_RS_TVMFFI_WHEEL`
- `FLASHINFER_RS_CACHE_DIR`

Extracted libraries are cached under:

- `~/.cache/flashinfer-rs/<artifact-hash>/`

## Quick Start

1. Place both wheel files in the repo directory (or set env vars above).
2. Initialize runtime once:

```rust
use flashinfer_rs::{FlashInferRuntime, RuntimeConfig};

let _rt = FlashInferRuntime::initialize(RuntimeConfig::default())?;
```

3. Call kernels through typed APIs.

## API Example: Gemma RMSNorm

```rust
use flashinfer_rs::{
    DType, GemmaRmsNormParams, Tensor1DDesc, Tensor2DDesc, gemma_rmsnorm,
};
use std::ffi::c_void;

let params = GemmaRmsNormParams::new(
    Tensor2DDesc {
        ptr: input_ptr as *const c_void, // device ptr
        rows,
        cols,
        stride_row: cols,
        stride_col: 1,
        dtype: DType::F16,
        device_id: 0,
    },
    Tensor1DDesc {
        ptr: weight_ptr as *const c_void, // device ptr
        len: cols,
        stride: 1,
        dtype: DType::F16,
        device_id: 0,
    },
    Tensor2DDesc {
        ptr: out_ptr as *const c_void, // device ptr
        rows,
        cols,
        stride_row: cols,
        stride_col: 1,
        dtype: DType::F16,
        device_id: 0,
    },
    1e-6,
    stream_ptr, // cudaStream_t as *mut c_void
);

gemma_rmsnorm(&params)?;
```

## API Example: SM90 Prefill (`cudarc`)

```rust
use flashinfer_rs::{DType, gdn_prefill_sm90_cudarc_with_options};

gdn_prefill_sm90_cudarc_with_options(
    stream.as_ref(),
    &mut output,
    &mut output_state,
    &q,
    &k,
    &v,
    &cu_seqlens,
    &mut workspace,
    packed_seq,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    head_size,
    DType::F16,
    None,         // input_state: Option<&IS>
    None,         // alpha: Option<&A>
    None,         // beta: Option<&B>
    0.0,          // scale (0.0 means kernel default behavior)
)?;
```

## Architecture Handling

FlashInfer host wrappers dispatch to architecture-specific kernels at runtime.

- This repo currently loads `gdn_prefill_sm90.so` and calls `__tvm_ffi_gdn_prefill`.
- The current launcher path is SM90A-only for this kernel family.
- On unsupported GPUs, calls fail with decoded TVM-FFI errors (`FlashInferError::TvmFfiCall`).

## Testing

```bash
cargo test
cargo test --features cudarc
FLASHINFER_RS_RUN_GPU_TESTS=1 cargo test --features cudarc --test gemma_rmsnorm_gpu
```

## Additional Notes

- Calls are asynchronous with respect to host execution (no implicit stream synchronize).
- Dynamic loading order is: `libtvm_ffi.so` -> `norm.so` -> `gdn_prefill_sm90.so`.
- Integration details: `docs/flashinfer-rs-integration.md`.
