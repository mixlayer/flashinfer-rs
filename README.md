# flashinfer-rs

Rust-first integration for calling precompiled FlashInfer kernels through TVM-FFI, without a C++ shim.

## Current Scope

- `gemma_rmsnorm` from `norm.so`
- `rmsnorm` (2D) and fused QK RMSNorm (3D) from `norm.so`
- `gdn_prefill` from `gdn_prefill_sm90.so` (SM90A path)
- MHA single prefill (`single_prefill_with_kv_cache`) via on-demand JIT-cache module loading
- MHA batched ragged/paged prefill (`batch_prefill_with_kv_cache`) via on-demand JIT-cache module loading
- MHA single decode (`single_decode_with_kv_cache`) via on-demand JIT-cache module loading
- MHA batched paged decode (`batch_decode_with_kv_cache`) via on-demand JIT-cache module loading
- Paged KV append (`append_paged_kv_cache`) and paged MLA KV append (`append_paged_mla_kv_cache`) from `page.so`
- Pure Rust TVM-FFI ABI packing and dynamic loading
- Optional `cudarc` convenience wrappers

This repository is currently optimized for local/internal Linux x86_64 deployment.

## Requirements

- NVIDIA GPU + CUDA runtime compatible with the selected wheel set
- `libcudart.so.13` available at runtime for the CUDA 13 wheel path
- Rust toolchain (edition 2024)
- Network access for the first build (the build script downloads pinned wheels)

## Artifact Sources

TVM-FFI wheel source (`libtvm_ffi.so` lives inside this wheel):

- https://pypi.org/project/apache-tvm-ffi/
- https://pypi.org/simple/apache-tvm-ffi/

FlashInfer prebuilt JIT cache wheels:

- Stable CUDA 13.0 index: https://flashinfer.ai/whl/cu130/
- Stable wheel list: https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/
- Nightly CUDA 13.0 index: https://flashinfer.ai/whl/nightly/cu130/
- Nightly wheel list: https://flashinfer.ai/whl/nightly/cu130/flashinfer-jit-cache/

This crate supports pinned wheel selection for CUDA 13.0 (`cu130`) and CUDA 13.1 (`cu131`) metadata keys. At build time, the CUDA version is auto-detected from the build host.

Also see FlashInfer installation docs:

- https://docs.flashinfer.ai/installation.html

## Pinned Wheels and Cache

This crate now pins wheels at build time via `Cargo.toml` metadata:

- `[package.metadata.flashinfer_rs.pinned_wheels.flashinfer_jit_cache.cu130.x86_64]`
- `[package.metadata.flashinfer_rs.pinned_wheels.flashinfer_jit_cache.cu130.aarch64]`
- `[package.metadata.flashinfer_rs.pinned_wheels.flashinfer_jit_cache.cu131.x86_64]`
- `[package.metadata.flashinfer_rs.pinned_wheels.flashinfer_jit_cache.cu131.aarch64]`
- `[package.metadata.flashinfer_rs.pinned_wheels.apache_tvm_ffi.cu130.x86_64]`
- `[package.metadata.flashinfer_rs.pinned_wheels.apache_tvm_ffi.cu130.aarch64]`
- `[package.metadata.flashinfer_rs.pinned_wheels.apache_tvm_ffi.cu131.x86_64]`
- `[package.metadata.flashinfer_rs.pinned_wheels.apache_tvm_ffi.cu131.aarch64]`

Build flow:

1. `build.rs` detects build-host CUDA version (13.0 or 13.1) and target architecture (`x86_64` or `aarch64`) to select pinned wheel entries.
2. `build.rs` downloads each selected wheel.
3. `build.rs` verifies wheel SHA256.
4. `build.rs` embeds wheel bytes via generated `include_bytes!` constants.

Runtime flow:

1. Embedded wheel bytes are materialized to `~/.cache/flashinfer-rs/wheels/` (or `FLASHINFER_RS_CACHE_DIR/wheels/`).
2. Existing cached wheel files are SHA256-validated and rewritten if mismatched.
3. Required `.so` members are extracted from cached wheel files into:
   - `~/.cache/flashinfer-rs/<artifact-hash>/`

Runtime env vars:

- `FLASHINFER_RS_CACHE_DIR`

## Quick Start

1. Build the crate (first build downloads pinned wheels from metadata URLs).
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

## API Example: MHA Single Prefill

```rust
use flashinfer_rs::{
    DType, MhaMaskMode, MhaQkvLayout, MhaSinglePrefillParams, MhaTensor1DU8Desc, MhaTensor3DDesc,
    mha_single_prefill,
};
use std::ffi::c_void;

let params = MhaSinglePrefillParams::new(
    MhaTensor3DDesc {
        ptr: q_ptr as *const c_void,
        dim0: qo_len,
        dim1: num_qo_heads,
        dim2: head_dim_qk,
        stride0: num_qo_heads * head_dim_qk,
        stride1: head_dim_qk,
        stride2: 1,
        dtype: DType::F16,
        device_id: 0,
    },
    MhaTensor3DDesc {
        ptr: k_ptr as *const c_void,
        dim0: kv_len,             // NHD layout
        dim1: num_kv_heads,
        dim2: head_dim_qk,
        stride0: num_kv_heads * head_dim_qk,
        stride1: head_dim_qk,
        stride2: 1,
        dtype: DType::F16,
        device_id: 0,
    },
    MhaTensor3DDesc {
        ptr: v_ptr as *const c_void,
        dim0: kv_len,             // NHD layout
        dim1: num_kv_heads,
        dim2: head_dim_vo,
        stride0: num_kv_heads * head_dim_vo,
        stride1: head_dim_vo,
        stride2: 1,
        dtype: DType::F16,
        device_id: 0,
    },
    MhaTensor1DU8Desc {
        ptr: tmp_ptr as *const c_void, // workspace buffer
        len: tmp_len,
        stride: 1,
        device_id: 0,
    },
    MhaTensor3DDesc {
        ptr: out_ptr as *const c_void,
        dim0: qo_len,
        dim1: num_qo_heads,
        dim2: head_dim_vo,
        stride0: num_qo_heads * head_dim_vo,
        stride1: head_dim_vo,
        stride2: 1,
        dtype: DType::F16,
        device_id: 0,
    },
    stream_ptr,
)
.with_mask_mode(MhaMaskMode::Causal)
.with_kv_layout(MhaQkvLayout::Nhd);

mha_single_prefill(&params)?;
```

## API Example: MHA Batched Ragged Prefill (`cudarc`)

```rust
use flashinfer_rs::{
    DType, MhaBatchPrefillCudarcOptions, MhaQkvLayout, mha_batch_prefill_cudarc_plan,
    mha_batch_prefill_cudarc_run,
};

let options = MhaBatchPrefillCudarcOptions {
    causal: true,
    ..Default::default()
};

let ragged_prefill_plan = mha_batch_prefill_cudarc_plan(
    stream.as_ref(),
    &qo_indptr_host, // host indptr used by plan()
    &kv_indptr_host, // host indptr used by plan()
    &mut float_workspace,
    &mut int_workspace,
    &mut page_locked_int_workspace,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    DType::F16,
    options,
)?;

mha_batch_prefill_cudarc_run(
    stream.as_ref(),
    &ragged_prefill_plan,
    &q,
    &k,
    &v,
    &qo_indptr_dev,
    &kv_indptr_dev,
    &mut float_workspace,
    &mut int_workspace,
    &mut out,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    MhaQkvLayout::Nhd,
    DType::F16,
    options,
)?;
```

## API Example: MHA Batched Paged Prefill (`cudarc`)

```rust
use flashinfer_rs::{
    DType, MhaBatchPrefillCudarcOptions, MhaQkvLayout, mha_batch_prefill_paged_cudarc_plan,
    mha_batch_prefill_paged_cudarc_run,
};

let options = MhaBatchPrefillCudarcOptions {
    causal: true,
    ..Default::default()
};

let paged_prefill_plan = mha_batch_prefill_paged_cudarc_plan(
    stream.as_ref(),
    &qo_indptr_host,
    &paged_kv_indptr_host,
    &kv_len_arr_host,
    &mut float_workspace,
    &mut int_workspace,
    &mut page_locked_int_workspace,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    page_size,
    DType::F16,
    options,
)?;

mha_batch_prefill_paged_cudarc_run(
    stream.as_ref(),
    &paged_prefill_plan,
    &q,
    &paged_k_cache,
    &paged_v_cache,
    &qo_indptr_dev,
    &paged_kv_indptr_dev,
    &paged_kv_indices_dev,
    &paged_kv_last_page_len_dev,
    &mut float_workspace,
    &mut int_workspace,
    &mut out,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    page_size,
    MhaQkvLayout::Nhd,
    DType::F16,
    options,
)?;
```

## API Example: MHA Decode (`cudarc`)

```rust
use flashinfer_rs::{
    DType, MhaBatchDecodeCudarcOptions, MhaSingleDecodeCudarcOptions,
    mha_batch_decode_paged_cudarc_plan, mha_batch_decode_paged_cudarc_run, mha_single_decode_cudarc,
};

mha_single_decode_cudarc(
    stream.as_ref(),
    &q,
    &k,
    &v,
    &mut tmp,
    &mut out,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    head_dim,
    DType::F16,
    MhaSingleDecodeCudarcOptions::default(),
)?;

let decode_plan = mha_batch_decode_paged_cudarc_plan(
    stream.as_ref(),
    &paged_kv_indptr_host,
    &mut float_workspace,
    &mut int_workspace,
    &mut page_locked_int_workspace,
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    head_dim,
    page_size,
    DType::F16,
    MhaBatchDecodeCudarcOptions::default(),
)?;

mha_batch_decode_paged_cudarc_run(
    stream.as_ref(),
    &decode_plan,
    &q_batch,
    &paged_k_cache,
    &paged_v_cache,
    &paged_kv_indptr_dev,
    &paged_kv_indices_dev,
    &paged_kv_last_page_len_dev,
    &mut float_workspace,
    &mut int_workspace,
    &mut out_batch,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    head_dim,
    page_size,
    DType::F16,
    MhaBatchDecodeCudarcOptions::default(),
)?;
```

## Architecture Handling

FlashInfer host wrappers dispatch to architecture-specific kernels at runtime.

- This repo currently loads `gdn_prefill_sm90.so` and calls `__tvm_ffi_gdn_prefill`.
- The current launcher path is SM90A-only for this kernel family.
- On unsupported GPUs, calls fail with decoded TVM-FFI errors (`FlashInferError::TvmFfiCall`).

## Testing

Unit tests (no GPU required):

```bash
cargo test
cargo test --features cudarc
```

GPU smoke tests (require `FLASHINFER_RS_RUN_GPU_TESTS=1` and a compatible GPU):

```bash
FLASHINFER_RS_RUN_GPU_TESTS=1 cargo test --features cudarc --test gemma_rmsnorm_gpu
FLASHINFER_RS_RUN_GPU_TESTS=1 cargo test --features cudarc --test rmsnorm_gpu
FLASHINFER_RS_RUN_GPU_TESTS=1 cargo test --features cudarc --test mha_batch_prefill_gpu
FLASHINFER_RS_RUN_GPU_TESTS=1 cargo test --features cudarc --test mha_batch_prefill_paged_gpu
FLASHINFER_RS_RUN_GPU_TESTS=1 cargo test --features cudarc --test mha_decode_gpu
```

### GDN Prefill Stress Tests

`tests/gdn_prefill_stress_gpu.rs` contains stress tests for the
`FlatKernelTmaWarpSpecializedDeltaRule` CUTLASS kernel, targeting
synchronization and race-condition bugs in the warp-specialized pipeline.
**Requires an SM 9.0 (Hopper) GPU** -- H100 or H200.

Run all stress tests sequentially (recommended to isolate hangs):

```bash
FLASHINFER_RS_RUN_GPU_TESTS=1 cargo test --features cudarc --test gdn_prefill_stress_gpu -- --test-threads=1
```

Run all stress tests concurrently for maximum stress:

```bash
FLASHINFER_RS_RUN_GPU_TESTS=1 cargo test --features cudarc --test gdn_prefill_stress_gpu
```

Run a single test case:

```bash
FLASHINFER_RS_RUN_GPU_TESTS=1 cargo test --features cudarc --test gdn_prefill_stress_gpu stress_gdn_prefill_qwen3_next_multi_stream_chained -- --test-threads=1
```

Each test has a 60-second watchdog. If `stream.synchronize()` does not return
within that window, the test fails with `HANG DETECTED` instead of blocking
forever. Adjust the `HANG_TIMEOUT` constant in the test file if needed.

Test coverage:

| Test | Kernel path exercised |
|------|----------------------|
| `stress_gdn_prefill_mixed_seqlens` | Tile scheduler with sub/at/above 64-token tile boundaries |
| `stress_gdn_prefill_many_single_token_seqs` | Degenerate single-tile path (first == final block) |
| `stress_gdn_prefill_rapid_fire_no_sync` | 50 back-to-back async launches, GVA + alpha/beta |
| `stress_gdn_prefill_all_options_gqa` | GQA + alpha + beta + input_state + explicit scale |
| `stress_gdn_prefill_all_options_gva` | GVA (`IsGVA=true`) + alpha + beta + input_state |
| `stress_gdn_prefill_state_chain_with_gates` | Qwen3-Next GVA, 20-step state chain with alpha/beta |
| `stress_gdn_prefill_multi_stream` | 4 concurrent streams, mixed GQA/GVA configs |
| `stress_gdn_prefill_gqa_head_ratios` | GQA head ratios: 4:1, 6:2, 8:2, 8:4, 16:4 |
| `stress_gdn_prefill_gva_head_ratios` | GVA head ratios: 1:2, 2:4, 4:8, 16:32 |
| `stress_gdn_prefill_extreme_disparity` | 64 single-token seqs + 1 1024-token seq, GVA |
| `stress_gdn_prefill_bf16_gva_all_options` | BF16 dtype, GVA + all options |
| `stress_gdn_prefill_qwen3_next_multi_stream_chained` | Full production config: GVA q=k=16/v=32, alpha+beta, state chain, 4 concurrent streams |

## Additional Notes

- Calls are asynchronous with respect to host execution (no implicit stream synchronize).
- Dynamic loading order is: `libtvm_ffi.so` -> `norm.so` -> `gdn_prefill_sm90.so` -> on-demand MHA prefill/decode modules.
- Integration details: `docs/flashinfer-rs-integration.md`.
