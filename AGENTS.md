# AGENTS.md

This file documents the expected process for adding new FlashInfer kernel bindings in this repo.

## Scope and Principles

- We bind precompiled kernels from wheels, not ad-hoc local builds, unless explicitly requested.
- We do not embed wheel bytes with `include_bytes!`; `build.rs` emits pinned wheel metadata (`filename`, `url`, `sha256`) only.
- Wheel bytes are fetched at runtime on cache miss with synchronous I/O (no async runtime dependency).
- We use pure Rust TVM-FFI call packing at the boundary.
- We always add `cudarc` wrappers for each public kernel API.
- We keep launch semantics async (no implicit synchronize in core APIs).
- For kernels with planning + launch stages, we always expose explicit `plan` and `run` APIs and never implicitly combine them in a single public launch call.
- We validate aggressively in Rust before crossing FFI.

## Kernel-Binding Workflow

1. Identify artifact in wheel
- Find the `.so` inside `flashinfer_jit_cache/.../jit_cache/...`.
- Prefer wheel member inspection first (zip listing) before code changes.
- Confirm whether kernel is:
  - a fixed `.so` loaded at runtime init (e.g. `norm.so`, `gdn_prefill_sm90.so`), or
  - a family of variant `.so` files loaded on demand (e.g. single prefill URIs).

2. Identify exported host symbol
- Use `nm -D` (or equivalent) on extracted `.so`.
- Match exported user-facing TVM-FFI symbol (`__tvm_ffi_*`), not cubin mangled device symbols.
- Common patterns:
  - dedicated symbol, e.g. `__tvm_ffi_gemma_rmsnorm`
  - generic entry point, e.g. `__tvm_ffi_run`, where function ABI is determined by the loaded module variant.

3. Verify ABI and call signature from FlashInfer source submodule
- Source of truth is the binding/export layer in `flashinfer/csrc/*_jit_binding.cu` and related launcher/config files.
- For function arguments and ordering, trace from Python wrapper paths in `flashinfer/flashinfer/*.py` to the JIT module `run/plan/...` call.
- Confirm enum/int values from headers (`layout`, `mask`, `pos_enc`, etc.) in `flashinfer/include/...`.

4. Implement Rust API + validation
- Add typed parameter structs and enums.
- Add field-level doc comments on every `*Params` struct:
  - expected tensor shapes (with layout variants where relevant),
  - units/semantics for scalar knobs,
  - stream semantics.
- Cross-reference the shape contract source in FlashInfer submodule comments (typically
  `flashinfer/csrc/*` and/or `flashinfer/flashinfer/*.py`).
- Validate:
  - pointer non-null
  - shape/layout constraints
  - dtype compatibility
  - stride/contiguity assumptions
  - device consistency
  - scalar domains (finite/positive/allowed range)
- Build stack-owned `DLTensor` shape/stride arrays and pack `TVMFFIAny` args in exact ABI order.
- Use stream set/restore guard around each call.

5. Runtime loading integration
- If symbol is fixed: resolve during runtime init and store function pointer.
- If symbol is variant-dependent: lazy extract/load by URI and cache library+function pointer for process lifetime.
- Keep wheel cache under `<cache_dir>/wheels/<sha256>-<filename>` with lock-file protection and SHA256 verification.
- On checksum mismatch, rewrite the cached wheel from the pinned URL.
- Keep wheel download logic synchronous and streaming; do not introduce async runtime dependencies.
- Keep `libtvm_ffi.so` loaded first with `RTLD_GLOBAL`.

6. Add cudarc wrapper (required)
- Add a `cudarc` convenience wrapper for each kernel API.
- Wrapper responsibilities:
  - shape/length checks from flat buffers
  - device pointer extraction
  - shape/stride derivation
  - constructing typed params and calling core API
- Do not duplicate kernel logic in wrapper.

7. Tests (minimum ABI safety gate)
- Unit tests for validation and packers must be added/updated.
- Minimum required integration test: launch smoke test against the targeted wheel artifact(s) to catch ABI drift/regression.
- Smoke test expectations:
  - runtime initializes
  - symbol/module resolves
  - kernel launch returns success for a valid tiny case
  - no panic on error path (decoded TVM-FFI errors where applicable)

## How symbol matching is done in practice here

- Start from wheel member names to identify the exact module URI or fixed path.
- Confirm exported symbols from that exact `.so`.
- If exported symbol is generic (`__tvm_ffi_run`), derive the ABI from:
  - matching `*_jit_binding.cu` function export
  - the Python-side `module.run(...)` call argument list for that module family.
- Then lock Rust argument order/types to that ABI.

## When and how to consult the submodule

Use the submodule when any of these are unclear:

- expected argument ordering
- optional tensor/scalar handling
- enum numeric values
- architecture/backend constraints (e.g. SM-only branches)
- URI naming logic for JIT-cache variants

Primary files to inspect:

- `flashinfer/csrc/*_jit_binding.cu`
- `flashinfer/csrc/*launcher*.cu`
- `flashinfer/flashinfer/jit/attention/modules.py` (URI and backend variant generation)
- `flashinfer/flashinfer/prefill.py` / related wrappers (runtime call argument mapping)
- `flashinfer/include/flashinfer/...` (enum values and contracts)

## Gotchas and Notes for Future Bindings

- Wheel naming does not guarantee architecture suffix in URI; always inspect actual wheel members.
- `build.rs` no longer downloads wheels; first runtime initialization on a cold cache performs network download and may block.
- A generic host symbol (`__tvm_ffi_run`) can correspond to different ABI shapes across module families.
- Do not infer ABI from cubin symbols; use host binding sources.
- Optional parameters often map to `None`/null tensor slots and must preserve positional ABI.
- Keep stream restoration robust across both success and failure paths.
- Re-check TVM-FFI type indices before adding new packed types.
- Prefer on-demand loading for large variant spaces to avoid startup bloat.
- Maintain feature parity: if core API is added, `cudarc` wrapper should be added in same change.
- Never reuse a single params struct for both `plan` and `run`; define explicit plan-only and run-only param types.

## Backend to URI Selection (fa2/fa3 and `_sm90`)

How upstream selects the prefill kernel family:

1. Backend decision:
- In upstream Python wrappers, `backend="auto"` is resolved by `determine_attention_backend(...)`.
- Source: `flashinfer/flashinfer/utils.py` and call sites in `flashinfer/flashinfer/prefill.py`.
- For SM90 + supported feature set, it returns `fa3`; otherwise `fa2`.

2. URI generation:
- Prefill URI comes from `get_batch_prefill_uri(backend, ...)`.
- Source: `flashinfer/flashinfer/jit/attention/modules.py`.
- The URI suffix `"_sm90"` is appended only when `backend == "fa3"`.

3. Module loading:
- The wrapper loads exactly that URI, so `fa3` selects `<uri>_sm90.so` and `fa2` selects `<uri>.so`.

Practical implication for Rust bindings:

- If Rust builds the prefill URI without a backend choice (or always uses the unsuffixed URI),
  runtime will load the generic prefill module even on Hopper.
- That can be materially slower than the FA3/SM90 specialized module and is a likely cause
  of large prefill regressions.
- For parity with upstream behavior, Rust should:
  - make backend explicit (`fa2` vs `fa3`) or implement equivalent auto-selection logic,
  - append `"_sm90"` when the chosen backend is `fa3`,
  - keep URI construction consistent with upstream `modules.py`.

## GEMM Integration Approach (TGV f16/bf16 + TRTLLM tactics)

This section captures the current GEMM binding approach used in this repo.

1. Scope
- f16/bf16 GEMM support is MM-only in current pinned wheels.
- Do not add f16/bf16 BMM API unless matching wheel exports are verified.

2. Wheel artifacts and exported symbols
- TGV MM artifacts:
  - `flashinfer_jit_cache/.../jit_cache/tgv_gemm_fp16/tgv_gemm_fp16.so`
  - `flashinfer_jit_cache/.../jit_cache/tgv_gemm_bf16/tgv_gemm_bf16.so`
- TGV symbols:
  - `__tvm_ffi_tgv_gemm`
  - `__tvm_ffi_tgv_gemm_tactic_num`
- TRTLLM GEMM artifact:
  - `flashinfer_jit_cache/.../jit_cache/trtllm_gemm/trtllm_gemm.so`
- TRTLLM GEMM symbol:
  - `__tvm_ffi_trtllm_gemm_tactics`
- TRTLLM low-latency GEMM artifact:
  - `flashinfer_jit_cache/.../jit_cache/trtllm_low_latency_gemm/trtllm_low_latency_gemm.so`
- TRTLLM low-latency symbols:
  - `__tvm_ffi_trtllm_low_latency_gemm_tactics`
  - `__tvm_ffi_get_workspace_size_in_bytes`

3. ABI mapping for TGV MM
- Public semantics: `out = A @ B (+bias)` with:
  - `A`: `[m, k]` row-major
  - `B`: `[k, n]` column-major
  - `out`: `[m, n]` row-major
  - optional `bias`: `[n]`
- TVM call packing must match upstream TGV runner convention:
  - `mat1 = B^T`, `mat2 = A^T`, `bias`, `tactic`, `out`, `pdl`
- Use transpose views only (no staging copy) by swapping shape/stride metadata.

4. Validation requirements (GEMM)
- Pointer non-null for all required tensors.
- Positive dimensions.
- Dtype consistency across inputs/outputs (`f16` or `bf16` for TGV path).
- Device consistency across all tensors.
- Stride/layout checks:
  - `A.stride_col == 1` (row-major)
  - `B.stride_row == 1` (column-major)
  - `out.stride_col == 1` (row-major)
  - `bias.stride == 1` when present
- Scalar domain checks:
  - tactic selector values in allowed range (`-1` or non-negative as API defines)
  - tactics/workspace query dims must be positive.

5. Runtime loading and result decoding
- Treat TGV/TRTLLM GEMM modules as lazy-loaded URI kernels and cache function pointers for process lifetime.
- For APIs returning `Array<int64_t>` (TRTLLM tactics), decode through TVM globals:
  - `ffi.ArraySize`
  - `ffi.ArrayGetItem`
- Resolve those globals using existing `get_global_function` + `call_function`, and preserve object/handle decref discipline.

6. API surface expectations
- Core typed API:
  - parameter structs for TGV launch
  - typed enums for TRTLLM dtype codes
  - typed query structs for tactics/workspace helpers
- `cudarc` wrappers are required for public TGV MM APIs (with and without bias).
- Default launch behavior should remain async with no implicit synchronize.

## PR/Change Checklist

- [ ] Wheel artifact path(s) identified and verified.
- [ ] `build.rs` metadata constants updated/verified (`filename`, `url`, `sha256`) for selected wheel matrix.
- [ ] Exported host symbol(s) verified from target `.so`.
- [ ] ABI/order confirmed from submodule source and wrapper calls.
- [ ] Field-level shape docs added/updated for all `*Params` struct fields.
- [ ] Rust typed API + validation implemented.
- [ ] Runtime loading strategy implemented (fixed vs lazy URI).
- [ ] Runtime wheel cache/download path validated (SHA256 check + lock behavior, no async runtime introduced).
- [ ] `cudarc` wrapper added.
- [ ] Unit tests added/updated.
- [ ] Wheel launch smoke test added/updated for ABI regression coverage.
- [ ] Docs (`README.md` and/or `docs/flashinfer-rs-integration.md`) updated.
- [ ] For GEMM tactics APIs, `Array<int64_t>` decode path (`ffi.ArraySize`/`ffi.ArrayGetItem`) validated.
