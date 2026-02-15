# AGENTS.md

This file documents the expected process for adding new FlashInfer kernel bindings in this repo.

## Scope and Principles

- We bind precompiled kernels from wheels, not ad-hoc local builds, unless explicitly requested.
- We use pure Rust TVM-FFI call packing at the boundary.
- We always add `cudarc` wrappers for each public kernel API.
- We keep launch semantics async (no implicit synchronize in core APIs).
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
- `flashinfer/flashinfer/gdn_decode.py` (CuTeDSL compile path and runtime call argument ordering)
- `flashinfer/include/flashinfer/...` (enum values and contracts)

## CuTeDSL Learnings (from `gdn_decode` experiment)

- CuTeDSL kernels can be exposed through TVM-FFI, but export ergonomics differ from wheel-packaged modules.
- The compiled object is often `TVMFFIJitCompiledFunction` and may expose `export_to_c(...)` but not `export_library()`.
- `export_to_c(...)` requires an explicit `function_name`; exported host symbol is `__tvm_ffi_<function_name>`.
- Do not assume a generic host entrypoint (`__tvm_ffi_run`) for CuTeDSL exports; expect dedicated symbols (for example `__tvm_ffi_gdn_decode_nontranspose_smoke`).
- Generated CuTeDSL shared libraries can have unresolved runtime symbols unless dependencies are preloaded globally.
- Typical unresolved symbols include `_cudaLibraryLoadData`, `cuda_dialect_*`, and TVM-FFI helpers such as `TVMFFIEnvGetStream`.
- In practice, load `libtvm_ffi.so` and `libcute_dsl_runtime.so` first with `RTLD_GLOBAL`, then load the CuTeDSL module.
- ABI can be strict and specialized: wrappers validate exact argument count/types (for example `Expects 12 parameters`).
- Stream handling may be encoded as an explicit ABI argument in CuTeDSL exports even when existing FlashInfer bindings use `TVMFFIEnvSetStream`.
- Ad-hoc CuTeDSL exports may target a single architecture (for example `sm_120a`) rather than wheel-style multi-arch packaging.
- Version skew is a real risk when compiling outside pinned wheel artifacts (`apache-tvm-ffi`, `nvidia-cutlass-dsl`, CUDA stack).

## CuTeDSL Venv Requirements (`scripts/compile_gdn_decode_tvmffi.py`)

- The script must run inside a Python venv with CUDA-capable PyTorch and CuTeDSL deps.
- Required Python imports for this script path:
  - `torch`
  - `cutlass` (provided by `nvidia-cutlass-dsl`)
  - `cuda.bindings.driver` (provided by `cuda-python`)
  - `flashinfer.gdn_decode` from the local `flashinfer` submodule checkout
- Recommended bootstrap from repo root:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `python3 -m pip install -U pip setuptools wheel`
  - `python3 -m pip install -r flashinfer/requirements.txt`
- If installing manually, ensure at least:
  - `torch`
  - `nvidia-cutlass-dsl`
  - `cuda-python`
  - `apache-tvm-ffi`
- Local source requirement:
  - `flashinfer` submodule must exist (`git submodule update --init --recursive`).
  - The script already prepends local `flashinfer/` to `sys.path`; no extra `PYTHONPATH` needed when run from this repo.
- Runtime/tooling requirements:
  - CUDA device visible to PyTorch (`torch.cuda.is_available()`).
  - GPU capability SM90+ unless using `--allow-non-sm90`.
  - A C++ linker (`c++`, `g++`, or `clang++`) is required when using `--export-so` fallback (`export_to_c + link` path).
- Known working package set in this repo during investigation:
  - `torch==2.10.0`
  - `nvidia-cutlass-dsl==4.4.0`
  - `cuda-python==12.9.4`
  - `apache-tvm-ffi==0.1.8.post2`

## Gotchas and Notes for Future Bindings

- Wheel naming does not guarantee architecture suffix in URI; always inspect actual wheel members.
- A generic host symbol (`__tvm_ffi_run`) can correspond to different ABI shapes across module families.
- Do not infer ABI from cubin symbols; use host binding sources.
- Optional parameters often map to `None`/null tensor slots and must preserve positional ABI.
- Keep stream restoration robust across both success and failure paths.
- Re-check TVM-FFI type indices before adding new packed types.
- Prefer on-demand loading for large variant spaces to avoid startup bloat.
- Maintain feature parity: if core API is added, `cudarc` wrapper should be added in same change.

## CuTeDSL TODOs

- [ ] Decide integration mode for CuTeDSL kernels: wheel-only (preferred) or optional local exported `.so` support.
- [ ] Extend runtime loader to optionally preload `libcute_dsl_runtime.so` with `RTLD_GLOBAL` before CuTeDSL module loads.
- [ ] Add dedicated symbol loading path for CuTeDSL exports that do not provide `__tvm_ffi_run`.
- [ ] Implement typed Rust `gdn_decode` API matching exact CuTeDSL TVM-FFI ABI order and arity.
- [ ] Handle explicit stream argument semantics for CuTeDSL ABI where required.
- [ ] Add `cudarc` wrapper for each CuTeDSL public API at the same time as core API integration.
- [ ] Add a launch smoke test against targeted CuTeDSL artifact(s) to detect ABI drift.
- [ ] Add at least one negative-path test validating decoded TVM-FFI errors for bad arg shape/count/type.
- [ ] Document CuTeDSL dependency matrix (`libtvm_ffi.so`, `libcute_dsl_runtime.so`, CUDA requirements) in integration docs.
- [ ] Verify architecture coverage (SM90/SM100/SM120) before enabling CuTeDSL bindings by default.

## PR/Change Checklist

- [ ] Wheel artifact path(s) identified and verified.
- [ ] Exported host symbol(s) verified from target `.so`.
- [ ] ABI/order confirmed from submodule source and wrapper calls.
- [ ] Field-level shape docs added/updated for all `*Params` struct fields.
- [ ] Rust typed API + validation implemented.
- [ ] Runtime loading strategy implemented (fixed vs lazy URI).
- [ ] `cudarc` wrapper added.
- [ ] Unit tests added/updated.
- [ ] Wheel launch smoke test added/updated for ABI regression coverage.
- [ ] Docs (`README.md` and/or `docs/flashinfer-rs-integration.md`) updated.
