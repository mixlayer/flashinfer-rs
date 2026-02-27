#!/usr/bin/env python3
"""Standalone smoke script for CuTeDSL gdn_decode TVM-FFI compilation.

This script forces compilation of FlashInfer's gdn_decode CuTeDSL kernels using
`--enable-tvm-ffi` (through `flashinfer.gdn_decode`), then attempts to surface
and export the resulting TVM module (if export is supported by the compiled
object in the current environment).

Typical usage from repo root:

  python3 scripts/compile_gdn_decode_tvmffi.py \
    --mode nontranspose \
    --dtype bf16 \
    --export-so /tmp/gdn_decode_tvmffi.so
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CompileResult:
    compiled: Any
    output: Any
    state: Any
    cache: dict[str, Any]
    scale: float


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _add_flashinfer_to_path() -> pathlib.Path:
    repo_root = _repo_root()
    flashinfer_repo = repo_root / "flashinfer"
    if not flashinfer_repo.exists():
        raise RuntimeError(f"flashinfer submodule directory not found: {flashinfer_repo}")
    sys.path.insert(0, str(flashinfer_repo))
    return flashinfer_repo


def _import_required() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "PyTorch is required. Install torch in the Python env used by this script."
        ) from exc

    missing = []
    for mod in ("cutlass", "cuda.bindings.driver"):
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        raise RuntimeError(
            "Missing CuTeDSL dependencies: "
            + ", ".join(missing)
            + ". Install FlashInfer's Python/CuTe stack in this environment."
        )

    try:
        gdn_decode = importlib.import_module("flashinfer.gdn_decode")
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "Failed to import flashinfer.gdn_decode from submodule checkout. "
            "Set up Python deps and ensure this repo has the flashinfer submodule."
        ) from exc

    return torch, gdn_decode


def _dtype_from_name(torch_mod: Any, dtype_name: str) -> Any:
    if dtype_name == "f16":
        return torch_mod.float16
    if dtype_name == "bf16":
        return torch_mod.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _check_cuda(torch_mod: Any, device_index: int) -> None:
    if not torch_mod.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")
    device_count = torch_mod.cuda.device_count()
    if device_index < 0 or device_index >= device_count:
        raise RuntimeError(
            f"Requested CUDA device {device_index}, but visible device count is {device_count}."
        )


def _check_sm90(torch_mod: Any, device_index: int, allow_non_sm90: bool) -> None:
    major, minor = torch_mod.cuda.get_device_capability(device_index)
    if major < 9 and not allow_non_sm90:
        raise RuntimeError(
            f"gdn_decode kernels require SM90+; got SM{major}{minor}. "
            "Pass --allow-non-sm90 to skip this guard."
        )


def _compile_nontranspose(
    torch_mod: Any,
    gdn_decode: Any,
    device: str,
    dtype: Any,
    bsz: int,
    q_heads: int,
    kv_heads: int,
    k_dim: int,
    v_dim: int,
    use_qk_l2norm: bool,
    scale: Optional[float],
) -> CompileResult:
    if scale is None:
        scale = k_dim ** -0.5

    q = torch_mod.randn((bsz, 1, q_heads, k_dim), device=device, dtype=dtype)
    k = torch_mod.randn((bsz, 1, q_heads, k_dim), device=device, dtype=dtype)
    v = torch_mod.randn((bsz, 1, kv_heads, v_dim), device=device, dtype=dtype)

    state = torch_mod.zeros((bsz, kv_heads, k_dim, v_dim), device=device, dtype=torch_mod.float32)
    a_log = torch_mod.randn((kv_heads,), device=device, dtype=torch_mod.float32)
    a = torch_mod.randn((bsz, 1, kv_heads), device=device, dtype=dtype)
    dt_bias = torch_mod.randn((kv_heads,), device=device, dtype=torch_mod.float32)
    beta = torch_mod.randn((bsz, 1, kv_heads), device=device, dtype=dtype)

    output, state_out = gdn_decode.gated_delta_rule_decode(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=a_log,
        a=a,
        dt_bias=dt_bias,
        b=beta,
        scale=scale,
        output=None,
        use_qk_l2norm=use_qk_l2norm,
    )

    cache = gdn_decode._get_compiled_decode_kernel_nontranspose(
        bsz,
        1,
        q_heads,
        kv_heads,
        k_dim,
        v_dim,
        dtype,
        scale,
        use_qk_l2norm,
    )
    compiled = cache.get("compiled")
    if compiled is None:
        raise RuntimeError("Nontranspose kernel cache does not contain compiled object.")

    return CompileResult(compiled=compiled, output=output, state=state_out, cache=cache, scale=scale)


def _compile_pretranspose(
    torch_mod: Any,
    gdn_decode: Any,
    device: str,
    dtype: Any,
    bsz: int,
    q_heads: int,
    kv_heads: int,
    k_dim: int,
    v_dim: int,
    use_qk_l2norm: bool,
    scale: Optional[float],
) -> CompileResult:
    if scale is None:
        scale = k_dim ** -0.5

    q = torch_mod.randn((bsz, 1, q_heads, k_dim), device=device, dtype=dtype)
    k = torch_mod.randn((bsz, 1, q_heads, k_dim), device=device, dtype=dtype)
    v = torch_mod.randn((bsz, 1, kv_heads, v_dim), device=device, dtype=dtype)

    state = torch_mod.zeros((bsz, kv_heads, v_dim, k_dim), device=device, dtype=torch_mod.float32)
    a_log = torch_mod.randn((kv_heads,), device=device, dtype=torch_mod.float32)
    a = torch_mod.randn((bsz, 1, kv_heads), device=device, dtype=dtype)
    dt_bias = torch_mod.randn((kv_heads,), device=device, dtype=torch_mod.float32)
    beta = torch_mod.randn((bsz, 1, kv_heads), device=device, dtype=dtype)

    output, state_out = gdn_decode.gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=state,
        A_log=a_log,
        a=a,
        dt_bias=dt_bias,
        b=beta,
        scale=scale,
        output=None,
        use_qk_l2norm=use_qk_l2norm,
    )

    cache = gdn_decode._get_compiled_decode_kernel(
        bsz,
        1,
        q_heads,
        kv_heads,
        k_dim,
        v_dim,
        dtype,
        scale,
        use_qk_l2norm,
    )
    compiled = cache.get("compiled")
    if compiled is None:
        raise RuntimeError("Pretranspose kernel cache does not contain compiled object.")

    return CompileResult(compiled=compiled, output=output, state=state_out, cache=cache, scale=scale)


def _interesting_attrs(obj: Any) -> list[str]:
    keys = ("module", "tvm", "ffi", "export", "path", "library", "so", "shared")
    attrs = []
    for name in dir(obj):
        lname = name.lower()
        if any(k in lname for k in keys):
            attrs.append(name)
    return sorted(set(attrs))


def _find_exportable_module(compiled: Any) -> Optional[Any]:
    candidates: list[Any] = [compiled]
    for attr in (
        "module",
        "_module",
        "tvm_module",
        "_tvm_module",
        "rt_mod",
        "runtime_module",
    ):
        if hasattr(compiled, attr):
            candidates.append(getattr(compiled, attr))

    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "export_library"):
            return candidate
    return None


def _dump_nm_symbols(path: pathlib.Path) -> list[str]:
    return _dump_nm_symbols_with_cmd(path, dynamic=True)


def _dump_nm_symbols_with_cmd(path: pathlib.Path, *, dynamic: bool) -> list[str]:
    cmd = ["nm"]
    if dynamic:
        cmd.append("-D")
    cmd.append(str(path))
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return [line for line in proc.stdout.splitlines() if "tvm_ffi" in line]


def _sanitize_symbol_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    symbol = "".join(out).strip("_")
    if not symbol:
        symbol = "gdn_decode_smoke"
    if symbol[0].isdigit():
        symbol = f"gdn_{symbol}"
    return symbol


def _derive_export_symbol(args: argparse.Namespace, compiled: Any) -> str:
    if args.export_function_name:
        return _sanitize_symbol_name(args.export_function_name)
    mode = _sanitize_symbol_name(args.mode)
    return f"gdn_decode_{mode}_smoke"


def _default_object_path(export_so: pathlib.Path) -> pathlib.Path:
    if export_so.suffix:
        return export_so.with_suffix(".o")
    return pathlib.Path(str(export_so) + ".o")


def _link_object_to_shared(obj_path: pathlib.Path, so_path: pathlib.Path) -> None:
    linker = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if linker is None:
        raise RuntimeError(
            "No C++ linker found (`c++`, `g++`, or `clang++`) to link exported object into shared library."
        )
    cmd = [linker, "-shared", "-o", str(so_path), str(obj_path)]
    subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile flashinfer.gdn_decode CuTeDSL kernels with TVM-FFI enabled.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("nontranspose", "pretranspose"), default="nontranspose")
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--k-dim", type=int, default=128)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-qk-l2norm", action="store_true")
    parser.add_argument("--allow-non-sm90", action="store_true")
    parser.add_argument(
        "--export-obj",
        type=pathlib.Path,
        default=None,
        help="Export compiled TVM-FFI wrapper to an object file using CuTe export_to_c.",
    )
    parser.add_argument(
        "--export-so",
        type=pathlib.Path,
        default=None,
        help=(
            "Export compiled TVM-FFI wrapper as shared library. "
            "Uses export_library() when available; otherwise falls back to export_to_c + linker."
        ),
    )
    parser.add_argument(
        "--export-function-name",
        type=str,
        default=None,
        help=(
            "Symbol base name for export_to_c path (resulting symbol is __tvm_ffi_<name>). "
            "Ignored when export_library() is used."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Shape guards aligned with gdn_decode.py expectations.
    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    if args.q_heads <= 0 or args.kv_heads <= 0:
        raise RuntimeError("--q-heads/--kv-heads must be > 0")
    if args.k_dim < 128:
        raise RuntimeError("--k-dim must be >= 128")
    if args.v_dim < 128:
        raise RuntimeError("--v-dim must be >= 128")
    if args.mode == "nontranspose" and args.v_dim % 32 != 0:
        raise RuntimeError("nontranspose mode requires --v-dim divisible by 32")
    if args.mode == "pretranspose" and args.v_dim % 8 != 0:
        raise RuntimeError("pretranspose mode requires --v-dim divisible by 8")

    _add_flashinfer_to_path()
    torch_mod, gdn_decode = _import_required()

    _check_cuda(torch_mod, args.device)
    _check_sm90(torch_mod, args.device, args.allow_non_sm90)

    torch_mod.cuda.set_device(args.device)
    dtype = _dtype_from_name(torch_mod, args.dtype)
    device = f"cuda:{args.device}"
    use_qk_l2norm = not args.no_qk_l2norm

    print(
        "[info] compile config:",
        f"mode={args.mode}",
        f"dtype={args.dtype}",
        f"B={args.batch_size}",
        f"H={args.q_heads}",
        f"HV={args.kv_heads}",
        f"K={args.k_dim}",
        f"V={args.v_dim}",
        f"device={device}",
        f"qk_l2norm={use_qk_l2norm}",
    )

    if args.mode == "nontranspose":
        result = _compile_nontranspose(
            torch_mod,
            gdn_decode,
            device,
            dtype,
            args.batch_size,
            args.q_heads,
            args.kv_heads,
            args.k_dim,
            args.v_dim,
            use_qk_l2norm,
            args.scale,
        )
    else:
        result = _compile_pretranspose(
            torch_mod,
            gdn_decode,
            device,
            dtype,
            args.batch_size,
            args.q_heads,
            args.kv_heads,
            args.k_dim,
            args.v_dim,
            use_qk_l2norm,
            args.scale,
        )

    torch_mod.cuda.synchronize(args.device)

    compiled = result.compiled
    print(f"[ok] compiled object type: {type(compiled)!r}")
    print(f"[ok] cache keys: {sorted(result.cache.keys())}")
    print(f"[ok] effective scale: {result.scale}")

    attrs = _interesting_attrs(compiled)
    if attrs:
        print("[info] interesting compiled attrs:", ", ".join(attrs))

    wants_obj = args.export_obj is not None
    wants_so = args.export_so is not None

    export_target = _find_exportable_module(compiled)

    if export_target is not None and wants_so:
        print(f"[ok] export-capable object type: {type(export_target)!r}")
        so_path = args.export_so.expanduser().resolve()
        so_path.parent.mkdir(parents=True, exist_ok=True)
        export_target.export_library(str(so_path))
        print(f"[ok] exported shared library via export_library: {so_path}")

        tvm_ffi_symbols = _dump_nm_symbols(so_path)
        if tvm_ffi_symbols:
            print("[ok] symbols matching 'tvm_ffi':")
            for line in tvm_ffi_symbols:
                print(f"  {line}")
        else:
            print(
                "[warn] export succeeded but no 'tvm_ffi' symbols were found via `nm -D` "
                "(or nm is unavailable)."
            )

        # Optional object export for inspection/debugging.
        if wants_obj and hasattr(compiled, "export_to_c"):
            obj_path = args.export_obj.expanduser().resolve()
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            function_name = _derive_export_symbol(args, compiled)
            compiled.export_to_c(
                str(obj_path),
                function_name=function_name,
                export_only_tvm_ffi_symbols=True,
            )
            print(f"[ok] exported object file via export_to_c: {obj_path}")
        return 0

    # export_to_c fallback path
    if wants_obj or wants_so:
        if not hasattr(compiled, "export_to_c"):
            print(
                "[warn] Could not find export_library() or export_to_c() on compiled object; "
                "cannot export object/shared library from this API object."
            )
            if args.verbose:
                print("[verbose] dir(compiled):")
                print(textwrap.indent("\n".join(sorted(dir(compiled))), "  "))
            return 0

        function_name = _derive_export_symbol(args, compiled)
        obj_path = (
            args.export_obj.expanduser().resolve()
            if wants_obj
            else _default_object_path(args.export_so.expanduser().resolve())
        )
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        compiled.export_to_c(
            str(obj_path),
            function_name=function_name,
            export_only_tvm_ffi_symbols=True,
        )
        print(f"[ok] exported object file via export_to_c: {obj_path}")
        print(f"[ok] exported symbol base: __tvm_ffi_{function_name}")

        obj_symbols = _dump_nm_symbols_with_cmd(obj_path, dynamic=False)
        if obj_symbols:
            print("[ok] object symbols matching 'tvm_ffi':")
            for line in obj_symbols:
                print(f"  {line}")
        else:
            print("[warn] no 'tvm_ffi' symbols found in object via `nm` (or nm unavailable).")

        if wants_so:
            so_path = args.export_so.expanduser().resolve()
            so_path.parent.mkdir(parents=True, exist_ok=True)
            _link_object_to_shared(obj_path, so_path)
            print(f"[ok] linked shared library from object: {so_path}")

            so_symbols = _dump_nm_symbols(so_path)
            if so_symbols:
                print("[ok] shared library symbols matching 'tvm_ffi':")
                for line in so_symbols:
                    print(f"  {line}")
            else:
                print(
                    "[warn] linked shared library but no 'tvm_ffi' symbols were found via `nm -D`."
                )
        return 0

    # No export requested: provide guidance for this object kind.
    if export_target is None:
        print(
            "[warn] Could not find an object exposing export_library(). "
            "Compilation succeeded. Use --export-obj and/or --export-so to use export_to_c fallback."
        )
    else:
        print(f"[ok] export-capable object type: {type(export_target)!r}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
