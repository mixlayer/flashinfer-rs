#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{Fp8GemmDType, fp8_gemm_cudarc};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn gpu_dense_fp8_gemm() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.new_stream().expect("create CUDA stream");
    let (major, minor) = ctx.compute_capability().expect("read compute capability");
    if major * 10 + minor < 100 {
        eprintln!("skipping dense FP8 smoke test on compute capability {major}.{minor}");
        return;
    }

    let (m, n, k) = (4, 128, 128);
    // E4M3 byte 0x38 represents 1.0.
    let input = stream
        .clone_htod(&vec![0x38_u8; m * k])
        .expect("copy FP8 input");
    let weight = stream
        .clone_htod(&vec![0x38_u8; n * k])
        .expect("copy FP8 weight");
    let input_scale = stream
        .clone_htod(&vec![1.0_f32; (k / 128) * m])
        .expect("copy input scale");
    let weight_scale = stream
        .clone_htod(&vec![1.0_f32; (k / 128) * (n / 128)])
        .expect("copy weight scale");
    let mut output = stream
        .alloc_zeros::<u16>(m * n)
        .expect("allocate BF16 output");
    let mut workspace = stream
        .alloc_zeros::<u8>(32 * 1024 * 1024)
        .expect("allocate workspace");

    fp8_gemm_cudarc(
        stream.as_ref(),
        &input,
        &weight,
        &input_scale,
        &weight_scale,
        &mut output,
        &mut workspace,
        m,
        n,
        k,
        Fp8GemmDType::BF16,
    )
    .expect("dense FP8 GEMM");
    stream.synchronize().expect("synchronize CUDA stream");

    let output = stream.clone_dtoh(&output).expect("copy GEMM output");
    for value in output {
        let value = half::bf16::from_bits(value).to_f32();
        assert!(
            (value - k as f32).abs() <= 1.0,
            "unexpected FP8 GEMM output {value}"
        );
    }
}
