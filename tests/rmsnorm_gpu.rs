#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{DType, fused_qk_rmsnorm_cudarc, rmsnorm_cudarc};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn encode_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

#[test]
fn gpu_smoke_launch_rmsnorm_2d() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let rows = 4_usize;
    let cols = 128_usize;
    let elems = rows * cols;

    let input_host: Vec<u16> = (0..elems)
        .map(|i| encode_f16(((i % 29) as f32 - 14.0) * 0.03125))
        .collect();
    let weight_host: Vec<u16> = (0..cols)
        .map(|i| encode_f16(((i % 17) as f32 - 8.0) * 0.015625))
        .collect();

    let input_dev = stream.clone_htod(&input_host).expect("copy input");
    let weight_dev = stream.clone_htod(&weight_host).expect("copy weight");
    let mut out_dev = stream.alloc_zeros::<u16>(elems).expect("alloc out");

    rmsnorm_cudarc(
        stream.as_ref(),
        &input_dev,
        &weight_dev,
        &mut out_dev,
        rows,
        cols,
        DType::F16,
        1e-6,
    )
    .expect("launch rmsnorm");

    stream.synchronize().expect("synchronize");
}

#[test]
fn gpu_smoke_launch_fused_qk_rmsnorm_3d() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let batch_size = 2_usize;
    let num_heads = 8_usize;
    let head_dim = 128_usize;
    let elems = batch_size * num_heads * head_dim;

    let input_host: Vec<u16> = (0..elems)
        .map(|i| encode_f16(((i % 31) as f32 - 15.0) * 0.03125))
        .collect();
    let weight_host: Vec<u16> = (0..head_dim)
        .map(|i| encode_f16(((i % 13) as f32 - 6.0) * 0.015625))
        .collect();

    let input_dev = stream.clone_htod(&input_host).expect("copy input");
    let weight_dev = stream.clone_htod(&weight_host).expect("copy weight");
    let mut out_dev = stream.alloc_zeros::<u16>(elems).expect("alloc out");

    fused_qk_rmsnorm_cudarc(
        stream.as_ref(),
        &input_dev,
        &weight_dev,
        &mut out_dev,
        batch_size,
        num_heads,
        head_dim,
        DType::F16,
        1e-6,
    )
    .expect("launch fused qk rmsnorm");

    stream.synchronize().expect("synchronize");
}
