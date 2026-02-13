#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{DType, FusedMoeBackend, fused_moe_cudarc};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn encode_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

fn backend_for_compute_capability(major: i32, minor: i32) -> Option<FusedMoeBackend> {
    match major * 10 + minor {
        90 => Some(FusedMoeBackend::Sm90),
        100 | 110 => Some(FusedMoeBackend::Sm100),
        103 => Some(FusedMoeBackend::Sm103),
        120 | 121 => Some(FusedMoeBackend::Sm120),
        _ => None,
    }
}

#[test]
fn gpu_smoke_launch_fused_moe() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    let Some(backend) = backend_for_compute_capability(major, minor) else {
        eprintln!(
            "skipping fused moe smoke test on unsupported compute capability {}.{}",
            major, minor
        );
        return;
    };

    let num_tokens = 8_usize;
    let num_experts = 4_usize;
    let top_k = 2_usize;
    let hidden_size = 128_usize;
    let inter_size = 64_usize;
    let fc1_inter_size = inter_size * 2; // Swiglu path in wrapper defaults.

    let input_len = num_tokens * hidden_size;
    let out_len = input_len;
    let selected_len = num_tokens * top_k;
    let fc1_len = num_experts * fc1_inter_size * hidden_size;
    let fc2_len = num_experts * hidden_size * inter_size;

    let input_host: Vec<u16> = (0..input_len)
        .map(|i| encode_f16(((i % 31) as f32 - 15.0) * 0.03125))
        .collect();
    let token_selected_host: Vec<i32> = (0..selected_len)
        .map(|i| (i % num_experts) as i32)
        .collect();
    let fc1_host: Vec<u16> = (0..fc1_len)
        .map(|i| encode_f16(((i % 17) as f32 - 8.0) * 0.015625))
        .collect();
    let fc2_host: Vec<u16> = (0..fc2_len)
        .map(|i| encode_f16(((i % 19) as f32 - 9.0) * 0.015625))
        .collect();

    let input_dev = stream.clone_htod(&input_host).expect("copy input");
    let token_selected_dev = stream
        .clone_htod(&token_selected_host)
        .expect("copy token_selected_experts");
    let fc1_dev = stream.clone_htod(&fc1_host).expect("copy fc1");
    let fc2_dev = stream.clone_htod(&fc2_host).expect("copy fc2");
    let mut out_dev = stream.alloc_zeros::<u16>(out_len).expect("alloc out");

    fused_moe_cudarc(
        stream.as_ref(),
        &input_dev,
        &token_selected_dev,
        &fc1_dev,
        &fc2_dev,
        &mut out_dev,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        inter_size,
        DType::F16,
        backend,
        Default::default(),
    )
    .expect("launch fused moe");

    stream.synchronize().expect("synchronize");
}
