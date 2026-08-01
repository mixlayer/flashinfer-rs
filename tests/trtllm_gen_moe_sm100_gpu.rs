#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{
    TrtllmGenFp8BlockScaleMoeSm100CudarcOptions, trtllm_gen_fp8_block_scale_moe_sm100_cudarc,
};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn gpu_smoke_launch_trtllm_gen_fp8_block_scale_moe_sm100() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if (major, minor) != (10, 0) {
        eprintln!("skipping TensorRT-LLM Gen MoE smoke test on compute capability {major}.{minor}");
        return;
    }
    let stream = ctx.new_stream().expect("create stream");

    let tokens = 1_usize;
    // Use GLM-5.2's production GEMM dimensions: TensorRT-LLM Gen does not provide tactics for
    // every mathematically valid multiple-of-128 shape.
    let hidden = 6144_usize;
    let intermediate = 2048_usize;
    let global_experts = 256_usize;
    let local_experts = 32_usize;

    let logits = vec![0.0_f32; tokens * global_experts];
    let routing_logits = stream.clone_htod(&logits).expect("copy routing logits");
    let routing_bias = stream
        .clone_htod(&vec![0.0_f32; global_experts])
        .expect("copy routing bias");
    let hidden_states = stream
        .alloc_zeros::<u8>(tokens * hidden)
        .expect("allocate FP8 hidden states");
    let hidden_states_scale = stream
        .clone_htod(&vec![1.0_f32; hidden / 128 * tokens])
        .expect("copy hidden-state scales");
    let fc1_weights = stream
        .alloc_zeros::<u8>(local_experts * 2 * intermediate * hidden)
        .expect("allocate FP8 fc1 weights");
    let fc1_scales = stream
        .clone_htod(&vec![
            1.0_f32;
            local_experts
                * (2 * intermediate / 128)
                * (hidden / 128)
        ])
        .expect("copy fc1 scales");
    let fc2_weights = stream
        .alloc_zeros::<u8>(local_experts * hidden * intermediate)
        .expect("allocate FP8 fc2 weights");
    let fc2_scales = stream
        .clone_htod(&vec![
            1.0_f32;
            local_experts * (hidden / 128) * (intermediate / 128)
        ])
        .expect("copy fc2 scales");
    let mut out = stream
        .alloc_zeros::<u16>(tokens * hidden)
        .expect("allocate BF16 output");

    trtllm_gen_fp8_block_scale_moe_sm100_cudarc(
        stream.as_ref(),
        &routing_logits,
        Some(&routing_bias),
        &hidden_states,
        &hidden_states_scale,
        &fc1_weights,
        &fc1_scales,
        &fc2_weights,
        &fc2_scales,
        &mut out,
        tokens,
        hidden,
        intermediate,
        global_experts,
        8,
        1,
        1,
        0,
        local_experts,
        1.0,
        TrtllmGenFp8BlockScaleMoeSm100CudarcOptions::default(),
    )
    .expect("launch TensorRT-LLM Gen FP8 block-scale MoE");

    stream.synchronize().expect("synchronize");
    let output = stream.clone_dtoh(&out).expect("copy output to host");
    assert!(
        output.iter().all(|value| *value == 0),
        "zero expert weights must produce zero BF16 output"
    );
}
