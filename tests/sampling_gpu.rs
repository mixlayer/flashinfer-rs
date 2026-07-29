#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{
    DEFAULT_SAMPLING_WORKSPACE_BYTES, sampling_from_probs_cudarc, sampling_softmax_cudarc,
};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn gpu_smoke_softmax_and_sampling() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");
    let rows = 4;
    let cols = 128;
    let logits_host: Vec<f32> = (0..rows * cols)
        .map(|index| (index % cols) as f32 * 0.01)
        .collect();
    let logits = stream.clone_htod(&logits_host).expect("copy logits");
    let mut probabilities = stream
        .alloc_zeros::<f32>(rows * cols)
        .expect("allocate probabilities");
    let mut workspace = stream
        .alloc_zeros::<u8>(DEFAULT_SAMPLING_WORKSPACE_BYTES)
        .expect("allocate workspace");
    let mut output = stream.alloc_zeros::<i32>(rows).expect("allocate output");

    sampling_softmax_cudarc(
        stream.as_ref(),
        &mut workspace,
        &logits,
        &mut probabilities,
        rows,
        cols,
        None,
        0.8,
        false,
    )
    .expect("launch online softmax");
    sampling_from_probs_cudarc(
        stream.as_ref(),
        &probabilities,
        &mut output,
        rows,
        cols,
        None,
        None,
        7,
        None,
        0,
        true,
    )
    .expect("launch probability sampling");

    let sampled = stream.clone_dtoh(&output).expect("copy output");
    assert!(
        sampled
            .iter()
            .all(|&token| (0..cols as i32).contains(&token))
    );
}
