#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{
    SAMPLING_WORKSPACE_BYTES, SamplingCudarcRandom, chain_speculative_sampling_cudarc,
    min_p_sampling_from_probs_cudarc, sampling_from_logits_cudarc, sampling_from_probs_cudarc,
    sampling_softmax_cudarc, top_k_mask_logits_cudarc, top_k_renorm_probs_cudarc,
    top_k_sampling_from_probs_cudarc, top_k_top_p_sampling_from_probs_cudarc,
    top_p_renorm_probs_cudarc, top_p_sampling_from_probs_cudarc,
};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn sampling_wheel_symbols_launch_smoke() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU smoke test; set FLASHINFER_RS_RUN_GPU_TESTS=1");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");
    let batch = 2;
    let vocab = 4;

    let probs = stream
        .clone_htod(&[1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        .expect("copy probabilities");
    let logits = stream
        .clone_htod(&[
            -100.0_f32, -100.0, -100.0, 100.0, -100.0, 100.0, -100.0, -100.0,
        ])
        .expect("copy logits");
    let row_indices = stream.clone_htod(&[1_i32, 0]).expect("copy row indices");
    let top_p_arr = stream
        .clone_htod(&[0.9_f32, 0.9])
        .expect("copy top-p array");
    let top_k_arr = stream.clone_htod(&[1_i32, 1]).expect("copy top-k array");
    let min_p_arr = stream
        .clone_htod(&[0.1_f32, 0.1])
        .expect("copy min-p array");
    let temperature_arr = stream
        .clone_htod(&[1.0_f32, 1.0])
        .expect("copy temperature array");
    let seed_arr = stream.clone_htod(&[7_u64, 8]).expect("copy seed array");
    let offset_arr = stream.clone_htod(&[0_u64, 32]).expect("copy offset array");
    let random = SamplingCudarcRandom {
        deterministic: true,
        seed: 7,
        offset: 0,
        seed_arr: Some(&seed_arr),
        offset_arr: Some(&offset_arr),
    };

    let mut sample_probs = stream.alloc_zeros::<i32>(batch).expect("allocate output");
    sampling_from_probs_cudarc(
        stream.as_ref(),
        &probs,
        &mut sample_probs,
        batch,
        vocab,
        Some(&row_indices),
        random,
    )
    .expect("launch sampling_from_probs");

    let mut sample_logits = stream.alloc_zeros::<i32>(batch).expect("allocate output");
    sampling_from_logits_cudarc(
        stream.as_ref(),
        &logits,
        &mut sample_logits,
        batch,
        vocab,
        None,
        SamplingCudarcRandom::new(9, 0).with_arrays(&seed_arr, &offset_arr),
    )
    .expect("launch sampling_from_logits");

    let mut sample_top_p = stream.alloc_zeros::<i32>(batch).expect("allocate output");
    top_p_sampling_from_probs_cudarc(
        stream.as_ref(),
        &probs,
        &mut sample_top_p,
        batch,
        vocab,
        None,
        0.9,
        Some(&top_p_arr),
        SamplingCudarcRandom::new(10, 0).with_arrays(&seed_arr, &offset_arr),
    )
    .expect("launch top_p_sampling_from_probs");

    let mut sample_top_k = stream.alloc_zeros::<i32>(batch).expect("allocate output");
    top_k_sampling_from_probs_cudarc(
        stream.as_ref(),
        &probs,
        &mut sample_top_k,
        batch,
        vocab,
        None,
        1,
        Some(&top_k_arr),
        SamplingCudarcRandom::new(11, 0).with_arrays(&seed_arr, &offset_arr),
    )
    .expect("launch top_k_sampling_from_probs");

    let mut sample_min_p = stream.alloc_zeros::<i32>(batch).expect("allocate output");
    min_p_sampling_from_probs_cudarc(
        stream.as_ref(),
        &probs,
        &mut sample_min_p,
        batch,
        vocab,
        None,
        0.1,
        Some(&min_p_arr),
        SamplingCudarcRandom::new(12, 0).with_arrays(&seed_arr, &offset_arr),
    )
    .expect("launch min_p_sampling_from_probs");

    let mut sample_top_k_top_p = stream.alloc_zeros::<i32>(batch).expect("allocate output");
    top_k_top_p_sampling_from_probs_cudarc(
        stream.as_ref(),
        &probs,
        &mut sample_top_k_top_p,
        batch,
        vocab,
        None,
        1,
        Some(&top_k_arr),
        0.9,
        Some(&top_p_arr),
        SamplingCudarcRandom::new(13, 0).with_arrays(&seed_arr, &offset_arr),
    )
    .expect("launch top_k_top_p_sampling_from_probs");

    let num_speculative_tokens = 2;
    let draft_probs = stream
        .clone_htod(&[
            // Batch row 0: draft tokens 0, 1.
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // Batch row 1: draft tokens 2, 3.
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ])
        .expect("copy draft probabilities");
    let draft_token_ids = stream
        .clone_htod(&[0_i32, 1, 2, 3])
        .expect("copy draft token IDs");
    let target_probs = stream
        .clone_htod(&[
            // Batch row 0 accepts both drafts and emits bonus token 2.
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            // Batch row 1 accepts token 2, rejects token 3, and samples token 1.
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ])
        .expect("copy target probabilities");
    let mut speculative_token_ids = stream
        .alloc_zeros::<i32>(batch * (num_speculative_tokens + 1))
        .expect("allocate speculative token output");
    let mut accepted_token_num = stream
        .alloc_zeros::<i32>(batch)
        .expect("allocate accepted counters");
    let mut emitted_draft_token_num = stream
        .alloc_zeros::<i32>(batch)
        .expect("allocate emitted counters");
    chain_speculative_sampling_cudarc(
        stream.as_ref(),
        &draft_probs,
        &draft_token_ids,
        &target_probs,
        &mut speculative_token_ids,
        &mut accepted_token_num,
        &mut emitted_draft_token_num,
        batch,
        num_speculative_tokens,
        vocab,
        SamplingCudarcRandom::new(14, 0).with_arrays(&seed_arr, &offset_arr),
    )
    .expect("launch chain_speculative_sampling");

    let mut workspace = stream
        .alloc_zeros::<u8>(SAMPLING_WORKSPACE_BYTES)
        .expect("allocate caller-owned workspace");
    let mut softmax_out = stream
        .alloc_zeros::<f32>(batch * vocab)
        .expect("allocate softmax output");
    sampling_softmax_cudarc(
        stream.as_ref(),
        &logits,
        &mut softmax_out,
        &mut workspace,
        batch,
        vocab,
        1.0,
        Some(&temperature_arr),
        false,
    )
    .expect("launch softmax");

    let mut top_p_renorm_out = stream
        .alloc_zeros::<f32>(batch * vocab)
        .expect("allocate top-p output");
    top_p_renorm_probs_cudarc(
        stream.as_ref(),
        &probs,
        &mut top_p_renorm_out,
        batch,
        vocab,
        0.9,
        Some(&top_p_arr),
    )
    .expect("launch top_p_renorm_probs");

    let mut top_k_renorm_out = stream
        .alloc_zeros::<f32>(batch * vocab)
        .expect("allocate top-k output");
    top_k_renorm_probs_cudarc(
        stream.as_ref(),
        &probs,
        &mut top_k_renorm_out,
        &mut workspace,
        batch,
        vocab,
        1,
        None,
    )
    .expect("launch top_k_renorm_probs");

    let mut mask_out = stream
        .alloc_zeros::<f32>(batch * vocab)
        .expect("allocate mask output");
    top_k_mask_logits_cudarc(
        stream.as_ref(),
        &logits,
        &mut mask_out,
        &mut workspace,
        batch,
        vocab,
        1,
        None,
    )
    .expect("launch top_k_mask_logits");

    stream.synchronize().expect("synchronize stream");
    assert_eq!(
        stream.clone_dtoh(&sample_probs).expect("copy output"),
        vec![2, 0]
    );
    assert_eq!(
        stream.clone_dtoh(&sample_logits).expect("copy output"),
        vec![3, 1]
    );
    for output in [
        &sample_top_p,
        &sample_top_k,
        &sample_min_p,
        &sample_top_k_top_p,
    ] {
        assert_eq!(stream.clone_dtoh(output).expect("copy output"), vec![0, 2]);
    }

    let softmax_host = stream.clone_dtoh(&softmax_out).expect("copy softmax");
    for row in softmax_host.chunks_exact(vocab) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
    assert_eq!(
        stream
            .clone_dtoh(&top_p_renorm_out)
            .expect("copy top-p renorm"),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );
    assert_eq!(
        stream
            .clone_dtoh(&top_k_renorm_out)
            .expect("copy top-k renorm"),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );
    let mask_host = stream.clone_dtoh(&mask_out).expect("copy masked logits");
    assert!(mask_host[0].is_infinite() && mask_host[0].is_sign_negative());
    assert_eq!(mask_host[3], 100.0);
    assert_eq!(mask_host[5], 100.0);
    assert_eq!(
        stream
            .clone_dtoh(&speculative_token_ids)
            .expect("copy speculative tokens"),
        vec![0, 1, 2, 2, 1, -1]
    );
    assert_eq!(
        stream
            .clone_dtoh(&accepted_token_num)
            .expect("copy accepted counters"),
        vec![2, 1]
    );
    assert_eq!(
        stream
            .clone_dtoh(&emitted_draft_token_num)
            .expect("copy emitted counters"),
        vec![2, 1]
    );
}
