#![cfg(feature = "cudarc")]

use cudarc::driver::{CudaContext, CudaSlice};
use flashinfer_rs::{DType, gdn_prefill_sm90_cudarc, gdn_prefill_sm90_cudarc_with_options};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn encode_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

fn should_run_sm90_tests(ctx: &CudaContext) -> bool {
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major != 9 {
        eprintln!(
            "skipping gdn_prefill smoke test on unsupported compute capability {}.{} (requires SM90)",
            major, minor
        );
        return false;
    }
    true
}

#[test]
fn gpu_smoke_launch_gdn_prefill_sm90() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    if !should_run_sm90_tests(&ctx) {
        return;
    }
    let stream = ctx.new_stream().expect("create stream");

    let packed_seq = 64_usize;
    let num_q_heads = 2_usize;
    let num_k_heads = 2_usize;
    let num_v_heads = 2_usize;
    let head_size = 64_usize;
    let num_sab_heads = num_q_heads.max(num_v_heads);

    let qkv_len = packed_seq * num_q_heads * head_size;
    let output_len = packed_seq * num_sab_heads * head_size;
    let output_state_len = num_sab_heads * head_size * head_size;

    let q_host: Vec<u16> = (0..qkv_len)
        .map(|i| encode_f16(((i % 29) as f32 - 14.0) * 0.03125))
        .collect();
    let k_host: Vec<u16> = (0..qkv_len)
        .map(|i| encode_f16(((i % 31) as f32 - 15.0) * 0.03125))
        .collect();
    let v_host: Vec<u16> = (0..qkv_len)
        .map(|i| encode_f16(((i % 37) as f32 - 18.0) * 0.03125))
        .collect();
    let cu_seqlens_host = vec![0_i64, packed_seq as i64];

    let q_dev = stream.clone_htod(&q_host).expect("copy q");
    let k_dev = stream.clone_htod(&k_host).expect("copy k");
    let v_dev = stream.clone_htod(&v_host).expect("copy v");
    let cu_seqlens_dev = stream
        .clone_htod(&cu_seqlens_host)
        .expect("copy cu_seqlens");
    let mut output_dev = stream.alloc_zeros::<u16>(output_len).expect("alloc output");
    let mut output_state_dev = stream
        .alloc_zeros::<f32>(output_state_len)
        .expect("alloc output_state");
    let mut workspace_dev = stream.alloc_zeros::<u8>(1 << 20).expect("alloc workspace");

    gdn_prefill_sm90_cudarc(
        stream.as_ref(),
        &mut output_dev,
        &mut output_state_dev,
        &q_dev,
        &k_dev,
        &v_dev,
        &cu_seqlens_dev,
        &mut workspace_dev,
        packed_seq,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        DType::F16,
    )
    .expect("launch gdn prefill");

    stream.synchronize().expect("synchronize");
}

#[test]
fn gpu_smoke_launch_gdn_prefill_sm90_with_checkpointing() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    if !should_run_sm90_tests(&ctx) {
        return;
    }
    let stream = ctx.new_stream().expect("create stream");

    let packed_seq = 64_usize;
    let num_q_heads = 2_usize;
    let num_k_heads = 2_usize;
    let num_v_heads = 2_usize;
    let head_size = 64_usize;
    let num_sab_heads = num_q_heads.max(num_v_heads);
    let checkpoint_every_n_tokens = 64_i64;

    let qkv_len = packed_seq * num_q_heads * head_size;
    let output_len = packed_seq * num_sab_heads * head_size;
    let output_state_len = num_sab_heads * head_size * head_size;
    let checkpoint_stride = num_sab_heads * head_size * head_size;
    let total_checkpoints = packed_seq / checkpoint_every_n_tokens as usize;
    let state_checkpoints_len = total_checkpoints * checkpoint_stride;

    let q_host: Vec<u16> = (0..qkv_len)
        .map(|i| encode_f16(((i % 23) as f32 - 11.0) * 0.03125))
        .collect();
    let k_host: Vec<u16> = (0..qkv_len)
        .map(|i| encode_f16(((i % 19) as f32 - 9.0) * 0.03125))
        .collect();
    let v_host: Vec<u16> = (0..qkv_len)
        .map(|i| encode_f16(((i % 17) as f32 - 8.0) * 0.03125))
        .collect();
    let cu_seqlens_host = vec![0_i64, packed_seq as i64];
    let checkpoint_cu_starts_host = vec![0_i64, total_checkpoints as i64];

    let q_dev = stream.clone_htod(&q_host).expect("copy q");
    let k_dev = stream.clone_htod(&k_host).expect("copy k");
    let v_dev = stream.clone_htod(&v_host).expect("copy v");
    let cu_seqlens_dev = stream
        .clone_htod(&cu_seqlens_host)
        .expect("copy cu_seqlens");
    let checkpoint_cu_starts_dev = stream
        .clone_htod(&checkpoint_cu_starts_host)
        .expect("copy checkpoint_cu_starts");
    let mut output_dev = stream.alloc_zeros::<u16>(output_len).expect("alloc output");
    let mut output_state_dev = stream
        .alloc_zeros::<f32>(output_state_len)
        .expect("alloc output_state");
    let mut workspace_dev = stream.alloc_zeros::<u8>(1 << 20).expect("alloc workspace");
    let mut state_checkpoints_dev = stream
        .alloc_zeros::<f32>(state_checkpoints_len)
        .expect("alloc state_checkpoints");

    let no_input_state: Option<&CudaSlice<f32>> = None;
    let no_alpha: Option<&CudaSlice<f32>> = None;
    let no_beta: Option<&CudaSlice<f32>> = None;

    gdn_prefill_sm90_cudarc_with_options(
        stream.as_ref(),
        &mut output_dev,
        &mut output_state_dev,
        &q_dev,
        &k_dev,
        &v_dev,
        &cu_seqlens_dev,
        &mut workspace_dev,
        packed_seq,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        DType::F16,
        no_input_state,
        no_alpha,
        no_beta,
        Some(&mut state_checkpoints_dev),
        Some(&checkpoint_cu_starts_dev),
        checkpoint_every_n_tokens,
        0.0,
    )
    .expect("launch gdn prefill with checkpointing");

    stream.synchronize().expect("synchronize");
}
