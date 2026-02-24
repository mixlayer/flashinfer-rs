#![cfg(feature = "cudarc")]

//! Stress tests for the GDN prefill (delta-rule) SM90 kernel.
//!
//! The underlying `FlatKernelTmaWarpSpecializedDeltaRule` CUTLASS kernel uses
//! complex warp-specialized scheduling with 8 producer-consumer pipelines and
//! ordered math barriers between warp groups.  These tests are designed to
//! surface synchronization and race-condition bugs by:
//!
//! - Mixing sequences of wildly different lengths (1-token through multi-tile)
//!   in a single batch, stressing the TileScheduler work distribution.
//! - Exercising all optional-tensor code paths (alpha, beta, input_state) which
//!   activate extra load-warp roles and pipeline stages.
//! - Testing both GQA (q_heads > kv_heads) and GVA (v_heads > q_heads) paths,
//!   which select entirely different kernel instantiations via the
//!   `IsGVA` template parameter.
//! - Performing rapid back-to-back kernel launches on the same stream without
//!   intermediate host synchronization, probing async launch ordering.
//! - Chaining state (output_state → input_state) with alpha+beta across
//!   launches to test the production chunked-prefill hot path.
//! - Launching from multiple concurrent CUDA streams to stress SM-level
//!   resource contention.
//! - Using a watchdog timeout to convert hangs into test failures instead of
//!   blocking CI forever.
//!
//! The production target model is Qwen3-Next-80B-A3B which uses GVA with
//! q=k=16, v=32, head_size=128, alpha+beta enabled, and state chaining
//! for chunked prefill in distributed inference.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use flashinfer_rs::{DType, gdn_prefill_sm90_cudarc, gdn_prefill_sm90_cudarc_with_options};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn encode_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

fn encode_bf16(value: f32) -> u16 {
    half::bf16::from_f32(value).to_bits()
}

const HEAD_SIZE: usize = 128;
/// Production workspace: sm_count * 128 bytes for TMA store tensormaps.
/// H100 has 132 SMs → 16896 bytes. We use 132*128 as a realistic tight size.
const WORKSPACE_BYTES_TIGHT: usize = 132 * 128;
/// Generous workspace for tests that aren't focused on workspace sizing.
const WORKSPACE_BYTES_LARGE: usize = 32 * 1024 * 1024;
const HANG_TIMEOUT: Duration = Duration::from_secs(60);

/// Build cu_seqlens (i64) on host from a slice of per-sequence lengths.
fn build_cu_seqlens(seq_lens: &[usize]) -> Vec<i64> {
    let mut cu = Vec::with_capacity(seq_lens.len() + 1);
    cu.push(0i64);
    for &len in seq_lens {
        cu.push(cu.last().unwrap() + len as i64);
    }
    cu
}

/// Deterministic non-zero f16 fill pattern that varies across elements.
fn deterministic_f16_vec(n: usize, seed: usize) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let v = (((i.wrapping_add(seed)) % 61) as f32 - 30.0) * 0.015625;
            encode_f16(v)
        })
        .collect()
}

/// Deterministic non-zero bf16 fill pattern.
fn deterministic_bf16_vec(n: usize, seed: usize) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let v = (((i.wrapping_add(seed)) % 61) as f32 - 30.0) * 0.015625;
            encode_bf16(v)
        })
        .collect()
}

/// Deterministic f32 fill in (0, 1] range suitable for alpha/beta (sigmoid output).
fn deterministic_f32_vec(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let raw = ((i.wrapping_add(seed)) % 97) as f32;
            0.01 + 0.99 * (raw / 97.0)
        })
        .collect()
}

/// Run a synchronize with a watchdog that panics on timeout (hang detection).
fn sync_with_timeout(stream: &cudarc::driver::CudaStream, label: &str) {
    let done = Arc::new(AtomicBool::new(false));
    let done2 = done.clone();
    let label = label.to_string();

    let watchdog = std::thread::spawn(move || {
        let start = Instant::now();
        while !done2.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(200));
            if start.elapsed() > HANG_TIMEOUT {
                panic!(
                    "HANG DETECTED in '{}': stream.synchronize() did not return within {:?}",
                    label, HANG_TIMEOUT
                );
            }
        }
    });

    stream.synchronize().expect("stream synchronize");
    done.store(true, Ordering::Release);
    watchdog.join().expect("watchdog join");
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

/// Mixed sequence lengths: 1, 3, 63, 64, 65, 127, 128, 129, 256 tokens.
/// This stresses the tile scheduler with sequences that are shorter than,
/// equal to, and longer than the 64-token tile boundary.
#[test]
fn stress_gdn_prefill_mixed_seqlens() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let seq_lens: Vec<usize> = vec![1, 3, 63, 64, 65, 127, 128, 129, 256];
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq: usize = *cu_seqlens_host.last().unwrap() as usize;
    let num_seqs = seq_lens.len();

    let num_q_heads = 8_usize;
    let num_kv_heads = 8_usize;

    let q_host = deterministic_f16_vec(packed_seq * num_q_heads * HEAD_SIZE, 7);
    let k_host = deterministic_f16_vec(packed_seq * num_kv_heads * HEAD_SIZE, 13);
    let v_host = deterministic_f16_vec(packed_seq * num_kv_heads * HEAD_SIZE, 29);

    let q_dev = stream.clone_htod(&q_host).expect("copy q");
    let k_dev = stream.clone_htod(&k_host).expect("copy k");
    let v_dev = stream.clone_htod(&v_host).expect("copy v");
    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    let num_sab_heads = num_q_heads.max(num_kv_heads);
    let mut output_dev = stream
        .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
        .expect("alloc output");
    let mut output_state_dev = stream
        .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
        .expect("alloc output_state");
    let mut workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
        .expect("alloc workspace");

    gdn_prefill_sm90_cudarc::<u16, _, _, _, _, _, _, _>(
        stream.as_ref(),
        &mut output_dev,
        &mut output_state_dev,
        &q_dev,
        &k_dev,
        &v_dev,
        &cu_seqlens_dev,
        &mut workspace,
        packed_seq,
        num_q_heads,
        num_kv_heads,
        num_kv_heads,
        HEAD_SIZE,
        DType::F16,
    )
    .expect("launch gdn_prefill mixed seqlens");

    sync_with_timeout(&stream, "stress_gdn_prefill_mixed_seqlens");
}

/// Many sequences of length 1 -- each sequence is a single token, exercising
/// the "final block == first block" path in every warp group simultaneously.
#[test]
fn stress_gdn_prefill_many_single_token_seqs() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let num_seqs = 256;
    let seq_lens = vec![1_usize; num_seqs];
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq = num_seqs;

    let num_q_heads = 4_usize;
    let num_kv_heads = 4_usize;
    let num_sab_heads = num_q_heads.max(num_kv_heads);

    let q_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_q_heads * HEAD_SIZE, 1))
        .expect("copy q");
    let k_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_kv_heads * HEAD_SIZE, 2))
        .expect("copy k");
    let v_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_kv_heads * HEAD_SIZE, 3))
        .expect("copy v");
    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    let mut output_dev = stream
        .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
        .expect("alloc output");
    let mut output_state_dev = stream
        .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
        .expect("alloc output_state");
    let mut workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
        .expect("alloc workspace");

    gdn_prefill_sm90_cudarc::<u16, _, _, _, _, _, _, _>(
        stream.as_ref(),
        &mut output_dev,
        &mut output_state_dev,
        &q_dev,
        &k_dev,
        &v_dev,
        &cu_seqlens_dev,
        &mut workspace,
        packed_seq,
        num_q_heads,
        num_kv_heads,
        num_kv_heads,
        HEAD_SIZE,
        DType::F16,
    )
    .expect("launch gdn_prefill single-token seqs");

    sync_with_timeout(&stream, "stress_gdn_prefill_many_single_token_seqs");
}

/// Rapid back-to-back launches on the same stream with NO intermediate sync.
/// Uses the Qwen3-Next GVA config (q=k=16, v=32) with alpha+beta enabled,
/// matching the production hot path.
#[test]
fn stress_gdn_prefill_rapid_fire_no_sync() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let seq_lens = vec![64_usize, 128, 32];
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
    let num_seqs = seq_lens.len();
    let num_q_heads = 16_usize;
    let num_k_heads = 16_usize;
    let num_v_heads = 32_usize;
    let num_sab_heads = num_q_heads.max(num_v_heads);

    let q_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_q_heads * HEAD_SIZE, 100))
        .expect("copy q");
    let k_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_k_heads * HEAD_SIZE, 200))
        .expect("copy k");
    let v_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_v_heads * HEAD_SIZE, 300))
        .expect("copy v");
    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    let alpha_host = deterministic_f32_vec(packed_seq * num_sab_heads, 400);
    let alpha_dev = stream.clone_htod(&alpha_host).expect("copy alpha");
    let beta_host = deterministic_f32_vec(packed_seq * num_sab_heads, 500);
    let beta_dev = stream.clone_htod(&beta_host).expect("copy beta");

    let mut output_dev = stream
        .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
        .expect("alloc output");
    let mut output_state_dev = stream
        .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
        .expect("alloc output_state");
    let mut workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
        .expect("alloc workspace");

    let num_launches = 50;
    for _ in 0..num_launches {
        gdn_prefill_sm90_cudarc_with_options::<u16, _, _, _, _, _, _, _, _, _, _>(
            stream.as_ref(),
            &mut output_dev,
            &mut output_state_dev,
            &q_dev,
            &k_dev,
            &v_dev,
            &cu_seqlens_dev,
            &mut workspace,
            packed_seq,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            HEAD_SIZE,
            DType::F16,
            None::<&cudarc::driver::CudaSlice<f32>>,
            Some(&alpha_dev),
            Some(&beta_dev),
            0.0,
        )
        .expect("launch gdn_prefill rapid fire");
    }

    sync_with_timeout(&stream, "stress_gdn_prefill_rapid_fire_no_sync");
}

/// All optional tensors enabled with GQA config: alpha, beta, and input_state.
/// This activates the LoadAlpha, LoadBeta warp roles and the
/// `kInitStateFromInput` compute path.
#[test]
fn stress_gdn_prefill_all_options_gqa() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let seq_lens = vec![1, 33, 64, 65, 127, 128, 200];
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
    let num_seqs = seq_lens.len();

    let num_q_heads = 6_usize;
    let num_k_heads = 2_usize;
    let num_v_heads = 2_usize;
    let num_sab_heads = num_q_heads.max(num_v_heads);

    let q_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_q_heads * HEAD_SIZE, 11))
        .expect("copy q");
    let k_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_k_heads * HEAD_SIZE, 22))
        .expect("copy k");
    let v_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_v_heads * HEAD_SIZE, 33))
        .expect("copy v");
    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    let input_state_host =
        deterministic_f32_vec(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE, 44);
    let input_state_dev = stream.clone_htod(&input_state_host).expect("copy input_state");

    let alpha_host = deterministic_f32_vec(packed_seq * num_sab_heads, 55);
    let alpha_dev = stream.clone_htod(&alpha_host).expect("copy alpha");

    let beta_host = deterministic_f32_vec(packed_seq * num_sab_heads, 66);
    let beta_dev = stream.clone_htod(&beta_host).expect("copy beta");

    let mut output_dev = stream
        .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
        .expect("alloc output");
    let mut output_state_dev = stream
        .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
        .expect("alloc output_state");
    let mut workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
        .expect("alloc workspace");

    gdn_prefill_sm90_cudarc_with_options::<u16, _, _, _, _, _, _, _, _, _, _>(
        stream.as_ref(),
        &mut output_dev,
        &mut output_state_dev,
        &q_dev,
        &k_dev,
        &v_dev,
        &cu_seqlens_dev,
        &mut workspace,
        packed_seq,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        HEAD_SIZE,
        DType::F16,
        Some(&input_state_dev),
        Some(&alpha_dev),
        Some(&beta_dev),
        1.0,
    )
    .expect("launch gdn_prefill all options GQA");

    sync_with_timeout(&stream, "stress_gdn_prefill_all_options_gqa");
}

/// All optional tensors enabled with GVA config (num_v_heads > num_q_heads).
/// This selects the `IsGVA=true` kernel instantiation, which is the
/// Qwen3-Next production path.
#[test]
fn stress_gdn_prefill_all_options_gva() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let seq_lens = vec![1, 33, 64, 65, 127, 128, 200];
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
    let num_seqs = seq_lens.len();

    let num_q_heads = 2_usize;
    let num_k_heads = 2_usize;
    let num_v_heads = 4_usize;
    let num_sab_heads = num_q_heads.max(num_v_heads);

    let q_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_q_heads * HEAD_SIZE, 11))
        .expect("copy q");
    let k_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_k_heads * HEAD_SIZE, 22))
        .expect("copy k");
    let v_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_v_heads * HEAD_SIZE, 33))
        .expect("copy v");
    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    let input_state_host =
        deterministic_f32_vec(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE, 44);
    let input_state_dev = stream.clone_htod(&input_state_host).expect("copy input_state");

    let alpha_host = deterministic_f32_vec(packed_seq * num_sab_heads, 55);
    let alpha_dev = stream.clone_htod(&alpha_host).expect("copy alpha");

    let beta_host = deterministic_f32_vec(packed_seq * num_sab_heads, 66);
    let beta_dev = stream.clone_htod(&beta_host).expect("copy beta");

    let mut output_dev = stream
        .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
        .expect("alloc output");
    let mut output_state_dev = stream
        .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
        .expect("alloc output_state");
    let mut workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
        .expect("alloc workspace");

    gdn_prefill_sm90_cudarc_with_options::<u16, _, _, _, _, _, _, _, _, _, _>(
        stream.as_ref(),
        &mut output_dev,
        &mut output_state_dev,
        &q_dev,
        &k_dev,
        &v_dev,
        &cu_seqlens_dev,
        &mut workspace,
        packed_seq,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        HEAD_SIZE,
        DType::F16,
        Some(&input_state_dev),
        Some(&alpha_dev),
        Some(&beta_dev),
        0.0,
    )
    .expect("launch gdn_prefill all options GVA");

    sync_with_timeout(&stream, "stress_gdn_prefill_all_options_gva");
}

/// State chaining with alpha+beta: the production modeld hot path.
/// Runs the kernel repeatedly, feeding output_state → input_state with
/// alpha+beta enabled throughout, using the Qwen3-Next GVA config.
#[test]
fn stress_gdn_prefill_state_chain_with_gates() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let seq_lens = vec![61_usize, 128, 251];
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
    let num_seqs = seq_lens.len();

    let num_q_heads = 16_usize;
    let num_k_heads = 16_usize;
    let num_v_heads = 32_usize;
    let num_sab_heads = num_q_heads.max(num_v_heads);
    let state_elems = num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE;

    let q_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_q_heads * HEAD_SIZE, 7))
        .expect("copy q");
    let k_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_k_heads * HEAD_SIZE, 11))
        .expect("copy k");
    let v_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_v_heads * HEAD_SIZE, 13))
        .expect("copy v");
    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    let alpha_host = deterministic_f32_vec(packed_seq * num_sab_heads, 17);
    let alpha_dev = stream.clone_htod(&alpha_host).expect("copy alpha");
    let beta_host = deterministic_f32_vec(packed_seq * num_sab_heads, 19);
    let beta_dev = stream.clone_htod(&beta_host).expect("copy beta");

    let mut output_dev = stream
        .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
        .expect("alloc output");
    let mut state_a = stream.alloc_zeros::<f32>(state_elems).expect("alloc state_a");
    let mut state_b = stream.alloc_zeros::<f32>(state_elems).expect("alloc state_b");
    let mut workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
        .expect("alloc workspace");

    let chain_steps = 20;
    for step in 0..chain_steps {
        let (input_state, output_state) = if step % 2 == 0 {
            (&state_a, &mut state_b)
        } else {
            (&state_b, &mut state_a)
        };

        let input_state_opt: Option<&cudarc::driver::CudaSlice<f32>> = if step > 0 {
            Some(input_state)
        } else {
            None
        };

        gdn_prefill_sm90_cudarc_with_options::<u16, _, _, _, _, _, _, _, _, _, _>(
            stream.as_ref(),
            &mut output_dev,
            output_state,
            &q_dev,
            &k_dev,
            &v_dev,
            &cu_seqlens_dev,
            &mut workspace,
            packed_seq,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            HEAD_SIZE,
            DType::F16,
            input_state_opt,
            Some(&alpha_dev),
            Some(&beta_dev),
            0.0,
        )
        .expect(&format!("launch gdn_prefill chain step {step}"));
    }

    sync_with_timeout(&stream, "stress_gdn_prefill_state_chain_with_gates");
}

/// Multi-stream concurrent launches.  Each stream independently launches
/// the kernel with different sequence configurations, stressing SM-level
/// resource contention (shared memory, registers, barriers) across
/// concurrent thread blocks from different kernel instances.
#[test]
fn stress_gdn_prefill_multi_stream() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");

    // (seq_lens, num_q_heads, num_k_heads, num_v_heads)
    let configs: Vec<(Vec<usize>, usize, usize, usize)> = vec![
        (vec![64, 128, 64], 8, 8, 8),       // equal heads
        (vec![1, 1, 1, 1, 256], 4, 4, 4),   // many tiny + one long
        (vec![33, 65, 129], 6, 2, 2),        // GQA
        (vec![128; 8], 16, 16, 32),          // GVA (Qwen3-Next)
    ];

    let handles: Vec<_> = configs
        .into_iter()
        .enumerate()
        .map(|(i, (seq_lens, num_q_heads, num_k_heads, num_v_heads))| {
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let stream = ctx.new_stream().expect("create stream");

                let cu_seqlens_host = build_cu_seqlens(&seq_lens);
                let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
                let num_seqs = seq_lens.len();
                let num_sab_heads = num_q_heads.max(num_v_heads);

                let q_dev = stream
                    .clone_htod(&deterministic_f16_vec(
                        packed_seq * num_q_heads * HEAD_SIZE,
                        i * 1000 + 1,
                    ))
                    .expect("copy q");
                let k_dev = stream
                    .clone_htod(&deterministic_f16_vec(
                        packed_seq * num_k_heads * HEAD_SIZE,
                        i * 1000 + 2,
                    ))
                    .expect("copy k");
                let v_dev = stream
                    .clone_htod(&deterministic_f16_vec(
                        packed_seq * num_v_heads * HEAD_SIZE,
                        i * 1000 + 3,
                    ))
                    .expect("copy v");
                let cu_seqlens_dev =
                    stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

                let mut output_dev = stream
                    .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
                    .expect("alloc output");
                let mut output_state_dev = stream
                    .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
                    .expect("alloc output_state");
                let mut workspace = stream
                    .alloc_zeros::<u8>(WORKSPACE_BYTES_LARGE)
                    .expect("alloc workspace");

                let launches_per_stream = 10;
                for _ in 0..launches_per_stream {
                    gdn_prefill_sm90_cudarc::<u16, _, _, _, _, _, _, _>(
                        stream.as_ref(),
                        &mut output_dev,
                        &mut output_state_dev,
                        &q_dev,
                        &k_dev,
                        &v_dev,
                        &cu_seqlens_dev,
                        &mut workspace,
                        packed_seq,
                        num_q_heads,
                        num_k_heads,
                        num_v_heads,
                        HEAD_SIZE,
                        DType::F16,
                    )
                    .expect("launch gdn_prefill multi-stream");
                }

                sync_with_timeout(
                    &stream,
                    &format!("stress_gdn_prefill_multi_stream[{i}]"),
                );
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        h.join()
            .unwrap_or_else(|e| panic!("multi-stream thread {i} panicked: {e:?}"));
    }
}

/// GQA configurations: num_q_heads > num_kv_heads.
/// Exercises the head-ratio broadcast path in the tile scheduler.
#[test]
fn stress_gdn_prefill_gqa_head_ratios() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    // (num_q_heads, num_kv_heads) -- all GQA (q >= kv)
    let gqa_configs: Vec<(usize, usize)> = vec![
        (4, 1),
        (6, 2),
        (8, 2),
        (8, 4),
        (16, 4),
    ];

    let seq_lens = vec![1, 63, 64, 65, 128, 129];
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
    let num_seqs = seq_lens.len();

    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    for (num_q_heads, num_kv_heads) in gqa_configs {
        let num_sab_heads = num_q_heads.max(num_kv_heads);

        let q_dev = stream
            .clone_htod(&deterministic_f16_vec(
                packed_seq * num_q_heads * HEAD_SIZE,
                num_q_heads * 100,
            ))
            .expect("copy q");
        let k_dev = stream
            .clone_htod(&deterministic_f16_vec(
                packed_seq * num_kv_heads * HEAD_SIZE,
                num_kv_heads * 100,
            ))
            .expect("copy k");
        let v_dev = stream
            .clone_htod(&deterministic_f16_vec(
                packed_seq * num_kv_heads * HEAD_SIZE,
                num_kv_heads * 200,
            ))
            .expect("copy v");

        let mut output_dev = stream
            .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
            .expect("alloc output");
        let mut output_state_dev = stream
            .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
            .expect("alloc output_state");
        let mut workspace = stream
            .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
            .expect("alloc workspace");

        gdn_prefill_sm90_cudarc::<u16, _, _, _, _, _, _, _>(
            stream.as_ref(),
            &mut output_dev,
            &mut output_state_dev,
            &q_dev,
            &k_dev,
            &v_dev,
            &cu_seqlens_dev,
            &mut workspace,
            packed_seq,
            num_q_heads,
            num_kv_heads,
            num_kv_heads,
            HEAD_SIZE,
            DType::F16,
        )
        .expect(&format!(
            "launch gdn_prefill GQA q={num_q_heads} kv={num_kv_heads}"
        ));
    }

    sync_with_timeout(&stream, "stress_gdn_prefill_gqa_head_ratios");
}

/// GVA configurations: num_v_heads > num_q_heads (Grouped Value Attention).
/// This selects the `IsGVA=true` kernel instantiation, which is a completely
/// different code path from GQA. Qwen3-Next uses GVA.
#[test]
fn stress_gdn_prefill_gva_head_ratios() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    // (num_q_heads, num_k_heads, num_v_heads) -- all GVA (v > q, q == k)
    let gva_configs: Vec<(usize, usize, usize)> = vec![
        (1, 1, 2),
        (2, 2, 4),
        (4, 4, 8),
        (16, 16, 32),
    ];

    let seq_lens = vec![1, 63, 64, 65, 128, 129];
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
    let num_seqs = seq_lens.len();

    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    for (num_q_heads, num_k_heads, num_v_heads) in gva_configs {
        let num_sab_heads = num_q_heads.max(num_v_heads);

        let q_dev = stream
            .clone_htod(&deterministic_f16_vec(
                packed_seq * num_q_heads * HEAD_SIZE,
                num_q_heads * 100 + 1,
            ))
            .expect("copy q");
        let k_dev = stream
            .clone_htod(&deterministic_f16_vec(
                packed_seq * num_k_heads * HEAD_SIZE,
                num_k_heads * 100 + 2,
            ))
            .expect("copy k");
        let v_dev = stream
            .clone_htod(&deterministic_f16_vec(
                packed_seq * num_v_heads * HEAD_SIZE,
                num_v_heads * 100 + 3,
            ))
            .expect("copy v");

        let mut output_dev = stream
            .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
            .expect("alloc output");
        let mut output_state_dev = stream
            .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
            .expect("alloc output_state");
        let mut workspace = stream
            .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
            .expect("alloc workspace");

        gdn_prefill_sm90_cudarc::<u16, _, _, _, _, _, _, _>(
            stream.as_ref(),
            &mut output_dev,
            &mut output_state_dev,
            &q_dev,
            &k_dev,
            &v_dev,
            &cu_seqlens_dev,
            &mut workspace,
            packed_seq,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            HEAD_SIZE,
            DType::F16,
        )
        .expect(&format!(
            "launch gdn_prefill GVA q={num_q_heads} k={num_k_heads} v={num_v_heads}"
        ));
    }

    sync_with_timeout(&stream, "stress_gdn_prefill_gva_head_ratios");
}

/// Large batch with extreme sequence length disparity.
#[test]
fn stress_gdn_prefill_extreme_disparity() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let mut seq_lens = vec![1_usize; 64];
    seq_lens.push(1024);
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
    let num_seqs = seq_lens.len();

    let num_q_heads = 16_usize;
    let num_k_heads = 16_usize;
    let num_v_heads = 32_usize;
    let num_sab_heads = num_q_heads.max(num_v_heads);

    let q_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_q_heads * HEAD_SIZE, 17))
        .expect("copy q");
    let k_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_k_heads * HEAD_SIZE, 19))
        .expect("copy k");
    let v_dev = stream
        .clone_htod(&deterministic_f16_vec(packed_seq * num_v_heads * HEAD_SIZE, 23))
        .expect("copy v");
    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    let mut output_dev = stream
        .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
        .expect("alloc output");
    let mut output_state_dev = stream
        .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
        .expect("alloc output_state");
    let mut workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
        .expect("alloc workspace");

    gdn_prefill_sm90_cudarc::<u16, _, _, _, _, _, _, _>(
        stream.as_ref(),
        &mut output_dev,
        &mut output_state_dev,
        &q_dev,
        &k_dev,
        &v_dev,
        &cu_seqlens_dev,
        &mut workspace,
        packed_seq,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        HEAD_SIZE,
        DType::F16,
    )
    .expect("launch gdn_prefill extreme disparity");

    sync_with_timeout(&stream, "stress_gdn_prefill_extreme_disparity");
}

/// BF16 dtype with GVA config and all options.
/// The upstream kernel has separate template instantiations for f16 and bf16.
#[test]
fn stress_gdn_prefill_bf16_gva_all_options() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let seq_lens = vec![31, 63, 93, 123, 150, 500];
    let cu_seqlens_host = build_cu_seqlens(&seq_lens);
    let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
    let num_seqs = seq_lens.len();

    let num_q_heads = 16_usize;
    let num_k_heads = 16_usize;
    let num_v_heads = 32_usize;
    let num_sab_heads = num_q_heads.max(num_v_heads);

    let q_dev = stream
        .clone_htod(&deterministic_bf16_vec(packed_seq * num_q_heads * HEAD_SIZE, 77))
        .expect("copy q");
    let k_dev = stream
        .clone_htod(&deterministic_bf16_vec(packed_seq * num_k_heads * HEAD_SIZE, 88))
        .expect("copy k");
    let v_dev = stream
        .clone_htod(&deterministic_bf16_vec(packed_seq * num_v_heads * HEAD_SIZE, 99))
        .expect("copy v");
    let cu_seqlens_dev = stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

    let input_state_host =
        deterministic_f32_vec(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE, 111);
    let input_state_dev = stream.clone_htod(&input_state_host).expect("copy input_state");

    let alpha_host = deterministic_f32_vec(packed_seq * num_sab_heads, 222);
    let alpha_dev = stream.clone_htod(&alpha_host).expect("copy alpha");

    let beta_host = deterministic_f32_vec(packed_seq * num_sab_heads, 333);
    let beta_dev = stream.clone_htod(&beta_host).expect("copy beta");

    let mut output_dev = stream
        .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
        .expect("alloc output");
    let mut output_state_dev = stream
        .alloc_zeros::<f32>(num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE)
        .expect("alloc output_state");
    let mut workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
        .expect("alloc workspace");

    gdn_prefill_sm90_cudarc_with_options::<u16, _, _, _, _, _, _, _, _, _, _>(
        stream.as_ref(),
        &mut output_dev,
        &mut output_state_dev,
        &q_dev,
        &k_dev,
        &v_dev,
        &cu_seqlens_dev,
        &mut workspace,
        packed_seq,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        HEAD_SIZE,
        DType::BF16,
        Some(&input_state_dev),
        Some(&alpha_dev),
        Some(&beta_dev),
        0.0,
    )
    .expect("launch gdn_prefill bf16 GVA all options");

    sync_with_timeout(&stream, "stress_gdn_prefill_bf16_gva_all_options");
}

/// Exact Qwen3-Next production config: q=k=16, v=32 (GVA), head_size=128,
/// alpha+beta enabled, state chaining, mixed sequence lengths from upstream
/// test suite, with concurrent multi-stream execution.
/// This is the closest approximation to the distributed modeld workload.
#[test]
fn stress_gdn_prefill_qwen3_next_multi_stream_chained() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let num_concurrent = 4_usize;
    let chain_steps = 10;

    let handles: Vec<_> = (0..num_concurrent)
        .map(|stream_idx| {
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let stream = ctx.new_stream().expect("create stream");

                let seq_lens: Vec<usize> = vec![61, 128, 251, 511];
                let cu_seqlens_host = build_cu_seqlens(&seq_lens);
                let packed_seq = *cu_seqlens_host.last().unwrap() as usize;
                let num_seqs = seq_lens.len();

                let num_q_heads = 16_usize;
                let num_k_heads = 16_usize;
                let num_v_heads = 32_usize;
                let num_sab_heads = num_q_heads.max(num_v_heads);
                let state_elems = num_seqs * num_sab_heads * HEAD_SIZE * HEAD_SIZE;

                let q_dev = stream
                    .clone_htod(&deterministic_f16_vec(
                        packed_seq * num_q_heads * HEAD_SIZE,
                        stream_idx * 10000 + 1,
                    ))
                    .expect("copy q");
                let k_dev = stream
                    .clone_htod(&deterministic_f16_vec(
                        packed_seq * num_k_heads * HEAD_SIZE,
                        stream_idx * 10000 + 2,
                    ))
                    .expect("copy k");
                let v_dev = stream
                    .clone_htod(&deterministic_f16_vec(
                        packed_seq * num_v_heads * HEAD_SIZE,
                        stream_idx * 10000 + 3,
                    ))
                    .expect("copy v");
                let cu_seqlens_dev =
                    stream.clone_htod(&cu_seqlens_host).expect("copy cu_seqlens");

                let alpha_host =
                    deterministic_f32_vec(packed_seq * num_sab_heads, stream_idx * 10000 + 4);
                let alpha_dev = stream.clone_htod(&alpha_host).expect("copy alpha");
                let beta_host =
                    deterministic_f32_vec(packed_seq * num_sab_heads, stream_idx * 10000 + 5);
                let beta_dev = stream.clone_htod(&beta_host).expect("copy beta");

                let mut output_dev = stream
                    .alloc_zeros::<u16>(packed_seq * num_sab_heads * HEAD_SIZE)
                    .expect("alloc output");
                let mut state_a = stream.alloc_zeros::<f32>(state_elems).expect("alloc state_a");
                let mut state_b = stream.alloc_zeros::<f32>(state_elems).expect("alloc state_b");
                let mut workspace = stream
                    .alloc_zeros::<u8>(WORKSPACE_BYTES_TIGHT)
                    .expect("alloc workspace");

                for step in 0..chain_steps {
                    let (input_state, output_state) = if step % 2 == 0 {
                        (&state_a, &mut state_b)
                    } else {
                        (&state_b, &mut state_a)
                    };

                    let input_state_opt: Option<&cudarc::driver::CudaSlice<f32>> =
                        if step > 0 { Some(input_state) } else { None };

                    gdn_prefill_sm90_cudarc_with_options::<u16, _, _, _, _, _, _, _, _, _, _>(
                        stream.as_ref(),
                        &mut output_dev,
                        output_state,
                        &q_dev,
                        &k_dev,
                        &v_dev,
                        &cu_seqlens_dev,
                        &mut workspace,
                        packed_seq,
                        num_q_heads,
                        num_k_heads,
                        num_v_heads,
                        HEAD_SIZE,
                        DType::F16,
                        input_state_opt,
                        Some(&alpha_dev),
                        Some(&beta_dev),
                        0.0,
                    )
                    .expect(&format!(
                        "launch gdn_prefill Qwen3-Next stream={stream_idx} step={step}"
                    ));
                }

                sync_with_timeout(
                    &stream,
                    &format!("stress_gdn_prefill_qwen3_next[{stream_idx}]"),
                );
            })
        })
        .collect();

    for (i, h) in handles.into_iter().enumerate() {
        h.join()
            .unwrap_or_else(|e| panic!("Qwen3-Next multi-stream thread {i} panicked: {e:?}"));
    }
}
