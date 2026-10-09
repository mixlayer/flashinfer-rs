//! Sweep FlashInfer SM120 NVFP4 GEMM tactics over Qwen3.8-27B MLP shapes.

use cudarc::driver::{CudaContext, sys};
use flashinfer_rs::{NvFp4DType, nvfp4_gemm_cudarc, nvfp4_gemm_tactic_num, swizzled_scale_len};

const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;
const WARMUP_RUNS: usize = 3;
const MEASUREMENT_ROUNDS: usize = 5;

fn repetitions() -> usize {
    std::env::var("FP4_BENCH_REPETITIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(20)
}

fn token_counts() -> Vec<usize> {
    std::env::var("FP4_BENCH_M")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::parse)
                .collect::<Result<Vec<_>, _>>()
                .expect("FP4_BENCH_M must be a comma-separated list of positive integers")
        })
        .filter(|values| !values.is_empty() && values.iter().all(|value| *value > 0))
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 64, 121, 128, 256])
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let (major, minor) = context.compute_capability()?;
    if major * 10 + minor < 120 {
        return Err(format!("NVFP4 tactics require SM120+, found {major}.{minor}").into());
    }
    let tactic_count = nvfp4_gemm_tactic_num()?;
    let repetitions = repetitions();
    let token_counts = token_counts();
    println!(
        "# device=sm{major}{minor},tactics={tactic_count},repetitions={repetitions},rounds={MEASUREMENT_ROUNDS}"
    );
    println!(
        "shape,m,best_tactic,default_us,candidate_us,paired_delta_us,paired_speedup,second_tactic,second_delta_us,second_speedup"
    );

    for (shape, n, k) in [("gate_up", 17_408, 5_120), ("down", 5_120, 17_408)] {
        let packed_k = k / 2;
        let weight = stream.clone_htod(&vec![0x11_u8; n * packed_k])?;
        let weight_scale_len = swizzled_scale_len(n as i64, (k / 16) as i64)? as usize;
        let weight_scale = stream.clone_htod(&vec![0x38_u8; weight_scale_len])?;
        let global_scale = stream.clone_htod(&[1.0_f32 / 256.0])?;
        let mut workspace = stream.alloc_zeros::<u8>(WORKSPACE_BYTES)?;

        for &m in &token_counts {
            let input = stream.clone_htod(&vec![0x11_u8; m * packed_k])?;
            let input_scale_len = swizzled_scale_len(m as i64, (k / 16) as i64)? as usize;
            let input_scale = stream.clone_htod(&vec![0x38_u8; input_scale_len])?;
            let mut output = stream.alloc_zeros::<u16>(m * n)?;

            let mut measure_batch =
                |tactic: i64, count: usize| -> Result<f64, Box<dyn std::error::Error>> {
                    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    for _ in 0..count {
                        nvfp4_gemm_cudarc(
                            stream.as_ref(),
                            &input,
                            &weight,
                            &input_scale,
                            &weight_scale,
                            &global_scale,
                            &mut output,
                            &mut workspace,
                            m,
                            n,
                            k,
                            NvFp4DType::BF16,
                            tactic,
                        )?;
                    }
                    let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    Ok(f64::from(start.elapsed_ms(&end)?) * 1_000.0 / count as f64)
                };

            let mut measured = Vec::with_capacity(tactic_count as usize);
            for tactic in 0..tactic_count {
                let result = (|| -> Result<_, Box<dyn std::error::Error>> {
                    measure_batch(-1, WARMUP_RUNS)?;
                    measure_batch(tactic, WARMUP_RUNS)?;
                    stream.synchronize()?;
                    let mut defaults = Vec::with_capacity(MEASUREMENT_ROUNDS);
                    let mut candidates = Vec::with_capacity(MEASUREMENT_ROUNDS);
                    let mut deltas = Vec::with_capacity(MEASUREMENT_ROUNDS);
                    let mut speedups = Vec::with_capacity(MEASUREMENT_ROUNDS);
                    for round in 0..MEASUREMENT_ROUNDS {
                        let (default_us, candidate_us) = if round % 2 == 0 {
                            (
                                measure_batch(-1, repetitions)?,
                                measure_batch(tactic, repetitions)?,
                            )
                        } else {
                            let candidate = measure_batch(tactic, repetitions)?;
                            (measure_batch(-1, repetitions)?, candidate)
                        };
                        defaults.push(default_us);
                        candidates.push(candidate_us);
                        deltas.push(default_us - candidate_us);
                        speedups.push(default_us / candidate_us);
                    }
                    Ok((
                        tactic,
                        median(&mut defaults),
                        median(&mut candidates),
                        median(&mut deltas),
                        median(&mut speedups),
                    ))
                })();
                match result {
                    Ok(result) => measured.push(result),
                    Err(error) => eprintln!("{shape},m={m},tactic={tactic}: {error}"),
                }
            }
            measured.sort_by(|left, right| right.3.total_cmp(&left.3));
            let best = measured[0];
            let second = measured[1];
            println!(
                "{shape},{m},{},{:.3},{:.3},{:.3},{:.4},{},{:.3},{:.4}",
                best.0, best.1, best.2, best.3, best.4, second.0, second.3, second.4
            );
        }
    }
    Ok(())
}
