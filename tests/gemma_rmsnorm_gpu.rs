#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::norm::{DType, gemma_rmsnorm_cudarc};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn decode(bits: u16, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => half::f16::from_bits(bits).to_f32(),
        DType::BF16 => half::bf16::from_bits(bits).to_f32(),
    }
}

fn encode(value: f32, dtype: DType) -> u16 {
    match dtype {
        DType::F16 => half::f16::from_f32(value).to_bits(),
        DType::BF16 => half::bf16::from_f32(value).to_bits(),
    }
}

fn reference_gemma_rmsnorm(
    input: &[u16],
    weight: &[u16],
    rows: usize,
    cols: usize,
    eps: f64,
    dtype: DType,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let row_start = r * cols;
        let row = &input[row_start..row_start + cols];
        let sum_sq: f32 = row
            .iter()
            .map(|x| {
                let v = decode(*x, dtype);
                v * v
            })
            .sum();
        let inv = 1.0_f32 / ((sum_sq / cols as f32 + eps as f32).sqrt());
        for c in 0..cols {
            let x = decode(row[c], dtype);
            let w = decode(weight[c], dtype);
            out[row_start + c] = x * inv * (w + 1.0);
        }
    }
    out
}

fn run_case(dtype: DType, rows: usize, cols: usize, eps: f64) {
    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let input_host: Vec<u16> = (0..rows * cols)
        .map(|i| {
            let v = ((i % 29) as f32 - 14.0) * 0.0625;
            encode(v, dtype)
        })
        .collect();

    let weight_host: Vec<u16> = (0..cols)
        .map(|i| {
            let v = ((i % 17) as f32 - 8.0) * 0.015625;
            encode(v, dtype)
        })
        .collect();

    let input_dev = stream
        .clone_htod(&input_host)
        .expect("copy input to device");
    let weight_dev = stream
        .clone_htod(&weight_host)
        .expect("copy weight to device");
    let mut out_dev = stream
        .alloc_zeros::<u16>(rows * cols)
        .expect("allocate out buffer");

    gemma_rmsnorm_cudarc(
        stream.as_ref(),
        &input_dev,
        &weight_dev,
        &mut out_dev,
        rows,
        cols,
        dtype,
        eps,
    )
    .expect("launch gemma_rmsnorm");

    stream.synchronize().expect("synchronize stream");
    let out_host = stream.clone_dtoh(&out_dev).expect("copy output to host");

    let expected = reference_gemma_rmsnorm(&input_host, &weight_host, rows, cols, eps, dtype);

    let mut max_abs = 0.0_f32;
    for (idx, (got_bits, exp)) in out_host.iter().zip(expected.iter()).enumerate() {
        let got = decode(*got_bits, dtype);
        let abs_err = (got - exp).abs();
        let rel_bound = 0.02 * exp.abs();
        let tol = match dtype {
            DType::F16 => 0.03,
            DType::BF16 => 0.06,
        } + rel_bound;
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        assert!(
            abs_err <= tol,
            "idx={idx} got={got} exp={exp} abs_err={abs_err} tol={tol}"
        );
    }

    eprintln!("dtype={dtype:?} rows={rows} cols={cols} max_abs={max_abs}");
}

#[test]
fn gpu_correctness_fp16_and_bf16() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    run_case(DType::F16, 4, 3072, 1e-6);
    run_case(DType::F16, 3, 3333, 1e-6);
    run_case(DType::BF16, 4, 4096, 1e-6);
    run_case(DType::BF16, 3, 3072, 1e-6);
}

#[test]
fn gpu_async_launch_behavior() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let rows = 256;
    let cols = 4096;
    let eps = 1e-6;

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let input_host = vec![encode(0.125, DType::F16); rows * cols];
    let weight_host = vec![encode(0.0, DType::F16); cols];
    let input_dev = stream
        .clone_htod(&input_host)
        .expect("copy input to device");
    let weight_dev = stream
        .clone_htod(&weight_host)
        .expect("copy weight to device");
    let mut out_dev = stream
        .alloc_zeros::<u16>(rows * cols)
        .expect("allocate out buffer");

    for _ in 0..4 {
        gemma_rmsnorm_cudarc(
            stream.as_ref(),
            &input_dev,
            &weight_dev,
            &mut out_dev,
            rows,
            cols,
            DType::F16,
            eps,
        )
        .expect("launch gemma_rmsnorm");
    }

    let event = stream.record_event(None).expect("record event");
    assert!(
        !event.is_complete(),
        "work queue completed immediately; expected asynchronous launch behavior"
    );

    stream.synchronize().expect("synchronize stream");
}
