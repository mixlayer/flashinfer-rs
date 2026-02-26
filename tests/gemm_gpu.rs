#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{
    DType, TgvGemmCudarcOptions, TrtllmGemmTacticsQuery, TrtllmInputDType,
    TrtllmLowLatencyGemmTacticsQuery, TrtllmOutputDType, tgv_gemm_cudarc,
    tgv_gemm_cudarc_with_bias, trtllm_gemm_tactics, trtllm_low_latency_gemm_tactics,
    trtllm_low_latency_workspace_size_in_bytes,
};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn encode_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

fn encode_bf16(value: f32) -> u16 {
    half::bf16::from_f32(value).to_bits()
}

fn tgv_supported(major: i32, minor: i32) -> bool {
    matches!(major * 10 + minor, 100 | 103)
}

fn trtllm_gemm_supported(major: i32, minor: i32) -> bool {
    matches!(major * 10 + minor, 100 | 103 | 110 | 120 | 121)
}

fn trtllm_low_latency_supported(major: i32, minor: i32) -> bool {
    matches!(major * 10 + minor, 120 | 121)
}

fn run_tgv_smoke(dtype: DType) {
    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if !tgv_supported(major, minor) {
        eprintln!(
            "skipping TGV GEMM smoke test on unsupported compute capability {}.{}",
            major, minor
        );
        return;
    }

    let m = 16_usize;
    let k = 32_usize;
    let n = 24_usize;

    let encode = match dtype {
        DType::F16 => encode_f16,
        DType::BF16 => encode_bf16,
    };

    let a_host: Vec<u16> = (0..m * k)
        .map(|i| encode(((i % 23) as f32 - 11.0) * 0.03125))
        .collect();
    // B is interpreted as shape [k, n] in column-major layout.
    let b_host: Vec<u16> = (0..k * n)
        .map(|i| encode(((i % 29) as f32 - 14.0) * 0.015625))
        .collect();

    let a_dev = stream.clone_htod(&a_host).expect("copy a");
    let b_dev = stream.clone_htod(&b_host).expect("copy b");
    let mut out_dev = stream.alloc_zeros::<u16>(m * n).expect("alloc out");

    tgv_gemm_cudarc(
        stream.as_ref(),
        &a_dev,
        &b_dev,
        &mut out_dev,
        m,
        k,
        n,
        dtype,
        TgvGemmCudarcOptions::default(),
    )
    .expect("launch tgv_gemm_cudarc");

    let bias_host: Vec<u16> = (0..n)
        .map(|i| encode(((i % 17) as f32 - 8.0) * 0.03125))
        .collect();
    let bias_dev = stream.clone_htod(&bias_host).expect("copy bias");

    tgv_gemm_cudarc_with_bias(
        stream.as_ref(),
        &a_dev,
        &b_dev,
        &bias_dev,
        &mut out_dev,
        m,
        k,
        n,
        dtype,
        TgvGemmCudarcOptions {
            tactic: -1,
            enable_pdl: false,
        },
    )
    .expect("launch tgv_gemm_cudarc_with_bias");

    stream.synchronize().expect("synchronize");
}

#[test]
fn gpu_smoke_launch_tgv_gemm_fp16_and_bf16() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    run_tgv_smoke(DType::F16);
    run_tgv_smoke(DType::BF16);
}

#[test]
fn gpu_smoke_trtllm_gemm_tactics_query() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if !trtllm_gemm_supported(major, minor) {
        eprintln!(
            "skipping TRTLLM GEMM tactics smoke test on unsupported compute capability {}.{}",
            major, minor
        );
        return;
    }

    let query = TrtllmGemmTacticsQuery {
        m: 16,
        n: 256,
        k: 1024,
        input_dtype: TrtllmInputDType::E4m3,
        output_dtype: TrtllmOutputDType::Bf16,
        use_8x4_sf_layout: false,
    };
    let _tactics = trtllm_gemm_tactics(&query).expect("query trtllm gemm tactics");
}

#[test]
fn gpu_smoke_trtllm_low_latency_tactics_and_workspace() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if !trtllm_low_latency_supported(major, minor) {
        eprintln!(
            "skipping TRTLLM low-latency GEMM smoke test on unsupported compute capability {}.{}",
            major, minor
        );
        return;
    }

    let query = TrtllmLowLatencyGemmTacticsQuery {
        m: 16,
        n: 2560,
        k: 32768,
        input_dtype: TrtllmInputDType::E4m3,
        output_dtype: TrtllmOutputDType::Bf16,
    };

    let tactics =
        trtllm_low_latency_gemm_tactics(&query).expect("query trtllm low latency gemm tactics");
    if tactics.is_empty() {
        eprintln!("skipping workspace-size check because no low-latency tactics were returned");
        return;
    }

    let _workspace_size =
        trtllm_low_latency_workspace_size_in_bytes(query.m, query.n, query.k, tactics[0])
            .expect("query trtllm low latency workspace size");
}
