#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{
    NvFp4DType, nvfp4_gemm_cudarc, nvfp4_quantize_cudarc, nvfp4_scale_interleave_cudarc,
    swizzled_scale_len,
};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn scale_index(row: usize, col: usize, total_cols: usize) -> usize {
    let padded_cols = total_cols.div_ceil(4) * 4;
    let col_in_group = col % 4;
    let col_group = col / 4;
    let row_in_group0 = row % 32;
    let row_in_group1 = row % 128 / 32;
    let row_group = row / 128;
    col_in_group
        + col_group * (4 * 128)
        + row_in_group0 * 16
        + row_in_group1 * 4
        + row_group * (128 * padded_cols)
}

#[test]
fn gpu_nvfp4_quantize_interleave_and_gemm() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.new_stream().expect("create CUDA stream");
    let (major, minor) = ctx.compute_capability().expect("read compute capability");
    if major * 10 + minor < 120 {
        eprintln!("skipping NVFP4 smoke test on compute capability {major}.{minor}");
        return;
    }

    let raw_rows = 3;
    let raw_cols = 5;
    let raw_host: Vec<u8> = (0..raw_rows * raw_cols).map(|value| value as u8).collect();
    let raw_dev = stream.clone_htod(&raw_host).expect("copy raw scales");
    let interleaved_len = swizzled_scale_len(raw_rows as i64, raw_cols as i64)
        .expect("interleaved scale length") as usize;
    let mut interleaved_dev = stream
        .alloc_zeros::<u8>(interleaved_len)
        .expect("allocate interleaved scales");
    nvfp4_scale_interleave_cudarc(
        stream.as_ref(),
        &raw_dev,
        &mut interleaved_dev,
        raw_rows,
        raw_cols,
    )
    .expect("launch scale interleave");

    let m = 2;
    let n = 128;
    let k = 128;
    let input_host: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 % 29) as f32 - 14.0) / 14.0)
        .collect();
    let weight_host: Vec<f32> = (0..n * k)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) / 15.0)
        .collect();
    let input_bf16: Vec<u16> = input_host
        .iter()
        .map(|value| half::bf16::from_f32(*value).to_bits())
        .collect();
    let weight_bf16: Vec<u16> = weight_host
        .iter()
        .map(|value| half::bf16::from_f32(*value).to_bits())
        .collect();
    let input_dev = stream.clone_htod(&input_bf16).expect("copy input");
    let weight_dev = stream.clone_htod(&weight_bf16).expect("copy weight");
    let quant_scale_dev = stream
        .clone_htod(&[2688.0_f32])
        .expect("copy quantization scale");
    let dequant_scale_dev = stream
        .clone_htod(&[1.0_f32 / (2688.0 * 2688.0)])
        .expect("copy dequantization scale");

    let input_scale_len = swizzled_scale_len(m as i64, (k / 16) as i64).unwrap() as usize;
    let weight_scale_len = swizzled_scale_len(n as i64, (k / 16) as i64).unwrap() as usize;
    let mut packed_input = stream
        .alloc_zeros::<u8>(m * k / 2)
        .expect("allocate packed input");
    let mut packed_weight = stream
        .alloc_zeros::<u8>(n * k / 2)
        .expect("allocate packed weight");
    let mut input_scale = stream
        .alloc_zeros::<u8>(input_scale_len)
        .expect("allocate input scales");
    let mut weight_scale = stream
        .alloc_zeros::<u8>(weight_scale_len)
        .expect("allocate weight scales");
    nvfp4_quantize_cudarc(
        stream.as_ref(),
        &input_dev,
        &quant_scale_dev,
        &mut packed_input,
        &mut input_scale,
        m,
        k,
        NvFp4DType::BF16,
        false,
    )
    .expect("quantize input");
    nvfp4_quantize_cudarc(
        stream.as_ref(),
        &weight_dev,
        &quant_scale_dev,
        &mut packed_weight,
        &mut weight_scale,
        n,
        k,
        NvFp4DType::BF16,
        false,
    )
    .expect("quantize weight");

    let mut output = stream
        .alloc_zeros::<u16>(m * n)
        .expect("allocate GEMM output");
    let mut workspace = stream
        .alloc_zeros::<u8>(32 * 1024 * 1024)
        .expect("allocate GEMM workspace");
    nvfp4_gemm_cudarc(
        stream.as_ref(),
        &packed_input,
        &packed_weight,
        &input_scale,
        &weight_scale,
        &dequant_scale_dev,
        &mut output,
        &mut workspace,
        m,
        n,
        k,
        NvFp4DType::BF16,
        -1,
    )
    .expect("launch NVFP4 GEMM");

    stream.synchronize().expect("synchronize CUDA stream");
    let interleaved = stream
        .clone_dtoh(&interleaved_dev)
        .expect("copy interleaved scales");
    for row in 0..raw_rows {
        for col in 0..raw_cols {
            assert_eq!(
                interleaved[scale_index(row, col, raw_cols)],
                raw_host[row * raw_cols + col]
            );
        }
    }

    let output_host = stream.clone_dtoh(&output).expect("copy GEMM output");
    let mut max_error = 0.0_f32;
    for row in 0..m {
        for col in 0..n {
            let expected = (0..k)
                .map(|inner| input_host[row * k + inner] * weight_host[col * k + inner])
                .sum::<f32>();
            let actual = half::bf16::from_bits(output_host[row * n + col]).to_f32();
            assert!(actual.is_finite(), "non-finite result at [{row},{col}]");
            max_error = max_error.max((actual - expected).abs());
        }
    }
    assert!(max_error < 3.0, "NVFP4 GEMM max error {max_error}");
}
