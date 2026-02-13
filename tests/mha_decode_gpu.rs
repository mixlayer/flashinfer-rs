#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{
    DType, MhaBatchDecodeCudarcOptions, MhaSingleDecodeCudarcOptions,
    mha_batch_decode_paged_cudarc, mha_single_decode_cudarc,
};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn encode_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

#[test]
fn gpu_smoke_launch_single_decode() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let kv_len = 5_usize;
    let num_qo_heads = 8_usize;
    let num_kv_heads = 4_usize;
    let head_dim = 128_usize;

    let q_len = num_qo_heads * head_dim;
    let k_len = kv_len * num_kv_heads * head_dim;
    let v_len = kv_len * num_kv_heads * head_dim;
    let out_len = num_qo_heads * head_dim;

    let q_host: Vec<u16> = (0..q_len)
        .map(|i| encode_f16(((i % 13) as f32 - 6.0) * 0.03125))
        .collect();
    let k_host: Vec<u16> = (0..k_len)
        .map(|i| encode_f16(((i % 17) as f32 - 8.0) * 0.03125))
        .collect();
    let v_host: Vec<u16> = (0..v_len)
        .map(|i| encode_f16(((i % 19) as f32 - 9.0) * 0.03125))
        .collect();

    let q_dev = stream.clone_htod(&q_host).expect("copy q");
    let k_dev = stream.clone_htod(&k_host).expect("copy k");
    let v_dev = stream.clone_htod(&v_host).expect("copy v");

    let mut tmp = stream
        .alloc_zeros::<u8>(32 * 1024 * 1024)
        .expect("alloc tmp");
    let mut out_dev = stream.alloc_zeros::<u16>(out_len).expect("alloc out");

    mha_single_decode_cudarc(
        stream.as_ref(),
        &q_dev,
        &k_dev,
        &v_dev,
        &mut tmp,
        &mut out_dev,
        kv_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        DType::F16,
        MhaSingleDecodeCudarcOptions::default(),
    )
    .expect("launch single decode");

    stream.synchronize().expect("synchronize");
}

#[test]
fn gpu_smoke_launch_batch_decode_paged() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let batch_size = 2_usize;
    let num_qo_heads = 8_usize;
    let num_kv_heads = 4_usize;
    let head_dim = 128_usize;
    let page_size = 2_usize;
    let num_pages = 3_usize;

    let q_len = batch_size * num_qo_heads * head_dim;
    let k_len = num_pages * page_size * num_kv_heads * head_dim;
    let v_len = num_pages * page_size * num_kv_heads * head_dim;
    let out_len = batch_size * num_qo_heads * head_dim;

    let q_host: Vec<u16> = (0..q_len)
        .map(|i| encode_f16(((i % 13) as f32 - 6.0) * 0.03125))
        .collect();
    let paged_k_host: Vec<u16> = (0..k_len)
        .map(|i| encode_f16(((i % 17) as f32 - 8.0) * 0.03125))
        .collect();
    let paged_v_host: Vec<u16> = (0..v_len)
        .map(|i| encode_f16(((i % 19) as f32 - 9.0) * 0.03125))
        .collect();

    let q_dev = stream.clone_htod(&q_host).expect("copy q");
    let paged_k_dev = stream.clone_htod(&paged_k_host).expect("copy paged_k");
    let paged_v_dev = stream.clone_htod(&paged_v_host).expect("copy paged_v");

    let paged_kv_indptr_host = vec![0_i32, 2, 3];
    let paged_kv_indices_host = vec![0_i32, 1, 2];
    let paged_kv_last_page_len_host = vec![2_i32, 1];

    let paged_kv_indptr_dev = stream
        .clone_htod(&paged_kv_indptr_host)
        .expect("copy paged_kv_indptr");
    let paged_kv_indices_dev = stream
        .clone_htod(&paged_kv_indices_host)
        .expect("copy paged_kv_indices");
    let paged_kv_last_page_len_dev = stream
        .clone_htod(&paged_kv_last_page_len_host)
        .expect("copy paged_kv_last_page_len");

    let mut float_workspace = stream
        .alloc_zeros::<u8>(16 * 1024 * 1024)
        .expect("alloc float workspace");
    let mut int_workspace = stream
        .alloc_zeros::<u8>(8 * 1024 * 1024)
        .expect("alloc int workspace");
    let mut page_locked_int_workspace = vec![0_u8; 8 * 1024 * 1024];
    let mut out_dev = stream.alloc_zeros::<u16>(out_len).expect("alloc out");

    mha_batch_decode_paged_cudarc(
        stream.as_ref(),
        &q_dev,
        &paged_k_dev,
        &paged_v_dev,
        &paged_kv_indptr_dev,
        &paged_kv_indices_dev,
        &paged_kv_last_page_len_dev,
        &paged_kv_indptr_host,
        &mut float_workspace,
        &mut int_workspace,
        &mut page_locked_int_workspace,
        &mut out_dev,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        DType::F16,
        MhaBatchDecodeCudarcOptions::default(),
    )
    .expect("launch batch decode paged");

    stream.synchronize().expect("synchronize");
}
