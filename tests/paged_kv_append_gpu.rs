#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{
    DType, MhaQkvLayout, append_paged_kv_cache_cudarc, append_paged_mla_kv_cache_cudarc,
};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn encode_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

#[test]
fn gpu_smoke_launch_append_paged_kv_cache() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let nnz = 3_usize;
    let num_kv_heads = 4_usize;
    let head_dim = 128_usize;
    let page_size = 2_usize;
    let num_pages = 3_usize;

    let append_len = nnz * num_kv_heads * head_dim;
    let paged_len = num_pages * page_size * num_kv_heads * head_dim;

    let append_key_host: Vec<u16> = (0..append_len)
        .map(|i| encode_f16(((i % 17) as f32 - 8.0) * 0.03125))
        .collect();
    let append_value_host: Vec<u16> = (0..append_len)
        .map(|i| encode_f16(((i % 19) as f32 - 9.0) * 0.03125))
        .collect();

    let mut paged_k_host = vec![encode_f16(0.0); paged_len];
    let mut paged_v_host = vec![encode_f16(0.0); paged_len];
    for i in 0..paged_len {
        paged_k_host[i] = encode_f16(((i % 23) as f32 - 11.0) * 0.015625);
        paged_v_host[i] = encode_f16(((i % 29) as f32 - 14.0) * 0.015625);
    }

    let batch_indices_host = vec![0_i32, 0, 1];
    let positions_host = vec![0_i32, 3, 0];
    let kv_indices_host = vec![0_i32, 1, 2];
    let kv_indptr_host = vec![0_i32, 2, 3];
    let kv_last_page_len_host = vec![2_i32, 1];

    let append_key_dev = stream
        .clone_htod(&append_key_host)
        .expect("copy append_key");
    let append_value_dev = stream
        .clone_htod(&append_value_host)
        .expect("copy append_value");
    let batch_indices_dev = stream
        .clone_htod(&batch_indices_host)
        .expect("copy batch_indices");
    let positions_dev = stream.clone_htod(&positions_host).expect("copy positions");
    let mut paged_k_dev = stream
        .clone_htod(&paged_k_host)
        .expect("copy paged_k_cache");
    let mut paged_v_dev = stream
        .clone_htod(&paged_v_host)
        .expect("copy paged_v_cache");
    let kv_indices_dev = stream
        .clone_htod(&kv_indices_host)
        .expect("copy kv_indices");
    let kv_indptr_dev = stream.clone_htod(&kv_indptr_host).expect("copy kv_indptr");
    let kv_last_page_len_dev = stream
        .clone_htod(&kv_last_page_len_host)
        .expect("copy kv_last_page_len");

    append_paged_kv_cache_cudarc(
        stream.as_ref(),
        &append_key_dev,
        &append_value_dev,
        &batch_indices_dev,
        &positions_dev,
        &mut paged_k_dev,
        &mut paged_v_dev,
        &kv_indices_dev,
        &kv_indptr_dev,
        &kv_last_page_len_dev,
        num_kv_heads,
        head_dim,
        page_size,
        MhaQkvLayout::Nhd,
        DType::F16,
    )
    .expect("launch append_paged_kv_cache");

    stream.synchronize().expect("synchronize");
}

#[test]
fn gpu_smoke_launch_append_paged_mla_kv_cache() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let nnz = 3_usize;
    let page_size = 2_usize;
    let num_pages = 3_usize;

    let append_ckv_len = nnz * 512;
    let append_kpe_len = nnz * 64;
    let ckv_cache_len = num_pages * page_size * 512;
    let kpe_cache_len = num_pages * page_size * 64;

    let append_ckv_host: Vec<u16> = (0..append_ckv_len)
        .map(|i| encode_f16(((i % 17) as f32 - 8.0) * 0.03125))
        .collect();
    let append_kpe_host: Vec<u16> = (0..append_kpe_len)
        .map(|i| encode_f16(((i % 13) as f32 - 6.0) * 0.03125))
        .collect();

    let mut ckv_cache_host = vec![encode_f16(0.0); ckv_cache_len];
    let mut kpe_cache_host = vec![encode_f16(0.0); kpe_cache_len];
    for i in 0..ckv_cache_len {
        ckv_cache_host[i] = encode_f16(((i % 23) as f32 - 11.0) * 0.015625);
    }
    for i in 0..kpe_cache_len {
        kpe_cache_host[i] = encode_f16(((i % 29) as f32 - 14.0) * 0.015625);
    }

    let batch_indices_host = vec![0_i32, 0, 1];
    let positions_host = vec![0_i32, 3, 0];
    let kv_indices_host = vec![0_i32, 1, 2];
    let kv_indptr_host = vec![0_i32, 2, 3];
    let kv_last_page_len_host = vec![2_i32, 1];

    let append_ckv_dev = stream
        .clone_htod(&append_ckv_host)
        .expect("copy append_ckv");
    let append_kpe_dev = stream
        .clone_htod(&append_kpe_host)
        .expect("copy append_kpe");
    let batch_indices_dev = stream
        .clone_htod(&batch_indices_host)
        .expect("copy batch_indices");
    let positions_dev = stream.clone_htod(&positions_host).expect("copy positions");
    let mut ckv_cache_dev = stream.clone_htod(&ckv_cache_host).expect("copy ckv_cache");
    let mut kpe_cache_dev = stream.clone_htod(&kpe_cache_host).expect("copy kpe_cache");
    let kv_indices_dev = stream
        .clone_htod(&kv_indices_host)
        .expect("copy kv_indices");
    let kv_indptr_dev = stream.clone_htod(&kv_indptr_host).expect("copy kv_indptr");
    let kv_last_page_len_dev = stream
        .clone_htod(&kv_last_page_len_host)
        .expect("copy kv_last_page_len");

    append_paged_mla_kv_cache_cudarc(
        stream.as_ref(),
        &append_ckv_dev,
        &append_kpe_dev,
        &batch_indices_dev,
        &positions_dev,
        &mut ckv_cache_dev,
        &mut kpe_cache_dev,
        &kv_indices_dev,
        &kv_indptr_dev,
        &kv_last_page_len_dev,
        page_size,
        DType::F16,
    )
    .expect("launch append_paged_mla_kv_cache");

    stream.synchronize().expect("synchronize");
}
