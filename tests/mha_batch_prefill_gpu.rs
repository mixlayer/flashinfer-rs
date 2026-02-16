#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{
    DType, MhaBatchPrefillCudarcOptions, MhaQkvLayout, mha_batch_prefill_cudarc_plan,
    mha_batch_prefill_cudarc_run,
};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

fn encode_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

#[test]
fn gpu_smoke_launch_batch_prefill_ragged() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create cuda context");
    let stream = ctx.new_stream().expect("create stream");

    let qo_indptr_host = vec![0_i32, 2, 3];
    let kv_indptr_host = vec![0_i32, 3, 5];

    let qo_total = *qo_indptr_host.last().expect("qo last") as usize;
    let kv_total = *kv_indptr_host.last().expect("kv last") as usize;

    let num_qo_heads = 8_usize;
    let num_kv_heads = 4_usize;
    let head_dim_qk = 128_usize;
    let head_dim_vo = 128_usize;

    let q_len = qo_total * num_qo_heads * head_dim_qk;
    let k_len = kv_total * num_kv_heads * head_dim_qk;
    let v_len = kv_total * num_kv_heads * head_dim_vo;
    let out_len = qo_total * num_qo_heads * head_dim_vo;

    let q_host: Vec<u16> = (0..q_len)
        .map(|i| encode_f16(((i % 17) as f32 - 8.0) * 0.03125))
        .collect();
    let k_host: Vec<u16> = (0..k_len)
        .map(|i| encode_f16(((i % 23) as f32 - 11.0) * 0.03125))
        .collect();
    let v_host: Vec<u16> = (0..v_len)
        .map(|i| encode_f16(((i % 29) as f32 - 14.0) * 0.03125))
        .collect();

    let q_dev = stream.clone_htod(&q_host).expect("copy q");
    let k_dev = stream.clone_htod(&k_host).expect("copy k");
    let v_dev = stream.clone_htod(&v_host).expect("copy v");
    let qo_indptr_dev = stream.clone_htod(&qo_indptr_host).expect("copy qo_indptr");
    let kv_indptr_dev = stream.clone_htod(&kv_indptr_host).expect("copy kv_indptr");

    let mut out_dev = stream.alloc_zeros::<u16>(out_len).expect("alloc out");
    let mut float_workspace = stream
        .alloc_zeros::<u8>(16 * 1024 * 1024)
        .expect("alloc float workspace");
    let mut int_workspace = stream
        .alloc_zeros::<u8>(8 * 1024 * 1024)
        .expect("alloc int workspace");
    let mut page_locked_int_workspace = vec![0_u8; 8 * 1024 * 1024];

    let options = MhaBatchPrefillCudarcOptions {
        causal: true,
        ..Default::default()
    };

    let plan = mha_batch_prefill_cudarc_plan(
        stream.as_ref(),
        &q_dev,
        &k_dev,
        &v_dev,
        &qo_indptr_dev,
        &kv_indptr_dev,
        &qo_indptr_host,
        &kv_indptr_host,
        &mut float_workspace,
        &mut int_workspace,
        &mut page_locked_int_workspace,
        &mut out_dev,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        MhaQkvLayout::Nhd,
        DType::F16,
        options,
    )
    .expect("plan batch prefill ragged");

    mha_batch_prefill_cudarc_run(
        stream.as_ref(),
        &plan,
        &q_dev,
        &k_dev,
        &v_dev,
        &qo_indptr_dev,
        &kv_indptr_dev,
        &qo_indptr_host,
        &kv_indptr_host,
        &mut float_workspace,
        &mut int_workspace,
        &mut page_locked_int_workspace,
        &mut out_dev,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        MhaQkvLayout::Nhd,
        DType::F16,
        options,
    )
    .expect("run batch prefill ragged");

    stream.synchronize().expect("synchronize");
}
