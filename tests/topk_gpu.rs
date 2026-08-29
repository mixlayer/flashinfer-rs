#![cfg(feature = "cudarc")]

use cudarc::driver::CudaContext;
use flashinfer_rs::{TOPK_ROW_STATES_BYTES, TopKDType, TopKOptions, TopKTieBreak, top_k_cudarc};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn deterministic_topk_wheel_symbol_launch_smoke() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU smoke test; set FLASHINFER_RS_RUN_GPU_TESTS=1");
        return;
    }

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.new_stream().expect("create CUDA stream");
    let rows = 2;
    let vocab_size = 248_320;
    let top_k = 16;

    let mut host_input = vec![-100.0_f32; rows * vocab_size];
    for value in &mut host_input[..32] {
        *value = 1.0;
    }
    for rank in 0..top_k {
        host_input[vocab_size + 100 + rank] = rank as f32;
    }

    let input = stream.clone_htod(&host_input).expect("copy input");
    let mut output_indices = stream
        .alloc_zeros::<i32>(rows * top_k)
        .expect("allocate output indices");
    let mut output_values = stream
        .alloc_zeros::<f32>(rows * top_k)
        .expect("allocate output values");
    let mut row_states = stream
        .alloc_zeros::<u8>(TOPK_ROW_STATES_BYTES)
        .expect("allocate row states");
    let options = TopKOptions {
        sorted_output: true,
        deterministic: true,
        tie_break: TopKTieBreak::Small,
        dsa_graph_safe: false,
    };

    let mut reference = None;
    for _ in 0..3 {
        top_k_cudarc(
            stream.as_ref(),
            &input,
            &mut output_indices,
            &mut output_values,
            &mut row_states,
            rows,
            vocab_size,
            top_k,
            TopKDType::F32,
            options,
        )
        .expect("launch deterministic top-k");
        stream.synchronize().expect("synchronize top-k");
        let indices = stream
            .clone_dtoh(&output_indices)
            .expect("copy output indices");
        if let Some(reference) = &reference {
            assert_eq!(&indices, reference);
        } else {
            reference = Some(indices.clone());
        }

        let mut tied = indices[..top_k].to_vec();
        tied.sort_unstable();
        assert_eq!(tied, (0..top_k as i32).collect::<Vec<_>>());
        assert_eq!(
            &indices[top_k..],
            &(0..top_k)
                .rev()
                .map(|rank| 100 + rank as i32)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn deterministic_topk_is_batch_position_independent() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU smoke test; set FLASHINFER_RS_RUN_GPU_TESTS=1");
        return;
    }

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.new_stream().expect("create CUDA stream");
    let rows = 4;
    let vocab_size = 248_320;
    let top_k = 16;
    let mut host_row = vec![-100.0_f32; vocab_size];
    for rank in 0..top_k {
        host_row[29 + rank * 10_003] = rank as f32;
    }
    let host_input = host_row.repeat(rows);
    let input = stream.clone_htod(&host_input).expect("copy batched input");
    let single_input = stream.clone_htod(&host_row).expect("copy single input");
    let mut output_indices = stream
        .alloc_zeros::<i32>(rows * top_k)
        .expect("allocate batched indices");
    let mut output_values = stream
        .alloc_zeros::<f32>(rows * top_k)
        .expect("allocate batched values");
    let mut single_indices = stream
        .alloc_zeros::<i32>(top_k)
        .expect("allocate single indices");
    let mut single_values = stream
        .alloc_zeros::<f32>(top_k)
        .expect("allocate single values");
    let mut row_states = stream
        .alloc_zeros::<u8>(TOPK_ROW_STATES_BYTES)
        .expect("allocate row states");
    let options = TopKOptions {
        sorted_output: true,
        deterministic: true,
        tie_break: TopKTieBreak::None,
        dsa_graph_safe: false,
    };

    top_k_cudarc(
        stream.as_ref(),
        &input,
        &mut output_indices,
        &mut output_values,
        &mut row_states,
        rows,
        vocab_size,
        top_k,
        TopKDType::F32,
        options,
    )
    .expect("launch batched deterministic top-k");
    top_k_cudarc(
        stream.as_ref(),
        &single_input,
        &mut single_indices,
        &mut single_values,
        &mut row_states,
        1,
        vocab_size,
        top_k,
        TopKDType::F32,
        options,
    )
    .expect("launch single-row deterministic top-k");
    stream.synchronize().expect("synchronize top-k");

    let batched = stream
        .clone_dtoh(&output_indices)
        .expect("copy batched indices");
    let single = stream
        .clone_dtoh(&single_indices)
        .expect("copy single indices");
    for row in batched.chunks_exact(top_k) {
        assert_eq!(row, single);
    }
}
