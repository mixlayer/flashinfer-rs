#![cfg(feature = "cudarc")]

use std::fs;
use std::mem::size_of;
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DeviceSlice, sys};
use flashinfer_rs::{
    TrtllmAllReduceBf16CudarcOptions, trtllm_allreduce_bf16_in_place_cudarc,
    trtllm_lamport_initialize_bf16_cudarc,
};

fn should_run_gpu_tests() -> bool {
    std::env::var("FLASHINFER_RS_RUN_GPU_TESTS").ok().as_deref() == Some("1")
}

#[test]
fn gpu_smoke_launch_trtllm_lamport_initialize_bf16() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }

    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.new_stream().expect("create CUDA stream");
    let mut lamport = stream
        .alloc_zeros::<u16>(1024)
        .expect("allocate BF16 Lamport buffer");

    trtllm_lamport_initialize_bf16_cudarc(stream.as_ref(), &mut lamport)
        .expect("launch BF16 Lamport initialization");
    let initialized = stream
        .clone_dtoh(&lamport)
        .expect("copy initialized Lamport buffer");

    assert!(
        initialized.iter().all(|value| *value == 0x8000),
        "BF16 Lamport initialization must fill the allocation with negative zero"
    );
}

#[test]
fn gpu_two_rank_trtllm_allreduce_bf16() {
    if !should_run_gpu_tests() {
        eprintln!("skipping GPU test (set FLASHINFER_RS_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    if CudaContext::device_count().expect("query CUDA device count") < 2 {
        eprintln!("skipping two-rank all-reduce test: fewer than two CUDA devices");
        return;
    }

    let coordination = tempfile::tempdir().expect("create IPC coordination directory");
    let executable = std::env::current_exe().expect("resolve integration-test executable");
    let mut children = (0..2)
        .map(|rank| {
            Command::new(&executable)
                .args([
                    "--exact",
                    "gpu_trtllm_allreduce_child",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("FLASHINFER_RS_ALLREDUCE_CHILD_RANK", rank.to_string())
                .env("FLASHINFER_RS_ALLREDUCE_COORD_DIR", coordination.path())
                .spawn()
                .expect("spawn all-reduce rank process")
        })
        .collect::<Vec<_>>();

    for (rank, child) in children.iter_mut().enumerate() {
        let status = child.wait().expect("wait for all-reduce rank process");
        assert!(status.success(), "all-reduce rank {rank} failed: {status}");
    }
}

#[test]
fn gpu_trtllm_allreduce_child() {
    let Ok(rank) = std::env::var("FLASHINFER_RS_ALLREDUCE_CHILD_RANK") else {
        return;
    };
    let rank = rank.parse::<usize>().expect("parse child rank");
    let coord_dir =
        std::env::var("FLASHINFER_RS_ALLREDUCE_COORD_DIR").expect("child coordination directory");
    run_allreduce_rank(rank, Path::new(&coord_dir));
}

fn run_allreduce_rank(rank: usize, coord_dir: &Path) {
    let world_size = 2_usize;
    let tokens = 1_usize;
    let hidden_size = 1024_usize;
    let message_elements = tokens * hidden_size;
    let lamport_comm_elements = world_size * message_elements;
    let context = CudaContext::new(rank).expect("create rank CUDA context");
    let stream = context.new_stream().expect("create rank CUDA stream");
    let comm = alloc_exportable::<u16>(&stream, lamport_comm_elements);
    let barrier = alloc_exportable::<i32>(&stream, world_size * 256);
    let mut lamport = alloc_exportable::<u16>(&stream, lamport_comm_elements * 3);

    trtllm_lamport_initialize_bf16_cudarc(stream.as_ref(), &mut lamport)
        .expect("initialize Lamport buffer");
    let local_handles = [
        ipc_handle(&comm, stream.as_ref()),
        ipc_handle(&barrier, stream.as_ref()),
        ipc_handle(&lamport, stream.as_ref()),
    ];
    write_handles_atomic(&coord_dir.join(format!("handles-{rank}")), &local_handles);
    let peer_rank = 1 - rank;
    let peer_handle_path = coord_dir.join(format!("handles-{peer_rank}"));
    wait_for_path(&peer_handle_path);
    let peer_handles = read_handles(&peer_handle_path);
    let remote = peer_handles.map(|handle| open_ipc(&context, handle));

    let local = [
        device_address(&comm, stream.as_ref()),
        device_address(&barrier, stream.as_ref()),
        device_address(&lamport, stream.as_ref()),
    ];
    let lamport_comm_bytes = i32::try_from(lamport_comm_elements * size_of::<u16>())
        .expect("Lamport communication bytes fit in i32");
    let metadata = stream
        .clone_htod(&[0_i32, 0, 0, lamport_comm_bytes, 0])
        .expect("copy metadata");
    let rank_order = |local_ptr, remote_ptr| {
        if rank == 0 {
            [local_ptr, remote_ptr]
        } else {
            [remote_ptr, local_ptr]
        }
    };
    let comm_ptrs = rank_order(local[0], remote[0]);
    let barrier_ptrs = rank_order(local[1], remote[1]);
    let lamport_ptrs = rank_order(local[2], remote[2]);
    let workspace_host = vec![
        comm_ptrs[0],
        comm_ptrs[1],
        barrier_ptrs[0],
        barrier_ptrs[1],
        lamport_ptrs[0],
        lamport_ptrs[1],
        device_address(&metadata, stream.as_ref()),
    ];
    let workspace = stream
        .clone_htod(&workspace_host)
        .expect("copy workspace table");
    let input_bits = if rank == 0 { 0x3f80_u16 } else { 0x4000_u16 };
    let mut input = stream
        .clone_htod(&vec![input_bits; message_elements])
        .expect("copy rank input");

    let ready_path = coord_dir.join(format!("ready-{rank}"));
    fs::write(&ready_path, []).expect("write ready marker");
    wait_for_path(&coord_dir.join(format!("ready-{peer_rank}")));
    trtllm_allreduce_bf16_in_place_cudarc(
        stream.as_ref(),
        &mut input,
        &workspace,
        tokens,
        hidden_size,
        world_size,
        rank,
        tokens,
        hidden_size,
        TrtllmAllReduceBf16CudarcOptions::default(),
    )
    .expect("launch all-reduce");
    let output = stream.clone_dtoh(&input).expect("copy all-reduce output");
    assert!(output.iter().all(|value| *value == 0x4040));

    let done_path = coord_dir.join(format!("done-{rank}"));
    fs::write(&done_path, []).expect("write done marker");
    wait_for_path(&coord_dir.join(format!("done-{peer_rank}")));
    close_ipc(&context, &remote);
    free_exportable(&context, comm);
    free_exportable(&context, barrier);
    free_exportable(&context, lamport);
}

fn device_address<T, S>(slice: &S, stream: &CudaStream) -> i64
where
    S: DeviceSlice<T> + DevicePtr<T>,
{
    let (ptr, _sync) = slice.device_ptr(stream);
    i64::try_from(ptr).expect("CUDA device pointer fits in i64")
}

fn ipc_handle<T, S>(slice: &S, stream: &CudaStream) -> sys::CUipcMemHandle
where
    S: DeviceSlice<T> + DevicePtr<T>,
{
    stream
        .context()
        .bind_to_thread()
        .expect("bind CUDA context");
    let (ptr, _sync) = slice.device_ptr(stream);
    let mut handle = sys::CUipcMemHandle_st { reserved: [0; 64] };
    // SAFETY: ptr is a live CUDA allocation and handle points to writable host storage.
    let result = unsafe { sys::cuIpcGetMemHandle(&mut handle, ptr) };
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "get IPC handle");
    handle
}

fn open_ipc(context: &CudaContext, handle: sys::CUipcMemHandle) -> i64 {
    context.bind_to_thread().expect("bind CUDA context");
    let mut ptr = 0_u64;
    // SAFETY: handle was exported from a live allocation in a different CUDA context.
    let result = unsafe {
        sys::cuIpcOpenMemHandle_v2(
            &mut ptr,
            handle,
            sys::CUipcMem_flags::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS as u32,
        )
    };
    assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "open IPC handle");
    i64::try_from(ptr).expect("CUDA IPC pointer fits in i64")
}

fn close_ipc(context: &CudaContext, pointers: &[i64]) {
    context.bind_to_thread().expect("bind CUDA context");
    for &ptr in pointers {
        // SAFETY: each pointer was opened exactly once in this context by open_ipc.
        let result = unsafe { sys::cuIpcCloseMemHandle(ptr as u64) };
        assert_eq!(result, sys::CUresult::CUDA_SUCCESS, "close IPC handle");
    }
}

fn alloc_exportable<T>(stream: &std::sync::Arc<CudaStream>, len: usize) -> CudaSlice<T> {
    stream
        .context()
        .bind_to_thread()
        .expect("bind CUDA context");
    let bytes = len
        .checked_mul(size_of::<T>())
        .expect("exportable allocation size overflow");
    let mut ptr = 0_u64;
    // SAFETY: ptr is writable host storage and bytes is the requested allocation size.
    let alloc_result = unsafe { sys::cuMemAlloc_v2(&mut ptr, bytes) };
    assert_eq!(
        alloc_result,
        sys::CUresult::CUDA_SUCCESS,
        "allocate exportable CUDA memory"
    );
    // SAFETY: ptr addresses bytes writable on this stream's current CUDA context.
    let memset_result = unsafe { sys::cuMemsetD8Async(ptr, 0, bytes, stream.cu_stream()) };
    assert_eq!(
        memset_result,
        sys::CUresult::CUDA_SUCCESS,
        "zero exportable CUDA memory"
    );
    // SAFETY: ptr is a live allocation with exactly len elements of storage.
    unsafe { stream.upgrade_device_ptr(ptr, len) }
}

fn free_exportable<T>(context: &CudaContext, slice: CudaSlice<T>) {
    let ptr = slice.leak();
    context.bind_to_thread().expect("bind CUDA context");
    // SAFETY: ptr came from cuMemAlloc_v2 and ownership was transferred out of CudaSlice::leak.
    let result = unsafe { sys::cuMemFree_v2(ptr) };
    assert_eq!(
        result,
        sys::CUresult::CUDA_SUCCESS,
        "free exportable CUDA memory"
    );
}

fn write_handles_atomic(path: &Path, handles: &[sys::CUipcMemHandle; 3]) {
    let mut bytes = Vec::with_capacity(3 * 64);
    for handle in handles {
        bytes.extend(handle.reserved.iter().map(|byte| *byte as u8));
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, bytes).expect("write IPC handles");
    fs::rename(temporary, path).expect("publish IPC handles");
}

fn read_handles(path: &Path) -> [sys::CUipcMemHandle; 3] {
    let bytes = fs::read(path).expect("read IPC handles");
    assert_eq!(bytes.len(), 3 * 64, "IPC handle payload length");
    let mut handles = [sys::CUipcMemHandle_st { reserved: [0; 64] }; 3];
    for (handle, chunk) in handles.iter_mut().zip(bytes.chunks_exact(64)) {
        for (destination, source) in handle.reserved.iter_mut().zip(chunk) {
            *destination = *source as i8;
        }
    }
    handles
}

fn wait_for_path(path: &Path) {
    let started = Instant::now();
    while !path.exists() {
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "timed out waiting for {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(10));
    }
}
