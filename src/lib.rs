pub mod error;
mod ffi;
pub mod fused_moe;
pub mod gdn_prefill;
pub mod mha_batch_prefill;
pub mod mha_batch_prefill_paged;
pub mod mha_decode;
pub mod mha_prefill;
pub mod norm;
pub mod runtime;

pub use error::FlashInferError;
pub use fused_moe::{
    fused_moe, FusedMoeActivationType, FusedMoeBackend, FusedMoeParams, FusedMoeTensor2DDesc,
    FusedMoeTensor2DF32Desc, FusedMoeTensor2DI32Desc, FusedMoeTensor3DDesc,
};
#[cfg(feature = "cudarc")]
pub use fused_moe::{fused_moe_cudarc, FusedMoeCudarcOptions};
pub use gdn_prefill::{
    gdn_prefill_sm90, GdnPrefillSm90Params, Tensor1DI64Desc, Tensor1DU8Desc, Tensor2DF32Desc,
    Tensor3DDesc, Tensor4DF32Desc,
};
#[cfg(feature = "cudarc")]
pub use gdn_prefill::{
    gdn_prefill_sm90_cudarc, gdn_prefill_sm90_cudarc_with_options,
    gdn_prefill_sm90_cudarc_with_scale,
};
pub use mha_batch_prefill::{
    mha_batch_prefill, MhaBatchPrefillParams, MhaHostTensor1DI32Desc, MhaHostTensor1DU8Desc,
    MhaTensor1DI32Desc, MhaTensor1DU16Desc, MhaTensor1DU32Desc,
};
#[cfg(feature = "cudarc")]
pub use mha_batch_prefill::{mha_batch_prefill_cudarc, MhaBatchPrefillCudarcOptions};
#[cfg(feature = "cudarc")]
pub use mha_batch_prefill_paged::mha_batch_prefill_paged_cudarc;
pub use mha_batch_prefill_paged::{
    mha_batch_prefill_paged, MhaBatchPagedPrefillParams, MhaTensor4DDesc,
};
#[cfg(feature = "cudarc")]
pub use mha_decode::{
    mha_batch_decode_paged_cudarc_plan, mha_batch_decode_paged_cudarc_run,
    mha_single_decode_cudarc, MhaBatchDecodeCudarcOptions, MhaSingleDecodeCudarcOptions,
};
pub use mha_decode::{
    mha_batch_decode_paged_plan, mha_batch_decode_paged_run, mha_single_decode,
    MhaBatchPagedDecodeParams, MhaBatchPagedDecodePlan, MhaBatchPagedDecodePlanParams,
    MhaSingleDecodeParams, MhaTensor2DDesc,
};
pub use mha_prefill::{
    mha_single_prefill, MhaMaskMode, MhaPosEncodingMode, MhaQkvLayout, MhaSinglePrefillParams,
    MhaTensor1DF32Desc, MhaTensor1DU8Desc, MhaTensor2DF32Desc, MhaTensor3DDesc,
};
#[cfg(feature = "cudarc")]
pub use mha_prefill::{mha_single_prefill_cudarc, MhaSinglePrefillCudarcOptions};
pub use norm::{
    fused_qk_rmsnorm, gemma_rmsnorm, qk_rmsnorm, rmsnorm, DType, FusedQkRmsNormParams,
    GemmaRmsNormParams, RmsNormParams, Tensor1DDesc, Tensor2DDesc,
    Tensor3DDesc as NormTensor3DDesc,
};
#[cfg(feature = "cudarc")]
pub use norm::{
    fused_qk_rmsnorm_cudarc, fused_qk_rmsnorm_cudarc_with_options, gemma_rmsnorm_cudarc,
    gemma_rmsnorm_cudarc_with_options, qk_rmsnorm_cudarc, rmsnorm_cudarc,
    rmsnorm_cudarc_with_options,
};
pub use runtime::{FlashInferRuntime, RuntimeConfig};
