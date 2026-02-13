pub mod error;
mod ffi;
pub mod gdn_prefill;
pub mod mha_batch_prefill;
pub mod mha_batch_prefill_paged;
pub mod mha_decode;
pub mod mha_prefill;
pub mod norm;
pub mod runtime;

pub use error::FlashInferError;
pub use gdn_prefill::{
    GdnPrefillSm90Params, Tensor1DI64Desc, Tensor1DU8Desc, Tensor2DF32Desc, Tensor3DDesc,
    Tensor4DF32Desc, gdn_prefill_sm90,
};
#[cfg(feature = "cudarc")]
pub use gdn_prefill::{
    gdn_prefill_sm90_cudarc, gdn_prefill_sm90_cudarc_with_options,
    gdn_prefill_sm90_cudarc_with_scale,
};
#[cfg(feature = "cudarc")]
pub use mha_batch_prefill::{MhaBatchPrefillCudarcOptions, mha_batch_prefill_cudarc};
pub use mha_batch_prefill::{
    MhaBatchPrefillParams, MhaHostTensor1DI32Desc, MhaHostTensor1DU8Desc, MhaTensor1DI32Desc,
    MhaTensor1DU16Desc, MhaTensor1DU32Desc, mha_batch_prefill,
};
#[cfg(feature = "cudarc")]
pub use mha_batch_prefill_paged::mha_batch_prefill_paged_cudarc;
pub use mha_batch_prefill_paged::{
    MhaBatchPagedPrefillParams, MhaTensor4DDesc, mha_batch_prefill_paged,
};
#[cfg(feature = "cudarc")]
pub use mha_decode::{
    MhaBatchDecodeCudarcOptions, MhaSingleDecodeCudarcOptions, mha_batch_decode_paged_cudarc,
    mha_single_decode_cudarc,
};
pub use mha_decode::{
    MhaBatchPagedDecodeParams, MhaSingleDecodeParams, MhaTensor2DDesc, mha_batch_decode_paged,
    mha_single_decode,
};
pub use mha_prefill::{
    MhaMaskMode, MhaPosEncodingMode, MhaQkvLayout, MhaSinglePrefillParams, MhaTensor1DF32Desc,
    MhaTensor1DU8Desc, MhaTensor2DF32Desc, MhaTensor3DDesc, mha_single_prefill,
};
#[cfg(feature = "cudarc")]
pub use mha_prefill::{MhaSinglePrefillCudarcOptions, mha_single_prefill_cudarc};
pub use norm::{
    DType, FusedQkRmsNormParams, GemmaRmsNormParams, RmsNormParams, Tensor1DDesc, Tensor2DDesc,
    Tensor3DDesc as NormTensor3DDesc, fused_qk_rmsnorm, gemma_rmsnorm, qk_rmsnorm, rmsnorm,
};
#[cfg(feature = "cudarc")]
pub use norm::{
    fused_qk_rmsnorm_cudarc, fused_qk_rmsnorm_cudarc_with_options, gemma_rmsnorm_cudarc,
    gemma_rmsnorm_cudarc_with_options, qk_rmsnorm_cudarc, rmsnorm_cudarc,
    rmsnorm_cudarc_with_options,
};
pub use runtime::{FlashInferRuntime, RuntimeConfig};
