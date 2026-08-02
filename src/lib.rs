pub mod error;
mod ffi;
pub mod fused_moe;
pub mod gdn_prefill;
pub mod mha_batch_prefill;
pub mod mha_batch_prefill_paged;
pub mod mha_decode;
pub mod mha_prefill;
pub mod mla_batch_paged;
pub mod norm;
pub mod paged_kv_append;
pub mod runtime;
pub mod sampling;
pub mod trtllm_allreduce;
pub mod trtllm_gen_moe;

pub use error::FlashInferError;
pub use fused_moe::{
    FusedMoeActivationType, FusedMoeBackend, FusedMoeDeepSeekFp8BlockScaleQuantParams,
    FusedMoeFp8ActScaleDesc, FusedMoeFp8PerTensorQuantParams, FusedMoeParams, FusedMoeQuantization,
    FusedMoeTensor0DF32Desc, FusedMoeTensor1DF32Desc, FusedMoeTensor2DDesc,
    FusedMoeTensor2DF32Desc, FusedMoeTensor2DI32Desc, FusedMoeTensor3DDesc,
    FusedMoeTensor3DF32Desc, fused_moe,
};
#[cfg(feature = "cudarc")]
pub use fused_moe::{
    FusedMoeCudarcOptions, fused_moe_cudarc, fused_moe_cudarc_deepseek_fp8_block_scale,
    fused_moe_cudarc_fp8_per_tensor,
};
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
pub use mha_batch_prefill::{
    MhaBatchPrefillCudarcOptions, mha_batch_prefill_cudarc_plan, mha_batch_prefill_cudarc_run,
};
pub use mha_batch_prefill::{
    MhaBatchPrefillParams, MhaBatchPrefillPlan, MhaBatchPrefillPlanParams, MhaHostTensor1DI32Desc,
    MhaHostTensor1DU8Desc, MhaTensor1DI32Desc, MhaTensor1DU16Desc, MhaTensor1DU32Desc,
    mha_batch_prefill_plan, mha_batch_prefill_run,
};
pub use mha_batch_prefill_paged::{
    MhaBatchPagedPrefillParams, MhaBatchPagedPrefillPlan, MhaBatchPagedPrefillPlanParams,
    MhaTensor4DDesc, mha_batch_prefill_paged_plan, mha_batch_prefill_paged_run,
};
#[cfg(feature = "cudarc")]
pub use mha_batch_prefill_paged::{
    mha_batch_prefill_paged_cudarc_plan, mha_batch_prefill_paged_cudarc_run,
};
#[cfg(feature = "cudarc")]
pub use mha_decode::{
    MhaBatchDecodeCudarcOptions, MhaSingleDecodeCudarcOptions, mha_batch_decode_paged_cudarc_plan,
    mha_batch_decode_paged_cudarc_run, mha_single_decode_cudarc,
};
pub use mha_decode::{
    MhaBatchPagedDecodeParams, MhaBatchPagedDecodePlan, MhaBatchPagedDecodePlanParams,
    MhaSingleDecodeParams, MhaTensor2DDesc, mha_batch_decode_paged_plan,
    mha_batch_decode_paged_run, mha_single_decode,
};
pub use mha_prefill::{
    MhaMaskMode, MhaPosEncodingMode, MhaQkvLayout, MhaSinglePrefillParams, MhaTensor1DF32Desc,
    MhaTensor1DU8Desc, MhaTensor2DF32Desc, MhaTensor3DDesc, mha_single_prefill,
};
#[cfg(feature = "cudarc")]
pub use mha_prefill::{MhaSinglePrefillCudarcOptions, mha_single_prefill_cudarc};
pub use mla_batch_paged::{
    MlaBackend, MlaBatchPagedAttentionParams, MlaBatchPagedAttentionPlan,
    MlaBatchPagedAttentionPlanParams, MlaTensor3DDesc, mla_batch_paged_plan, mla_batch_paged_run,
};
pub use norm::{
    DType, FusedQkRmsNormParams, GemmaFusedAddRmsNormParams, GemmaRmsNormParams, RmsNormParams,
    Tensor1DDesc, Tensor2DDesc, Tensor3DDesc as NormTensor3DDesc, fused_qk_rmsnorm,
    gemma_fused_add_rmsnorm, gemma_rmsnorm, qk_rmsnorm, rmsnorm,
};
#[cfg(feature = "cudarc")]
pub use norm::{
    fused_qk_rmsnorm_cudarc, fused_qk_rmsnorm_cudarc_with_options, gemma_rmsnorm_cudarc,
    gemma_rmsnorm_cudarc_with_options, qk_rmsnorm_cudarc, rmsnorm_cudarc,
    rmsnorm_cudarc_with_options,
};
pub use paged_kv_append::{
    PagedKvAppendParams, PagedMlaKvAppendParams, PagedMlaTensor2DDesc, append_paged_kv_cache,
    append_paged_mla_kv_cache,
};
#[cfg(feature = "cudarc")]
pub use paged_kv_append::{append_paged_kv_cache_cudarc, append_paged_mla_kv_cache_cudarc};
pub use runtime::{FlashInferRuntime, RuntimeConfig};
pub use sampling::{
    MinPSamplingParams, SAMPLING_WORKSPACE_BYTES, SamplingFromLogitsParams,
    SamplingFromProbsParams, SamplingParams, SamplingRandomParams, SamplingSoftmaxParams,
    SamplingTensor1DF32Desc, SamplingTensor1DI32Desc, SamplingTensor1DU64Desc,
    SamplingTensor2DF32Desc, SamplingWorkspaceDesc, TopKMaskLogitsParams, TopKRenormParams,
    TopKSamplingParams, TopKTopPSamplingParams, TopPRenormParams, TopPSamplingParams,
    min_p_sampling_from_probs, sampling_from_logits, sampling_from_probs, sampling_softmax,
    top_k_mask_logits, top_k_renorm_probs, top_k_sampling_from_probs,
    top_k_top_p_sampling_from_probs, top_p_renorm_probs, top_p_sampling_from_probs,
};
#[cfg(feature = "cudarc")]
pub use sampling::{
    SamplingCudarcRandom, min_p_sampling_from_probs_cudarc, sampling_from_logits_cudarc,
    sampling_from_probs_cudarc, sampling_softmax_cudarc, top_k_mask_logits_cudarc,
    top_k_renorm_probs_cudarc, top_k_sampling_from_probs_cudarc,
    top_k_top_p_sampling_from_probs_cudarc, top_p_renorm_probs_cudarc,
    top_p_sampling_from_probs_cudarc,
};
#[cfg(feature = "cudarc")]
pub use trtllm_allreduce::{
    TrtllmAllReduceBf16CudarcOptions, trtllm_allreduce_bf16_in_place_cudarc,
    trtllm_allreduce_residual_rmsnorm_bf16_cudarc, trtllm_lamport_initialize_bf16_cudarc,
};
pub use trtllm_allreduce::{
    TrtllmAllReduceBf16Params, TrtllmAllReduceBf16TensorDesc, TrtllmAllReduceBf16VectorDesc,
    TrtllmAllReduceResidualRmsNormBf16Params, TrtllmAllReduceWorkspaceDesc,
    TrtllmLamportInitializeBf16Params, trtllm_allreduce_bf16_in_place,
    trtllm_allreduce_residual_rmsnorm_bf16, trtllm_lamport_initialize_bf16,
};
#[cfg(feature = "cudarc")]
pub use trtllm_gen_moe::{
    TrtllmGenFp8BlockScaleMoeSm100CudarcOptions, trtllm_gen_fp8_block_scale_moe_sm100_cudarc,
};
pub use trtllm_gen_moe::{
    TrtllmGenFp8BlockScaleMoeSm100Params, TrtllmGenMoeDType, TrtllmGenMoeTensor1DDesc,
    TrtllmGenMoeTensor2DDesc, TrtllmGenMoeTensor3DDesc, trtllm_gen_fp8_block_scale_moe_sm100,
};
