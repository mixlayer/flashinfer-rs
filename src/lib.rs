pub mod error;
mod ffi;
pub mod norm;
pub mod prefill;
pub mod runtime;

pub use error::FlashInferError;
pub use norm::{DType, GemmaRmsNormParams, Tensor1DDesc, Tensor2DDesc, gemma_rmsnorm};
#[cfg(feature = "cudarc")]
pub use norm::{gemma_rmsnorm_cudarc, gemma_rmsnorm_cudarc_with_options};
pub use prefill::{
    GdnPrefillSm90Params, Tensor1DI64Desc, Tensor1DU8Desc, Tensor2DF32Desc, Tensor3DDesc,
    Tensor4DF32Desc, gdn_prefill_sm90,
};
#[cfg(feature = "cudarc")]
pub use prefill::{
    gdn_prefill_sm90_cudarc, gdn_prefill_sm90_cudarc_with_options,
    gdn_prefill_sm90_cudarc_with_scale,
};
pub use runtime::{FlashInferRuntime, RuntimeConfig};
