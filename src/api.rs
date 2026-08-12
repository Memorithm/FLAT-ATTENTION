//! Backend-neutral reusable API contract.
//!
//! `api::v1` is intentionally independent of WGPU. It describes logical
//! attention geometry, configuration and ownership without exposing backend
//! handles. Backend adapters may consume the resident form with their own buffer
//! type while preserving the same validated contract.

/// Explicit WGPU-facing reusable state that is intentionally outside the
/// backend-neutral `v1` namespace.
#[cfg(feature = "wgpu")]
pub mod wgpu {
    pub use crate::wgpu_forward_grouped::PreparedGroupedForward;
}

pub mod v1 {
    use core::fmt;

    use crate::{AsymmetricGroupedAttentionShape, FlatAttentionConfig};

    /// Version of the backend-neutral API namespace.
    pub const API_VERSION: u16 = 1;

    /// Backend-neutral native GQA/MQA geometry.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct AttentionShape {
        pub batch: usize,
        pub q_heads: usize,
        pub kv_heads: usize,
        pub query_len: usize,
        pub kv_len: usize,
        pub head_dim: usize,
        pub query_position_offset: usize,
    }

    impl AttentionShape {
        pub fn validate(self) -> Result<(), ApiError> {
            if self.batch == 0
                || self.q_heads == 0
                || self.kv_heads == 0
                || self.query_len == 0
                || self.kv_len == 0
                || self.head_dim == 0
            {
                return Err(ApiError::ZeroDimension);
            }
            if !self.q_heads.is_multiple_of(self.kv_heads) {
                return Err(ApiError::InvalidHeadGrouping {
                    q_heads: self.q_heads,
                    kv_heads: self.kv_heads,
                });
            }
            self.q_elements()?;
            self.kv_elements()?;
            self.lse_elements()?;
            self.query_position_offset
                .checked_add(self.query_len - 1)
                .ok_or(ApiError::PositionOverflow)?;
            Ok(())
        }

        pub fn q_elements(self) -> Result<usize, ApiError> {
            checked_product(&[self.batch, self.q_heads, self.query_len, self.head_dim])
        }

        pub fn kv_elements(self) -> Result<usize, ApiError> {
            checked_product(&[self.batch, self.kv_heads, self.kv_len, self.head_dim])
        }

        pub fn lse_elements(self) -> Result<usize, ApiError> {
            checked_product(&[self.batch, self.q_heads, self.query_len])
        }

        pub fn group_size(self) -> Result<usize, ApiError> {
            self.validate()?;
            Ok(self.q_heads / self.kv_heads)
        }

        pub fn to_core_shape(self) -> Result<AsymmetricGroupedAttentionShape, ApiError> {
            self.validate()?;
            Ok(AsymmetricGroupedAttentionShape {
                batch: self.batch,
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
                query_len: self.query_len,
                kv_len: self.kv_len,
                head_dim: self.head_dim,
                query_position_offset: self.query_position_offset,
            })
        }
    }

    /// Backend-neutral execution configuration.
    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct AttentionConfig {
        pub causal: bool,
        pub softmax_scale: Option<f32>,
    }

    impl AttentionConfig {
        pub fn validate(self, head_dim: usize) -> Result<(), ApiError> {
            if head_dim == 0 {
                return Err(ApiError::ZeroDimension);
            }
            let scale = self
                .softmax_scale
                .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
            if !scale.is_finite() || scale <= 0.0 {
                return Err(ApiError::InvalidScale(scale));
            }
            Ok(())
        }

        pub fn to_core_config(self) -> FlatAttentionConfig {
            FlatAttentionConfig {
                causal: self.causal,
                softmax_scale: self.softmax_scale,
            }
        }
    }

    /// Borrowed host-memory request. No allocation or copy is implied.
    #[derive(Debug, Clone, Copy)]
    pub struct BorrowedAttentionRequest<'a> {
        pub shape: AttentionShape,
        pub config: AttentionConfig,
        pub q: &'a [f32],
        pub k: &'a [f32],
        pub v: &'a [f32],
    }

    impl BorrowedAttentionRequest<'_> {
        pub fn validate(self) -> Result<(), ApiError> {
            self.shape.validate()?;
            self.config.validate(self.shape.head_dim)?;
            validate_len("Q", self.q.len(), self.shape.q_elements()?)?;
            validate_len("K", self.k.len(), self.shape.kv_elements()?)?;
            validate_len("V", self.v.len(), self.shape.kv_elements()?)?;
            validate_finite("Q", self.q)?;
            validate_finite("K", self.k)?;
            validate_finite("V", self.v)?;
            Ok(())
        }

        pub fn to_owned(self) -> Result<OwnedAttentionRequest, ApiError> {
            self.validate()?;
            Ok(OwnedAttentionRequest {
                shape: self.shape,
                config: self.config,
                q: self.q.to_vec(),
                k: self.k.to_vec(),
                v: self.v.to_vec(),
            })
        }
    }

    /// Owned host-memory request for reusable standalone callers.
    #[derive(Debug, Clone, PartialEq)]
    pub struct OwnedAttentionRequest {
        pub shape: AttentionShape,
        pub config: AttentionConfig,
        pub q: Vec<f32>,
        pub k: Vec<f32>,
        pub v: Vec<f32>,
    }

    impl OwnedAttentionRequest {
        pub fn validate(&self) -> Result<(), ApiError> {
            self.as_borrowed().validate()
        }

        pub fn as_borrowed(&self) -> BorrowedAttentionRequest<'_> {
            BorrowedAttentionRequest {
                shape: self.shape,
                config: self.config,
                q: &self.q,
                k: &self.k,
                v: &self.v,
            }
        }
    }

    /// Backend-neutral resident request. `B` is owned by the embedding runtime;
    /// FLAT does not prescribe a device-buffer implementation at this layer.
    #[derive(Debug, Clone, Copy)]
    pub struct ResidentAttentionRequest<'a, B> {
        pub shape: AttentionShape,
        pub config: AttentionConfig,
        pub q: &'a B,
        pub k: &'a B,
        pub v: &'a B,
    }

    impl<B> ResidentAttentionRequest<'_, B> {
        /// Validate all backend-independent invariants. Buffer byte-size and
        /// ownership checks remain the responsibility of the concrete adapter.
        pub fn validate_contract(&self) -> Result<(), ApiError> {
            self.shape.validate()?;
            self.config.validate(self.shape.head_dim)
        }
    }

    /// Explicit reusable-API errors. No backend fallback is implied by an error.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ApiError {
        ZeroDimension,
        ShapeOverflow,
        PositionOverflow,
        InvalidHeadGrouping {
            q_heads: usize,
            kv_heads: usize,
        },
        InvalidScale(f32),
        LengthMismatch {
            tensor: &'static str,
            actual: usize,
            expected: usize,
        },
        NonFiniteInput {
            tensor: &'static str,
            index: usize,
        },
    }

    impl fmt::Display for ApiError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::ZeroDimension => write!(f, "attention dimensions must be non-zero"),
                Self::ShapeOverflow => write!(f, "attention shape overflows usize"),
                Self::PositionOverflow => write!(f, "query position range overflows usize"),
                Self::InvalidHeadGrouping { q_heads, kv_heads } => write!(
                    f,
                    "q_heads ({q_heads}) must be exactly divisible by kv_heads ({kv_heads})"
                ),
                Self::InvalidScale(scale) => {
                    write!(f, "softmax scale must be finite and positive, got {scale}")
                }
                Self::LengthMismatch {
                    tensor,
                    actual,
                    expected,
                } => write!(
                    f,
                    "tensor {tensor} contains {actual} elements, expected {expected}"
                ),
                Self::NonFiniteInput { tensor, index } => write!(
                    f,
                    "tensor {tensor} contains a non-finite value at index {index}"
                ),
            }
        }
    }

    impl std::error::Error for ApiError {}

    fn checked_product(values: &[usize]) -> Result<usize, ApiError> {
        values
            .iter()
            .try_fold(1usize, |acc, &value| acc.checked_mul(value))
            .ok_or(ApiError::ShapeOverflow)
    }

    fn validate_len(tensor: &'static str, actual: usize, expected: usize) -> Result<(), ApiError> {
        if actual != expected {
            return Err(ApiError::LengthMismatch {
                tensor,
                actual,
                expected,
            });
        }
        Ok(())
    }

    fn validate_finite(tensor: &'static str, values: &[f32]) -> Result<(), ApiError> {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(ApiError::NonFiniteInput { tensor, index });
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn shape() -> AttentionShape {
            AttentionShape {
                batch: 1,
                q_heads: 4,
                kv_heads: 2,
                query_len: 3,
                kv_len: 5,
                head_dim: 8,
                query_position_offset: 2,
            }
        }

        #[test]
        fn validates_native_gqa_lengths_and_conversion() {
            let shape = shape();
            assert_eq!(shape.q_elements().unwrap(), 96);
            assert_eq!(shape.kv_elements().unwrap(), 80);
            assert_eq!(shape.lse_elements().unwrap(), 12);
            assert_eq!(shape.group_size().unwrap(), 2);
            let core = shape.to_core_shape().unwrap();
            assert_eq!(core.query_len, 3);
            assert_eq!(core.kv_len, 5);
            assert_eq!(core.query_position_offset, 2);
        }

        #[test]
        fn borrowed_and_owned_requests_preserve_contract() {
            let shape = shape();
            let q = vec![0.25; shape.q_elements().unwrap()];
            let k = vec![0.5; shape.kv_elements().unwrap()];
            let v = vec![0.75; shape.kv_elements().unwrap()];
            let borrowed = BorrowedAttentionRequest {
                shape,
                config: AttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                q: &q,
                k: &k,
                v: &v,
            };
            borrowed.validate().unwrap();
            let owned = borrowed.to_owned().unwrap();
            owned.validate().unwrap();
            assert_eq!(owned.q, q);
            assert_eq!(owned.k, k);
            assert_eq!(owned.v, v);
        }

        #[test]
        fn resident_request_validates_without_touching_backend_handle() {
            let q = 1u8;
            let k = 2u8;
            let v = 3u8;
            let request = ResidentAttentionRequest {
                shape: shape(),
                config: AttentionConfig::default(),
                q: &q,
                k: &k,
                v: &v,
            };
            request.validate_contract().unwrap();
            assert_eq!((*request.q, *request.k, *request.v), (1, 2, 3));
        }

        #[test]
        fn rejects_invalid_grouping_and_non_finite_input() {
            let mut invalid = shape();
            invalid.q_heads = 3;
            assert_eq!(
                invalid.validate(),
                Err(ApiError::InvalidHeadGrouping {
                    q_heads: 3,
                    kv_heads: 2,
                })
            );

            let shape = shape();
            let mut q = vec![0.0; shape.q_elements().unwrap()];
            q[7] = f32::NAN;
            let k = vec![0.0; shape.kv_elements().unwrap()];
            let v = k.clone();
            let request = BorrowedAttentionRequest {
                shape,
                config: AttentionConfig::default(),
                q: &q,
                k: &k,
                v: &v,
            };
            assert_eq!(
                request.validate(),
                Err(ApiError::NonFiniteInput {
                    tensor: "Q",
                    index: 7,
                })
            );
        }
    }
}
