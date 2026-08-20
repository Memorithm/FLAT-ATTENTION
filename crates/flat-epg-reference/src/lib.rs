//! Correctness-first Elastic Positional Geometry (EPG) reference implementation.
//!
//! This crate is intentionally separate from FLAT's production kernel surface.
//! It depends only on FLAT's public contracts and provides an oracle against
//! which future fused WGSL/CUDA EPG kernels can be qualified.

#![forbid(unsafe_code)]

mod geometry;
mod oracle;

pub use geometry::{EpgEmbeddingConfig, EpgError, So4Geometry, EPG_CONTRACT_VERSION};
pub use oracle::forward_reference_grouped_epg;
