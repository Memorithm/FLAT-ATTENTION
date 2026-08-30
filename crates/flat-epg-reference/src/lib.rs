//! Correctness-first Elastic Positional Geometry (EPG) reference implementation.
//!
//! This crate is intentionally separate from FLAT's production kernel surface.
//! It adapts the runtime-neutral `epg-core` contract to FLAT's public attention
//! contracts and provides the oracle used to qualify future fused kernels.
//! Research observability is opt-in and does not participate in production routing.

#![forbid(unsafe_code)]

mod geometry;
mod oracle;
mod research_observability;
mod rotation;

pub use epg_core::{EpgGeometryDescriptor, EpgGeometryKind, EpgPositionDomain};
pub use geometry::{EpgEmbeddingConfig, EpgError, So4Geometry, EPG_CONTRACT_VERSION};
pub use oracle::forward_reference_grouped_epg;
pub use research_observability::{
    forward_reference_grouped_epg_observed, BoundedResearchTrace, ContributionObservation,
    InterventionDecision, NoIntervention, QueryDiagnostics, ResearchEvent,
    ResearchObservationContext, ResearchObserver, ResearchSemanticIdentity,
};