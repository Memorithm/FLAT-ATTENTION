#![forbid(unsafe_code)]

//! Research-only algebraic control primitives for FLAT-ATTENTION.
//!
//! This crate is intentionally outside `flat_attention::api::v1`. Its role is
//! to qualify algebraic decision mechanisms before any public API or runtime
//! routing promotion.
//!
//! Compact screening bootstrap and research gates are documented in
//! `docs/research/COMPACT_PRESELECTION_BOOTSTRAP.md` and the companion roadmap.

pub mod adaptive_repair;
pub mod compact_preselection;
pub mod cooperation;
pub mod cycloid_cost;
pub mod cycloid_execution;
pub mod edge_pages;
pub mod evidence;
pub mod experiment;
pub mod f2;
pub mod holdout;
mod holdout_eq;
pub mod max_plus;
pub mod qk_signature;
pub mod rae_policy_audit;
pub mod qualification;
pub mod readiness;
pub mod rising_edge;
pub mod softmax_mass;
pub mod survivor_floor;
pub mod survivor_set;
pub mod zhegalkin;
