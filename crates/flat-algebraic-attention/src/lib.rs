#![forbid(unsafe_code)]

//! Research-only algebraic control primitives for FLAT-ATTENTION.
//!
//! This crate is intentionally outside `flat_attention::api::v1`. Its role is
//! to qualify algebraic decision mechanisms before any public API or runtime
//! routing promotion.

pub mod cooperation;
pub mod evidence;
pub mod f2;
pub mod max_plus;
pub mod qualification;
pub mod survivor_set;
pub mod zhegalkin;
