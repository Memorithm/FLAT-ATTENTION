//! Research-only FDAL5b qualification of deterministic DA-LUC tier baselines.
//!
//! This crate does not implement another KV codec. Every assigned segment is
//! materialized exclusively through FDAL1 `DalucOraclePayload::encode` and
//! reconstructed through that payload's public decode surface.

#![forbid(unsafe_code)]

mod qualification;

pub use qualification::*;
