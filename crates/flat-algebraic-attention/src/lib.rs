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
pub mod boolean_authenticated_frames;
pub mod boolean_benchmark_protocol;
pub mod boolean_coupled_learning;
pub mod boolean_edge_memory;
pub mod boolean_indexed_fronts;
pub mod boolean_neuron_activity;
pub mod boolean_neuron_graph;
pub mod boolean_operator_channels;
pub mod boolean_parallel_rotating_scan;
pub mod boolean_parallel_scan;
pub mod boolean_physical_protocol;
pub mod boolean_rotating_memory;
pub mod boolean_scan_lstm;
pub mod boolean_scan_training;
pub mod boolean_sequenced_publisher;
pub mod boolean_two_tap_memory;
pub mod boolean_watermarked_stream;
pub mod boolean_work_efficient_scan;
pub mod compact_preselection;
pub mod cooperation;
pub mod cycloid_cost;
pub mod cycloid_execution;
pub mod cycloid_scenario_execution;
pub mod cycloid_scenarios;
pub mod edge_pages;
pub mod evidence;
pub mod experiment;
pub mod f2;
pub mod holdout;
mod holdout_eq;
pub mod max_plus;
pub mod qk_signature;
pub mod qualification;
pub mod rae_dirty_words;
pub mod rae_hybrid_execution;
pub mod rae_hybrid_repair;
pub mod rae_mass_panel;
pub mod rae_norm_certificate;
pub mod rae_numeric_mass;
pub mod rae_page_norm_ledger;
pub mod rae_policy_audit;
pub mod readiness;
pub mod rising_edge;
pub mod softmax_mass;
pub mod survivor_floor;
pub mod survivor_set;
pub mod tdi22_torsor_projection;
pub mod torsor_octree;
pub mod zhegalkin;
