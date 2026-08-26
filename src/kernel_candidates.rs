//! Deterministic candidate generation for the dense Q4 family (roadmap M25).
//!
//! FLAT owns the candidate space: this module maps
//! `problem + device capabilities + selection policy` to an **ordered** list
//! of kernel candidates. Determinism requirements:
//!
//! - same inputs produce the same candidates in the same order;
//! - ordering is a total order over stable identities, never collection or
//!   adapter enumeration order;
//! - only variants with actual executable machinery on `main` are registered;
//!   retired/rejected historical realizations are structurally absent and can
//!   never re-enter selection;
//! - generation is bounded: the registry is finite, lifecycle policy filters
//!   it, capability prefiltering prunes it, and a hard per-call cap truncates
//!   it.
//!
//! Candidates carry static requirements, not measured performance. Measured
//! evidence belongs to the autotuning layer and must never be conflated with
//! these planning facts.

use crate::device_model::{RuntimeDeviceCapabilities, RuntimeKernelId};
use crate::kernel_ir::{AttentionProblem, KernelConfig, KernelFamily, KernelIrError, KernelModule};
use crate::kernel_wgsl::CodegenVersion;
use std::fmt;

/// Lifecycle of a registered kernel realization.
///
/// Only [`CandidateLifecycle::Qualified`] participates in normal routing and
/// only additionally [`CandidateLifecycle::Experimental`] when policy opts in.
/// Rejected and retired states exist so negative results stay representable
/// in evidence records; no generator enumerates them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CandidateLifecycle {
    /// Qualified for normal selection on capable devices.
    Qualified,
    /// Executable and qualified for correctness but excluded from default
    /// selection until physical evidence justifies promotion (e.g. M7).
    Experimental,
}

/// Stable identity of one candidate realization.
///
/// Derived from the family, the exact tuning configuration and the codegen
/// version through the repository's FNV-1a-64 discipline over a canonical
/// record. It is a cache-key/evidence identifier, never authentication, and
/// never a process-local counter that changes across runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CandidateId(u64);

impl CandidateId {
    fn build(family: KernelFamily, config: &KernelConfig) -> Self {
        let record = format!(
            "candidate/v1;cg=v{};family={};config={}",
            CodegenVersion::CURRENT,
            family.canonical(),
            config.canonical_record()
        );
        Self(crate::fingerprint::fnv1a64(record.as_bytes()))
    }

    /// Numeric fingerprint backing the identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for CandidateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// One selectable realization: family + tuning configuration + lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCandidate {
    /// Stable identity derived from the actual configuration.
    pub id: CandidateId,
    /// Kernel architecture.
    pub family: KernelFamily,
    /// Tuning configuration (the HOW).
    pub config: KernelConfig,
    /// Lifecycle state governing eligibility.
    pub lifecycle: CandidateLifecycle,
}

impl KernelCandidate {
    /// Static capability requirements imposed by this candidate's config.
    #[must_use]
    pub fn static_requirements(&self) -> Vec<crate::kernel_ir::CapabilityRequirement> {
        self.config.static_requirements()
    }

    /// Build the validated module realizing `problem` under this candidate.
    ///
    /// # Errors
    ///
    /// Returns [`KernelIrError`] when the problem has no executable path for
    /// this configuration.
    pub fn module_for(&self, problem: &AttentionProblem) -> Result<KernelModule, KernelIrError> {
        KernelModule::build(self.family, *problem, self.config)
    }

    /// Passive telemetry identity of the executable variant behind this
    /// candidate, when one exists today.
    #[must_use]
    pub fn runtime_kernel_id(&self) -> Option<RuntimeKernelId> {
        use crate::kernel_ir::{KvStaging, ScoreReduction, VectorWidth};
        if self.config.score_reduction == ScoreReduction::SubgroupAssisted {
            Some(RuntimeKernelId::Q4Subgroup)
        } else if self.config.kv_staging == KvStaging::DoubleBuffered {
            Some(RuntimeKernelId::Q4Vec4DoubleBuffered)
        } else if self.config.vector_width == VectorWidth::Vec4 {
            Some(RuntimeKernelId::Q4Vec4Portable)
        } else {
            Some(RuntimeKernelId::Q4Portable)
        }
    }
}

/// Policy controlling how the registry collapses into an ordered candidate
/// list for one generation call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionPolicy {
    /// Include experimental candidates after qualified ones. Default false:
    /// experimental realizations are opt-in until promoted by evidence.
    pub allow_experimental: bool,
    /// Hard cap on returned candidates. Default keeps generation cheap; full
    /// offline qualification may raise it explicitly.
    pub max_candidates: usize,
}

impl Default for SelectionPolicy {
    fn default() -> Self {
        Self {
            allow_experimental: false,
            max_candidates: 8,
        }
    }
}

/// Fixed registry of active dense-family realizations on `main`, in declared
/// order. Historical rejected/retired realizations (for example the ADA-A1
/// branch-specialized control) deliberately have no entry: they live outside
/// the router as frozen negative controls and can never be enumerated here.
const REGISTRY: &[(KernelConfig, CandidateLifecycle)] = &[
    (KernelConfig::PORTABLE_SCALAR, CandidateLifecycle::Qualified),
    (KernelConfig::PORTABLE_VEC4, CandidateLifecycle::Qualified),
    (
        KernelConfig::SUBGROUP_ASSISTED,
        CandidateLifecycle::Qualified,
    ),
    (
        KernelConfig::DOUBLE_BUFFERED_VEC4,
        CandidateLifecycle::Experimental,
    ),
];

/// Hard structural bound on registry size used by tests to prove boundedness.
pub const REGISTRY_LEN: usize = REGISTRY.len();

/// Generate the ordered candidate list for one problem on one device.
///
/// The pipeline per candidate is: lifecycle policy filter → problem/module
/// executability → static capability prefilter → stable total order → hard
/// truncation. Every step is deterministic.
#[must_use]
pub fn generate_candidates(
    problem: &AttentionProblem,
    capabilities: &RuntimeDeviceCapabilities,
    policy: &SelectionPolicy,
) -> Vec<KernelCandidate> {
    let mut candidates = Vec::new();
    for (config, lifecycle) in REGISTRY {
        if *lifecycle == CandidateLifecycle::Experimental && !policy.allow_experimental {
            continue;
        }
        let candidate = KernelCandidate {
            id: CandidateId::build(KernelFamily::DenseQ4Forward, config),
            family: KernelFamily::DenseQ4Forward,
            config: *config,
            lifecycle: *lifecycle,
        };
        // Problem executability: e.g. vec4 has no path for head_dim 80.
        let Ok(module) = candidate.module_for(problem) else {
            continue;
        };
        // Capability pruning before any pipeline could ever be created.
        if crate::kernel_prefilter::check_module(&module, capabilities).is_err() {
            continue;
        }
        candidates.push(candidate);
    }
    // Total deterministic order: lifecycle rank first (qualified before
    // experimental), then stable id.
    candidates.sort_by_key(|candidate| (candidate.lifecycle, candidate.id));
    candidates.truncate(policy.max_candidates.max(1));
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::VectorWidth;
    use crate::{AttentionShape, FlatAttentionConfig};

    fn problem(head_dim: u32) -> AttentionProblem {
        AttentionProblem::from_shape(
            &AttentionShape {
                batch: 2,
                heads: 4,
                seq_len: 129,
                head_dim: head_dim as usize,
            },
            FlatAttentionConfig {
                causal: true,
                softmax_scale: None,
            },
        )
        .unwrap()
    }

    fn caps(subgroup: bool) -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65535,
            max_workgroup_size_x: 64,
            max_workgroup_size_y: 1024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32768,
            max_binding_entries: 8,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_supported: subgroup,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f16_supported: true,
        }
    }

    #[test]
    fn same_inputs_produce_same_ordered_candidates() {
        let p = problem(64);
        let c = caps(true);
        let a = generate_candidates(&p, &c, &SelectionPolicy::default());
        let b = generate_candidates(&p, &c, &SelectionPolicy::default());
        assert_eq!(a, b);
        assert!(!a.is_empty());
        // Qualified-before-experimental rank, then ascending ids.
        assert!(a
            .windows(2)
            .all(|w| (w[0].lifecycle, w[0].id) <= (w[1].lifecycle, w[1].id)));
    }

    #[test]
    fn unsupported_subgroup_removes_the_subgroup_candidate() {
        let p = problem(64);
        let without = generate_candidates(&p, &caps(false), &SelectionPolicy::default());
        assert!(without.iter().all(
            |c| c.config.score_reduction != crate::kernel_ir::ScoreReduction::SubgroupAssisted
        ));
        let with = generate_candidates(&p, &caps(true), &SelectionPolicy::default());
        assert!(with.iter().any(
            |c| c.config.score_reduction == crate::kernel_ir::ScoreReduction::SubgroupAssisted
        ));
    }

    #[test]
    fn unaligned_head_dim_prunes_vector_candidates_only() {
        let p = problem(80);
        let candidates = generate_candidates(&p, &caps(true), &SelectionPolicy::default());
        // Vec4 machinery has no D80 path; scalar and subgroup (scalar-width)
        // realizations remain eligible.
        assert!(candidates
            .iter()
            .all(|c| c.config.vector_width == VectorWidth::Scalar));
        assert_eq!(
            candidates
                .iter()
                .map(|c| c.runtime_kernel_id())
                .collect::<Vec<_>>(),
            vec![
                Some(RuntimeKernelId::Q4Portable),
                Some(RuntimeKernelId::Q4Subgroup)
            ]
        );
    }

    #[test]
    fn experimental_candidates_are_opt_in() {
        let p = problem(128);
        let conservative = generate_candidates(&p, &caps(true), &SelectionPolicy::default());
        assert!(conservative
            .iter()
            .all(|c| c.lifecycle == CandidateLifecycle::Qualified));

        let opt_in = SelectionPolicy {
            allow_experimental: true,
            ..SelectionPolicy::default()
        };
        let expanded = generate_candidates(&p, &caps(true), &opt_in);
        assert!(expanded
            .iter()
            .any(|c| c.lifecycle == CandidateLifecycle::Experimental
                && c.config.kv_staging == crate::kernel_ir::KvStaging::DoubleBuffered));
    }

    #[test]
    fn candidate_counts_remain_bounded() {
        let p = problem(64);
        let generous = SelectionPolicy {
            allow_experimental: true,
            max_candidates: usize::MAX,
        };
        let all = generate_candidates(&p, &caps(true), &generous);
        assert!(all.len() <= REGISTRY_LEN);

        let capped = generate_candidates(
            &p,
            &caps(true),
            &SelectionPolicy {
                max_candidates: 2,
                ..SelectionPolicy::default()
            },
        );
        assert_eq!(capped.len(), 2);
        // Truncation preserves the deterministic prefix.
        assert_eq!(&capped[..], &all[..2]);
    }

    #[test]
    fn identities_are_stable_and_config_sensitive() {
        let a = CandidateId::build(KernelFamily::DenseQ4Forward, &KernelConfig::PORTABLE_SCALAR);
        let b = CandidateId::build(KernelFamily::DenseQ4Forward, &KernelConfig::PORTABLE_SCALAR);
        assert_eq!(a, b);
        let c = CandidateId::build(KernelFamily::DenseQ4Forward, &KernelConfig::PORTABLE_VEC4);
        assert_ne!(a, c);
        // Display form is fixed-width hex.
        assert_eq!(format!("{a}").len(), 16);
    }

    #[test]
    fn tiny_device_yields_empty_list_not_invalid_ones() {
        let p = problem(64);
        let mut minimal = caps(false);
        minimal.max_workgroup_size_x = 8;
        minimal.max_workgroup_storage_bytes = 512;
        let candidates = generate_candidates(&p, &minimal, &SelectionPolicy::default());
        assert!(candidates.is_empty());
    }

    #[test]
    fn every_candidate_maps_to_a_runtime_kernel_identity() {
        let p = problem(64);
        let all = generate_candidates(
            &p,
            &caps(true),
            &SelectionPolicy {
                allow_experimental: true,
                ..SelectionPolicy::default()
            },
        );
        assert!(all.iter().all(|c| c.runtime_kernel_id().is_some()));
    }
}
