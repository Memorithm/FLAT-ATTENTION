//! Prepared WGPU kernel routing for qualified FLAT-ATTENTION realizations.
//!
//! Every admitted candidate is compiled before it becomes selectable. Applying
//! a candidate therefore changes only the active prepared route; it does not
//! compile a pipeline, change attention semantics, or silently substitute a
//! fallback kernel. Schema v1 is deliberately host-I/O-only: each prepared
//! candidate owns its own WGPU context, so resident buffers and KV state are not
//! transferable between routes.

#![forbid(unsafe_code)]

use core::fmt;
use std::collections::BTreeSet;

use flat_attention::kernel_candidates::{CandidateId, CandidateLifecycle, KernelCandidate};
use flat_attention::kernel_ir::AttentionProblem;
use flat_attention::{
    AttentionShape, FlatAttentionConfig, FlatAttentionOutput, RuntimeDeviceFingerprint,
    RuntimeDispatchTelemetry, RuntimeKernelId, WgpuFlatAttention, WgpuFlatAttentionError,
};

/// Schema version for [`WgpuPreparedKernelTransitionRequirementsV1`].
pub const WGPU_PREPARED_KERNEL_TRANSITION_REQUIREMENTS_VERSION: u32 = 1;

/// Explicit transition boundary of the prepared host-I/O WGPU router.
///
/// `route_swap_live` means applying another *already prepared* route requires
/// no pipeline compilation. It does not imply that resident buffers or KV state
/// survive because schema v1 prepares a distinct WGPU context per candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuPreparedKernelTransitionRequirementsV1 {
    pub schema_version: u32,
    pub target_must_be_prepared_before_apply: bool,
    pub route_swap_live: bool,
    pub host_io_only: bool,
    pub resident_buffers_preserved: bool,
    pub kv_state_preserved: bool,
}

impl WgpuPreparedKernelTransitionRequirementsV1 {
    #[must_use]
    pub const fn current() -> Self {
        Self {
            schema_version: WGPU_PREPARED_KERNEL_TRANSITION_REQUIREMENTS_VERSION,
            target_must_be_prepared_before_apply: true,
            route_swap_live: true,
            host_io_only: true,
            resident_buffers_preserved: false,
            kv_state_preserved: false,
        }
    }

    /// Validate schema-v1 invariants. This prevents callers from re-labeling
    /// the current host-I/O route swap as resident-state-preserving.
    pub fn validate(&self) -> Result<(), WgpuPreparedKernelRouterError> {
        if self.schema_version != WGPU_PREPARED_KERNEL_TRANSITION_REQUIREMENTS_VERSION {
            return Err(WgpuPreparedKernelRouterError::UnsupportedTransitionSchema {
                actual: self.schema_version,
                expected: WGPU_PREPARED_KERNEL_TRANSITION_REQUIREMENTS_VERSION,
            });
        }
        if !self.target_must_be_prepared_before_apply
            || !self.route_swap_live
            || !self.host_io_only
            || self.resident_buffers_preserved
            || self.kv_state_preserved
        {
            return Err(WgpuPreparedKernelRouterError::InvalidTransitionRequirements);
        }
        Ok(())
    }
}

/// Fail-closed errors for prepared kernel routing.
#[derive(Debug)]
#[non_exhaustive]
pub enum WgpuPreparedKernelRouterError {
    EmptyCandidateSet,
    DuplicateCandidate { candidate_id: u64 },
    InitialCandidateMissing { candidate_id: u64 },
    CandidateMissing { candidate_id: u64 },
    CandidateNotQualified { candidate_id: u64 },
    CandidateProblem {
        candidate_id: u64,
        detail: String,
    },
    CandidateHasNoRuntimeKernel { candidate_id: u64 },
    CandidateSubstitution {
        candidate_id: u64,
        expected: RuntimeKernelId,
        actual: RuntimeKernelId,
    },
    DeviceMismatch {
        candidate_id: u64,
        expected: RuntimeDeviceFingerprint,
        actual: RuntimeDeviceFingerprint,
    },
    Backend {
        candidate_id: u64,
        source: WgpuFlatAttentionError,
    },
    UnsupportedTransitionSchema { actual: u32, expected: u32 },
    InvalidTransitionRequirements,
}

impl fmt::Display for WgpuPreparedKernelRouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet => f.write_str("prepared WGPU kernel router requires candidates"),
            Self::DuplicateCandidate { candidate_id } => {
                write!(f, "duplicate prepared candidate {candidate_id:016x}")
            }
            Self::InitialCandidateMissing { candidate_id } => write!(
                f,
                "initial candidate {candidate_id:016x} is not in the prepared set"
            ),
            Self::CandidateMissing { candidate_id } => {
                write!(f, "candidate {candidate_id:016x} is not prepared")
            }
            Self::CandidateNotQualified { candidate_id } => write!(
                f,
                "candidate {candidate_id:016x} is not qualified for normal live routing"
            ),
            Self::CandidateProblem {
                candidate_id,
                detail,
            } => write!(
                f,
                "candidate {candidate_id:016x} cannot execute the requested attention problem: {detail}"
            ),
            Self::CandidateHasNoRuntimeKernel { candidate_id } => write!(
                f,
                "candidate {candidate_id:016x} has no executable WGPU runtime identity"
            ),
            Self::CandidateSubstitution {
                candidate_id,
                expected,
                actual,
            } => write!(
                f,
                "candidate {candidate_id:016x} requested {expected:?} but runtime would execute {actual:?}"
            ),
            Self::DeviceMismatch {
                candidate_id,
                expected,
                actual,
            } => write!(
                f,
                "candidate {candidate_id:016x} prepared on a different WGPU device: expected `{}`, observed `{}`",
                expected.canonical_record(),
                actual.canonical_record()
            ),
            Self::Backend {
                candidate_id,
                source,
            } => write!(
                f,
                "WGPU preparation/execution failed for candidate {candidate_id:016x}: {source}"
            ),
            Self::UnsupportedTransitionSchema { actual, expected } => write!(
                f,
                "unsupported prepared WGPU transition schema {actual}; expected {expected}"
            ),
            Self::InvalidTransitionRequirements => f.write_str(
                "prepared WGPU transition requirements must stay live-prepared, host-I/O-only, and must not claim resident/KV preservation",
            ),
        }
    }
}

impl std::error::Error for WgpuPreparedKernelRouterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend { source, .. } => Some(source),
            _ => None,
        }
    }
}

struct PreparedKernelRoute {
    candidate: KernelCandidate,
    executor: WgpuFlatAttention,
}

/// A set of already-compiled qualified WGPU kernel routes with one active route.
///
/// Applying/restoring a candidate is a host-side route-state change. Physical
/// execution occurs when [`Self::forward`] dispatches through the active prepared
/// context. Candidate identity is checked against passive runtime telemetry before
/// every dispatch so shape-dependent fallback cannot impersonate the requested
/// realization.
pub struct WgpuPreparedKernelRouter {
    routes: Vec<PreparedKernelRoute>,
    active: usize,
    device: RuntimeDeviceFingerprint,
}

impl WgpuPreparedKernelRouter {
    /// Compile every qualified candidate before making the router available.
    ///
    /// Preparation fails closed if candidates are duplicated, unavailable, not
    /// qualified, or resolve to different physical adapter fingerprints.
    pub fn prepare(
        candidates: Vec<KernelCandidate>,
        initial_candidate: CandidateId,
    ) -> Result<Self, WgpuPreparedKernelRouterError> {
        if candidates.is_empty() {
            return Err(WgpuPreparedKernelRouterError::EmptyCandidateSet);
        }

        let mut seen = BTreeSet::new();
        let mut routes = Vec::with_capacity(candidates.len());
        let mut device: Option<RuntimeDeviceFingerprint> = None;

        for candidate in candidates {
            let candidate_id = candidate.id.get();
            if candidate.lifecycle != CandidateLifecycle::Qualified {
                return Err(WgpuPreparedKernelRouterError::CandidateNotQualified { candidate_id });
            }
            if !seen.insert(candidate.id) {
                return Err(WgpuPreparedKernelRouterError::DuplicateCandidate { candidate_id });
            }

            let executor = WgpuFlatAttention::with_kernel_candidate(&candidate).map_err(|source| {
                WgpuPreparedKernelRouterError::Backend {
                    candidate_id,
                    source,
                }
            })?;
            let fingerprint = executor
                .runtime_telemetry(probe_shape())
                .map_err(|source| WgpuPreparedKernelRouterError::Backend {
                    candidate_id,
                    source,
                })?
                .device;

            if let Some(expected) = &device {
                if expected != &fingerprint {
                    return Err(WgpuPreparedKernelRouterError::DeviceMismatch {
                        candidate_id,
                        expected: expected.clone(),
                        actual: fingerprint,
                    });
                }
            } else {
                device = Some(fingerprint);
            }

            routes.push(PreparedKernelRoute {
                candidate,
                executor,
            });
        }

        routes.sort_by_key(|route| route.candidate.id);
        let active = routes
            .iter()
            .position(|route| route.candidate.id == initial_candidate)
            .ok_or(WgpuPreparedKernelRouterError::InitialCandidateMissing {
                candidate_id: initial_candidate.get(),
            })?;

        Ok(Self {
            routes,
            active,
            device: device.expect("non-empty routes establish a device fingerprint"),
        })
    }

    #[must_use]
    pub const fn transition_requirements() -> WgpuPreparedKernelTransitionRequirementsV1 {
        WgpuPreparedKernelTransitionRequirementsV1::current()
    }

    #[must_use]
    pub fn device_fingerprint(&self) -> &RuntimeDeviceFingerprint {
        &self.device
    }

    #[must_use]
    pub fn current_candidate_id(&self) -> CandidateId {
        self.routes[self.active].candidate.id
    }

    #[must_use]
    pub fn prepared_candidate_ids(&self) -> Vec<CandidateId> {
        self.routes.iter().map(|route| route.candidate.id).collect()
    }

    /// Validate that `candidate` is prepared and that the exact requested
    /// realization would execute for `shape` without shape-dependent fallback.
    pub fn validate_candidate(
        &self,
        candidate: CandidateId,
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<RuntimeDispatchTelemetry, WgpuPreparedKernelRouterError> {
        let route = self.route(candidate)?;
        validate_exact_dispatch(route, shape, config)
    }

    /// Activate one already-prepared route and return the previous route identity.
    /// No WGPU compilation or queue submission happens here.
    pub fn apply_candidate(
        &mut self,
        candidate: CandidateId,
    ) -> Result<CandidateId, WgpuPreparedKernelRouterError> {
        let next = self.route_index(candidate)?;
        let previous = self.current_candidate_id();
        self.active = next;
        Ok(previous)
    }

    /// Verify the control-plane route actually stored by the router.
    #[must_use]
    pub fn verify_candidate(&self, candidate: CandidateId) -> bool {
        self.current_candidate_id() == candidate
    }

    /// Restore a previously returned candidate identity.
    pub fn restore_candidate(
        &mut self,
        previous: CandidateId,
    ) -> Result<(), WgpuPreparedKernelRouterError> {
        self.active = self.route_index(previous)?;
        Ok(())
    }

    /// Execute through the active prepared candidate with exact-candidate
    /// validation immediately before physical dispatch.
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<FlatAttentionOutput, WgpuPreparedKernelRouterError> {
        let route = &self.routes[self.active];
        validate_exact_dispatch(route, shape, config)?;
        route
            .executor
            .forward(q, k, v, shape, config)
            .map_err(|source| WgpuPreparedKernelRouterError::Backend {
                candidate_id: route.candidate.id.get(),
                source,
            })
    }

    /// Passive dispatch telemetry for the active route after exact-candidate
    /// validation. No queue submission occurs.
    pub fn runtime_telemetry(
        &self,
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<RuntimeDispatchTelemetry, WgpuPreparedKernelRouterError> {
        validate_exact_dispatch(&self.routes[self.active], shape, config)
    }

    fn route(
        &self,
        candidate: CandidateId,
    ) -> Result<&PreparedKernelRoute, WgpuPreparedKernelRouterError> {
        self.routes
            .iter()
            .find(|route| route.candidate.id == candidate)
            .ok_or(WgpuPreparedKernelRouterError::CandidateMissing {
                candidate_id: candidate.get(),
            })
    }

    fn route_index(
        &self,
        candidate: CandidateId,
    ) -> Result<usize, WgpuPreparedKernelRouterError> {
        self.routes
            .iter()
            .position(|route| route.candidate.id == candidate)
            .ok_or(WgpuPreparedKernelRouterError::CandidateMissing {
                candidate_id: candidate.get(),
            })
    }
}

fn validate_exact_dispatch(
    route: &PreparedKernelRoute,
    shape: AttentionShape,
    config: FlatAttentionConfig,
) -> Result<RuntimeDispatchTelemetry, WgpuPreparedKernelRouterError> {
    let candidate_id = route.candidate.id.get();
    let problem = AttentionProblem::from_shape(&shape, config).map_err(|error| {
        WgpuPreparedKernelRouterError::CandidateProblem {
            candidate_id,
            detail: error.to_string(),
        }
    })?;
    route.candidate.module_for(&problem).map_err(|error| {
        WgpuPreparedKernelRouterError::CandidateProblem {
            candidate_id,
            detail: error.to_string(),
        }
    })?;
    let expected = route.candidate.runtime_kernel_id().ok_or(
        WgpuPreparedKernelRouterError::CandidateHasNoRuntimeKernel { candidate_id },
    )?;
    let telemetry = route
        .executor
        .runtime_telemetry(shape)
        .map_err(|source| WgpuPreparedKernelRouterError::Backend {
            candidate_id,
            source,
        })?;
    if telemetry.kernel_id != expected {
        return Err(WgpuPreparedKernelRouterError::CandidateSubstitution {
            candidate_id,
            expected,
            actual: telemetry.kernel_id,
        });
    }
    Ok(telemetry)
}

const fn probe_shape() -> AttentionShape {
    AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 1,
        head_dim: 64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::kernel_candidates::{generate_candidates, SelectionPolicy};
    use flat_attention::{RuntimeDeviceCapabilities, WgpuSubgroupPolicy};

    fn config() -> FlatAttentionConfig {
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        }
    }

    fn fake_caps() -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65_535,
            max_workgroup_size_x: 64,
            max_workgroup_size_y: 1_024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32_768,
            max_binding_entries: 8,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_supported: false,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            f16_supported: false,
        }
    }

    #[test]
    fn transition_requirements_do_not_claim_resident_state_preservation() {
        let requirements = WgpuPreparedKernelRouter::transition_requirements();
        requirements.validate().unwrap();
        assert!(requirements.route_swap_live);
        assert!(requirements.host_io_only);
        assert!(!requirements.resident_buffers_preserved);
        assert!(!requirements.kv_state_preserved);

        let mut false_claim = requirements;
        false_claim.resident_buffers_preserved = true;
        assert!(false_claim.validate().is_err());
    }

    #[test]
    fn prepared_router_switches_exact_runtime_route_when_wgpu_is_available() {
        let bootstrap = match WgpuFlatAttention::with_subgroup_policy(WgpuSubgroupPolicy::Disable) {
            Ok(context) => context,
            Err(error) => {
                if std::env::var("FLAT_REQUIRE_WGPU").as_deref() == Ok("1") {
                    panic!("FLAT_REQUIRE_WGPU=1 but WGPU router bootstrap failed: {error}");
                }
                eprintln!("skipped: no WGPU device for prepared router test: {error}");
                return;
            }
        };
        let shape = AttentionShape {
            batch: 1,
            heads: 1,
            seq_len: 2,
            head_dim: 64,
        };
        let problem = AttentionProblem::from_shape(&shape, config()).unwrap();
        let candidates = generate_candidates(
            &problem,
            &bootstrap.device_capabilities(),
            &SelectionPolicy::default(),
        )
        .into_iter()
        .filter(|candidate| {
            matches!(
                candidate.runtime_kernel_id(),
                Some(RuntimeKernelId::Q4Portable | RuntimeKernelId::Q4Vec4Portable)
            )
        })
        .collect::<Vec<_>>();
        if candidates.len() < 2 {
            if std::env::var("FLAT_REQUIRE_WGPU").as_deref() == Ok("1") {
                panic!("required WGPU device did not expose both qualified portable routes");
            }
            eprintln!("skipped: device lacks two qualified portable WGPU routes");
            return;
        }

        let first = candidates[0].id;
        let second = candidates[1].id;
        let mut router = WgpuPreparedKernelRouter::prepare(candidates, first).unwrap();
        assert!(router.verify_candidate(first));
        let first_kernel = router.runtime_telemetry(shape, config()).unwrap().kernel_id;

        router.validate_candidate(second, shape, config()).unwrap();
        let previous = router.apply_candidate(second).unwrap();
        assert_eq!(previous, first);
        assert!(router.verify_candidate(second));
        let second_kernel = router.runtime_telemetry(shape, config()).unwrap().kernel_id;
        assert_ne!(first_kernel, second_kernel);

        let tensor_len = shape.tensor_len().unwrap();
        let q = vec![0.25_f32; tensor_len];
        let k = vec![0.5_f32; tensor_len];
        let v = vec![0.75_f32; tensor_len];
        let output = router.forward(&q, &k, &v, shape, config()).unwrap();
        assert_eq!(output.output.len(), tensor_len);
        assert_eq!(output.lse.len(), shape.lse_len().unwrap());

        router.restore_candidate(previous).unwrap();
        assert!(router.verify_candidate(first));
        assert_eq!(
            router.runtime_telemetry(shape, config()).unwrap().kernel_id,
            first_kernel
        );
    }

    #[test]
    fn host_candidate_generation_still_has_a_non_wgpu_reference_for_contract_tests() {
        let shape = probe_shape();
        let problem = AttentionProblem::from_shape(&shape, config()).unwrap();
        let candidates = generate_candidates(&problem, &fake_caps(), &SelectionPolicy::default());
        assert!(candidates.iter().any(|candidate| {
            candidate.lifecycle == CandidateLifecycle::Qualified
                && candidate.runtime_kernel_id() == Some(RuntimeKernelId::Q4Portable)
        }));
    }
}
